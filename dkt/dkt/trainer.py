import math
import os

import numpy as np
import torch
import wandb

from .criterion import get_criterion
from .dataloader import get_loaders
from .metric import get_metric
from .model import LSTM, LSTMATTN, GRUATTN, Bert, Saint
from .optimizer import get_optimizer
from .scheduler import get_scheduler


def run(args, train_data, valid_data):
    train_loader, valid_loader = get_loaders(args, train_data, valid_data)

    # only when using warmup scheduler
    args.total_steps = int(math.ceil(len(train_loader.dataset) / args.batch_size)) * (
        args.n_epochs
    )
    args.warmup_steps = args.total_steps // 10

    model = get_model(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    best_auc = -1
    early_stopping_counter = 0
    for epoch in range(args.n_epochs):

        print(f"Start Training: Epoch {epoch + 1}")

        ### TRAIN
        train_auc, train_acc, train_loss = train(
            train_loader, model, optimizer, scheduler, args
        )

        ### VALID
        auc, acc = validate(valid_loader, model, args)

        ### TODO: model save or early stopping
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_auc": train_auc,
                "train_acc": train_acc,
                "valid_auc": auc,
                "valid_acc": acc,
            }
        )
        if auc > best_auc:
            best_auc = auc
            # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
            model_to_save = model.module if hasattr(model, "module") else model
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model_to_save.state_dict(),
                },
                args.model_dir,
                "model.pt",
            )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                print(
                    f"EarlyStopping counter: {early_stopping_counter} out of {args.patience}"
                )
                break

        # scheduler
        if args.scheduler == "plateau":
            scheduler.step(best_auc)


def train(train_loader, model, optimizer, scheduler, args):
    model.train()

    total_preds = []
    total_targets = []
    losses = []
    for step, batch in enumerate(train_loader):
        input, targets = process_batch(batch, args)

        preds = model(input)
        #targets = input[-1]  # answerCode

        loss = compute_loss(preds, targets)
        update_params(loss, model, optimizer, scheduler, args)

        if step % args.log_steps == 0:
            print(f"Training steps: {step} Loss: {str(loss.item())}")

        # predictions
        preds = preds[:, -1]
        targets = targets[:, -1]

        if args.gpu == "gpu":
            preds = preds.to("cpu").detach().numpy()
            targets = targets.to("cpu").detach().numpy()
        else:  # cpu
            preds = preds.to("cpu").detach().numpy()
            targets = targets.to("cpu").detach().numpy()

        total_preds.append(preds)
        total_targets.append(targets)
        losses.append(loss)

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)
    loss_avg = sum(losses) / len(losses)
    print(f"TRAIN AUC : {auc} ACC : {acc}")
    return auc, acc, loss_avg


def validate(valid_loader, model, args):
    model.eval()

    total_preds = []
    total_targets = []
    for step, batch in enumerate(valid_loader):
        input, targets = process_batch(batch, args)

        preds = model(input)
        #targets = input[-1]  # answerCode

        # predictions
        preds = preds[:, -1]
        targets = targets[:, -1]

        if args.gpu == "gpu":
            preds = preds.to("cpu").detach().numpy()
            targets = targets.to("cpu").detach().numpy()
        else:  # cpu
            preds = preds.to("cpu").detach().numpy()
            targets = targets.to("cpu").detach().numpy()

        total_preds.append(preds)
        total_targets.append(targets)

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)

    print(f"VALID AUC : {auc} ACC : {acc}\n")

    return auc, acc


def inference(args, test_data):

    model = load_model(args)
    model.eval()
    _, test_loader = get_loaders(args, None, test_data)

    total_preds = []

    for step, batch in enumerate(test_loader):
        input, targets = process_batch(batch, args)

        preds = model(input)

        # predictions
        preds = preds[:, -1]

        if args.gpu == "gpu":
            preds = preds.to("cpu").detach().numpy()
        else:  # cpu
            preds = preds.to("cpu").detach().numpy()

        total_preds += list(preds)

    write_path = os.path.join(args.output_dir, "submission.csv")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(write_path, "w", encoding="utf8") as w:
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write("{},{}\n".format(id, p))


def get_model(args):
    """
    Load model and move tensors to a given devices.
    """
    if args.model == "lstm":
        model = LSTM(args)
    if args.model == "lstmattn":
        model = LSTMATTN(args)
    if args.model == "gruattn":
        model = GRUATTN(args)
    if args.model == "bert":
        model = Bert(args)
    if args.model == "saint":
        model = Saint(args)

    model.to(args.device)

    return model


# 배치 전처리
def process_batch(batch, args):
    """ 
    dataloader에서 input에 맞춰 전처리를 한번 더 해준다. 
    (type변경, gpu설정, mask 사용하여 padding 적용)
    
    Parameters:
    batch(dtype=tuple): categorical features + continuous features + answerCode + mask
        
    Returns:
    data(dtype=tuple): categorical features + concatenated continuous features + mask 포함
                       (categorical features: dtype=int64,
                        continuous features, mask: dtype=FloatTensor)
    answerCode(dtype=FloatTensor): Target
    """

    #test, question, tag, correct, mask = batch
    data = list(batch[:len(args.cate_cols)])
    cont_data = list(batch[len(args.cate_cols) : len(args.cate_cols)+len(args.cont_cols)])
    correct = batch[-2]
    mask = batch[-1]

    # change to float
    mask = mask.type(torch.FloatTensor)
    correct = correct.type(torch.FloatTensor)

    # categorical features
    for i, col in enumerate(data):
        data[i] = ((col + 1) * mask).to(torch.int64).to(args.device)

    # Concatenate the continuous features
    concat = (cont_data[0] * mask).view(-1, args.max_seq_len, 1)
    for i in range(1, len(cont_data)):
        tmp = cont_data[i] * mask
        concat = torch.cat((concat, tmp.view(-1, args.max_seq_len, 1)), dim=2)

    data.append(concat.type(torch.FloatTensor).to(args.device))    
    data.append(mask.to(args.device))
    correct = correct.to(args.device)
    
    return tuple(data), correct 


# loss계산하고 parameter update!
def compute_loss(preds, targets):
    """
    Args :
        preds   : (batch_size, max_seq_len)
        targets : (batch_size, max_seq_len)

    """
    loss = get_criterion(preds, targets)
    # 마지막 시퀀드에 대한 값만 loss 계산
    loss = loss[:, -1]
    loss = torch.mean(loss)
    return loss


def update_params(loss, model, optimizer, scheduler, args):
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    if args.scheduler == "linear_warmup":
        scheduler.step()
    optimizer.step()
    optimizer.zero_grad()


def save_checkpoint(state, model_dir, model_filename):
    print("saving model ...")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(state, os.path.join(model_dir, model_filename))


def load_model(args):

    model_path = os.path.join(args.model_dir, args.model_name)
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # load model state
    model.load_state_dict(load_state["state_dict"], strict=True)

    print("Loading Model from:", model_path, "...Finished.")
    return model
