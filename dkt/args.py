import argparse


def parse_args(mode="train"):
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")

    parser.add_argument("--device", default="cuda", type=str, help="cpu or gpu")
    parser.add_argument("--gpu", default="gpu", type=str, help="gpu or not")

    parser.add_argument(
        "--data_dir", default="/opt/ml/input/data/", type=str, help="data directory",)
    
    parser.add_argument(
        "--asset_dir", default="asset/", type=str, help="data directory"
    )

    parser.add_argument(
        "--file_name", default="train_data.csv", type=str, help="train file name"
    )

    parser.add_argument(
        "--model_dir", default="/opt/ml/input/models/", type=str, help="model directory"
    )
    parser.add_argument(
        "--model_name", default="model.pt", type=str, help="model file name"
    )

    parser.add_argument(
        "--output_dir", default="/opt/ml/input/output/", type=str, help="output directory"
    )
    parser.add_argument(
        "--test_file_name", default="test_data.csv", type=str, help="test file name"
    )

    parser.add_argument(
        "--max_seq_len", default=20, type=int, help="max sequence length"
    )
    parser.add_argument("--num_workers", default=1, type=int, help="number of workers")

    # 데이터 전처리
    parser.add_argument(
        "--cate_cols", nargs="+", type=list, 
        default=[#"assessmentItemID", 
                 #"testId", 
                 #"KnowledgeTag", 
                 #"hour", 
                 "hour_mode",                 
                 "correct_shift_-2", 
                 "correct_shift_-1", 
                 "correct_shift_1", 
                 "correct_shift_2"], 
        help='name of categorical features')
    
    parser.add_argument(
        "--cont_cols", nargs="+", type=list, 
        default=["user_acc", 
                 #"user_correct_answer", 
                 #"user_total_answer", 
                 #"future_correct", 
                 "average_content_correct", 
                 "mean_time", 
                 "time_median",
                 "correct_per_hour", 
                 "time", 
                 #"normalized_time",
                 #"relative_time", 
                 "assess_mean", 
                 "assess_sum", 
                 "tag_mean", 
                 "tag_sum", 
                 "test_mean", 
                 "test_sum"], 
        help='name of numerical features')
    parser.add_argument(
        "--num_emb", nargs="+", type=dict, default={}, help='number of categories for embedding layer')

    # 모델
    parser.add_argument(
        "--hidden_dim", default=64, type=int, help="hidden dimension size") 
    parser.add_argument("--n_layers", default=2, type=int, help="number of layers")
    parser.add_argument("--n_heads", default=2, type=int, help="number of heads")
    parser.add_argument("--drop_out", default=0.5, type=float, help="drop out rate")

    # 훈련
    parser.add_argument("--n_epochs", default=20, type=int, help="number of epochs")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")
    parser.add_argument("--clip_grad", default=10, type=int, help="clip grad")
    parser.add_argument("--patience", default=5, type=int, help="for early stopping")

    parser.add_argument(
        "--log_steps", default=50, type=int, help="print log per n steps"
    )

    ### 중요 ###
    parser.add_argument("--model", default="lstm", type=str, help="model type")
    parser.add_argument("--optimizer", default="adam", type=str, help="optimizer type")
    parser.add_argument(
        "--scheduler", default="plateau", type=str, help="scheduler type"
    )

    args = parser.parse_args()

    return args
