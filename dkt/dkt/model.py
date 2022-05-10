import torch
import torch.nn as nn

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel
except:
    from transformers.models.bert.modeling_bert import (BertConfig,
                                                        BertEncoder, BertModel)


def comb_layers(args):
    # categorical features
    cate_embedding_layers = []

    for col in args.cate_cols:
        # categorical features에서 padding을 추가 했기 때문에 갯수를 +1씩 해줌
        cate_embedding_layers.append(nn.Embedding(args.num_emb[col]+1, args.hidden_dim//3).to(args.device))

    # embedding combination projection
    cate_proj = nn.Sequential(nn.Linear((len(args.cate_cols) * (args.hidden_dim//3)), args.hidden_dim),
                              nn.LayerNorm(args.hidden_dim),
                              ).to(args.device)

    # continous features
    cont_proj = nn.Sequential(nn.Linear((len(args.cont_cols)), args.hidden_dim),
                              nn.LayerNorm(args.hidden_dim),
                              ).to(args.device)    

    # concatenate all features
    comb_proj = nn.Sequential(nn.ReLU(),
                              nn.Linear(args.hidden_dim*2, args.hidden_dim),
                              nn.LayerNorm(args.hidden_dim),
                              ).to(args.device)

    return cate_embedding_layers, cate_proj, cont_proj, comb_proj


def forward_comb(args, input, cate_embedding_layers, cate_proj, cont_bn, cont_proj, comb_proj):
    batch_size = input[0].size(0)
    n_cate, n_cont = len(args.cate_cols), len(args.cont_cols)
    mask = input[-1]
    cate_x = []

    # categorical features        
    for i in range(n_cate):
        cate_x.append(cate_embedding_layers[i](input[i]))

    cate_embed = torch.cat(cate_x, 2)
    cate_embed_x = cate_proj(cate_embed)

    # continuous features
    cont_x = input[n_cate]
    cont_bn_x = cont_bn(cont_x.view(-1, n_cont))
    cont_bn_x = cont_bn_x.view(batch_size, -1, n_cont)
    cont_embed_x = cont_proj(cont_bn_x)

    seq_emb = torch.cat([cate_embed_x, cont_embed_x], 2)
    X = comb_proj(seq_emb)
    
    return X, batch_size, mask 


class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.device = args.device
        self.drop_out = self.args.drop_out

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        self.cont_bn = nn.BatchNorm1d(len(self.args.cont_cols))

        self.cate_embedding_layers, self.cate_proj, self.cont_proj, self.comb_proj = \
            comb_layers(self.args)

        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True, dropout=self.drop_out
        )

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):
        X, batch_size, mask = forward_comb(self.args, 
                                           input, 
                                           self.cate_embedding_layers, 
                                           self.cate_proj, 
                                           self.cont_bn, 
                                           self.cont_proj, 
                                           self.comb_proj)

        # LSTM model
        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds


class LSTMATTN(nn.Module):
    def __init__(self, args):
        super(LSTMATTN, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out

        self.cont_bn = nn.BatchNorm1d(len(self.args.cont_cols))

        self.cate_embedding_layers, self.cate_proj, self.cont_proj, self.comb_proj = \
            comb_layers(self.args)

        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )

        self.config = BertConfig(
            3,  # not used
            hidden_dim=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):
        X, batch_size, mask = forward_comb(self.args, 
                                           input, 
                                           self.cate_embedding_layers, 
                                           self.cate_proj, 
                                           self.cont_bn, 
                                           self.cont_proj, 
                                           self.comb_proj)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers

        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]

        out = self.fc(sequence_output)

        preds = self.activation(out).view(batch_size, -1)

        return preds


class Bert(nn.Module):
    def __init__(self, args):
        super(Bert, self).__init__()
        self.args = args
        self.device = args.device

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        self.cont_bn = nn.BatchNorm1d(len(self.args.cont_cols))

        self.cate_embedding_layers, self.cate_proj, self.cont_proj, self.comb_proj = \
            comb_layers(self.args)

        # Bert config
        self.config = BertConfig(
            3,  # not used
            hidden_dim=self.hidden_dim,
            num_hidden_layers=self.args.n_layers,
            num_attention_heads=self.args.n_heads,
            max_position_embeddings=self.args.max_seq_len,
        )

        # Defining the layers
        # Bert Layer
        self.encoder = BertModel(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def forward(self, input):
        X, batch_size, mask = forward_comb(self.args, 
                                           input, 
                                           self.cate_embedding_layers, 
                                           self.cate_proj, 
                                           self.cont_bn, 
                                           self.cont_proj, 
                                           self.comb_proj)

        # Bert
        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0]

        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds
