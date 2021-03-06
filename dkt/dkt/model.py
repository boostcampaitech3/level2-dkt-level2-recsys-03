import torch
import torch.nn as nn
import numpy as np
import math

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
    # cate_proj = nn.Sequential(nn.Linear((len(args.cate_cols) * (args.hidden_dim//3)), args.hidden_dim),
    #                           nn.LayerNorm(args.hidden_dim)).to(args.device)
    cate_proj = nn.Linear((len(args.cate_cols) * (args.hidden_dim//3)), args.hidden_dim).to(args.device)                              

    # continous features
    # cont_proj = nn.Sequential(nn.Linear((len(args.cont_cols)), args.hidden_dim),
    #                                       nn.LayerNorm(args.hidden_dim)).to(args.device)    
    cont_proj = nn.Linear((len(args.cont_cols)), args.hidden_dim).to(args.device) 

    # concatenate all features
    # comb_proj = nn.Sequential(nn.ReLU(),
    #                           nn.Linear(args.hidden_dim*2, args.hidden_dim),
    #                           nn.LayerNorm(args.hidden_dim)).to(args.device)
    comb_proj = nn.Linear(args.hidden_dim*2, args.hidden_dim).to(args.device)

    return cate_embedding_layers, cate_proj, cont_proj, comb_proj


def get_reg(args):
    return nn.Sequential(
    nn.Linear(args.hidden_dim, args.hidden_dim),
    nn.Dropout(args.drop_out), 
    nn.Linear(args.hidden_dim, args.hidden_dim),          
    nn.Linear(args.hidden_dim, 1),
    nn.Sigmoid(),
)     


def forward_comb(args, input, cate_embedding_layers, cate_proj, cont_proj, comb_proj):
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
    #cont_bn_x = cont_bn(cont_x.view(-1, n_cont))
    #cont_bn_x = cont_bn_x.view(batch_size, -1, n_cont)
    #cont_embed_x = cont_proj(cont_bn_x)
    cont_embed_x = cont_proj(cont_x)

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
        
        self.cate_embedding_layers, self.cate_proj, self.cont_proj, self.comb_proj = \
            comb_layers(self.args)

        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True, dropout=self.drop_out
        )

        # Fully connected layer
        # self.fc = nn.Linear(self.args.hidden_dim, 1)

        # self.activation = nn.Sigmoid()
        self.reg_layer = get_reg(self.args)

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
                                           self.cont_proj, 
                                           self.comb_proj)

        # LSTM model
        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        # out = self.fc(out)
        # preds = self.activation(out).view(batch_size, -1)
        preds = self.reg_layer(out).view(batch_size, -1)

        return preds

class RNNATTN(nn.Module):
    def __init__(self, args):
        super(RNNATTN, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out

        self.cate_embedding_layers, self.cate_proj, self.cont_proj, self.comb_proj = \
            comb_layers(self.args)


        if self.args.model == "lstmattn":
            self.lstm = nn.LSTM(
                self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
            )
        else:
            self.gru = nn.GRU(
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
        # self.fc = nn.Linear(self.args.hidden_dim, 1)

        # self.activation = nn.Sigmoid()
        self.reg_layer = get_reg(self.args)

    def init_hidden(self, batch_size):
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c = c.to(self.device)


        if self.args.model == "lstmattn":
            return (h, c)
        else:
            return h

    def forward(self, input):
        X, batch_size, mask = forward_comb(self.args, 
                                           input, 
                                           self.cate_embedding_layers, 
                                           self.cate_proj, 
                                           self.cont_proj, 
                                           self.comb_proj)

        hidden = self.init_hidden(batch_size)
        
        if self.args.model == "lstmattn":
            out, hidden = self.lstm(X, hidden)
        else:
            out, hidden = self.gru(X, hidden)

        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers

        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]

        # out = self.fc(sequence_output)

        # preds = self.activation(out).view(batch_size, -1)
        preds = self.reg_layer(sequence_output).view(batch_size, -1)

        return preds

class Bert(nn.Module):
    def __init__(self, args):
        super(Bert, self).__init__()
        self.args = args
        self.device = args.device

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

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
        # self.fc = nn.Linear(self.args.hidden_dim, 1)

        # self.activation = nn.Sigmoid()
        self.reg_layer = get_reg(self.args)

    def forward(self, input):
        X, batch_size, mask = forward_comb(self.args, 
                                           input, 
                                           self.cate_embedding_layers, 
                                           self.cate_proj, 
                                           self.cont_proj, 
                                           self.comb_proj)

        # Bert
        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0]

        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        # out = self.fc(out)
        # preds = self.activation(out).view(batch_size, -1)
        preds = self.reg_layer(out).view(batch_size, -1)

        return preds

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.drop_out = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.drop_out(x)

class Saint(nn.Module):
    def __init__(self, args):
        super(Saint, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        
        self.drop_out = self.args.drop_out
        
        ### Embedding 
        # ENCODER embedding
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim//3)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim//3)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim//3)
        
        # encoder combination projection
        self.enc_comb_proj = nn.Linear((self.hidden_dim//3)*3, self.hidden_dim)

        # DECODER embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//3)
        
        # decoder combination projection
        self.dec_comb_proj = nn.Linear((self.hidden_dim//3)*1, self.hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.hidden_dim, self.drop_out, self.args.max_seq_len)
        self.pos_decoder = PositionalEncoding(self.hidden_dim, self.drop_out, self.args.max_seq_len)
        

        self.transformer = nn.Transformer(
            d_model=self.hidden_dim, 
            nhead=self.args.n_heads,
            num_encoder_layers=self.args.n_layers, 
            num_decoder_layers=self.args.n_layers, 
            dim_feedforward=self.hidden_dim, 
            dropout=self.drop_out,
            activation='relu')

        self.fc = nn.Linear(self.hidden_dim, 1)
        self.activation = nn.Sigmoid()

        self.enc_mask = None
        self.dec_mask = None
        self.enc_dec_mask = None
    
    def get_mask(self, seq_len):
        mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1))

        return mask.masked_fill(mask==1, float('-inf'))

    def forward(self, input):
        test, question, tag, _, mask, interaction= input

        batch_size = interaction.size(0)
        seq_len = interaction.size(1)

        # 신나는 embedding
        # ENCODER
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed_enc = torch.cat([embed_test,
                               embed_question,
                               embed_tag], 2)

        embed_enc = self.enc_comb_proj(embed_enc)
        
        # DECODER     
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed_interaction = self.embedding_interaction(interaction)

        embed_dec = torch.cat([embed_test,
                               embed_question,
                               embed_tag,
                               embed_interaction], 2)

        embed_dec = self.dec_comb_proj(embed_dec)

        # ATTENTION MASK 생성
        # encoder하고 decoder의 mask는 가로 세로 길이가 모두 동일하여
        # 사실 이렇게 3개로 나눌 필요가 없다
        if self.enc_mask is None or self.enc_mask.size(0) != seq_len:
            self.enc_mask = self.get_mask(seq_len).to(self.device)
            
        if self.dec_mask is None or self.dec_mask.size(0) != seq_len:
            self.dec_mask = self.get_mask(seq_len).to(self.device)
            
        if self.enc_dec_mask is None or self.enc_dec_mask.size(0) != seq_len:
            self.enc_dec_mask = self.get_mask(seq_len).to(self.device)
            
  
        embed_enc = embed_enc.permute(1, 0, 2)
        embed_dec = embed_dec.permute(1, 0, 2)
        
        # Positional encoding
        embed_enc = self.pos_encoder(embed_enc)
        embed_dec = self.pos_decoder(embed_dec)
        
        out = self.transformer(embed_enc, embed_dec,
                               src_mask=self.enc_mask,
                               tgt_mask=self.dec_mask,
                               memory_mask=self.enc_dec_mask)

        out = out.permute(1, 0, 2)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)

        preds = self.activation(out).view(batch_size, -1)

        return preds

