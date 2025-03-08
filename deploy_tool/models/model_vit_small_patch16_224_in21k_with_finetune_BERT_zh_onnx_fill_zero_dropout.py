import copy

import numpy as np
import torch
from torch import nn
from deploy_tool.models.pos_emb import AbsolutePositionalEncoding
from deploy_tool.models.transformer import build_transformer

torch.set_printoptions(profile="full")


class LineNormLayers(nn.Module):
    def __init__(self, in_model, out_model, dropout_rate=0.1):
        super(LineNormLayers, self).__init__()
        self.liner1 = nn.Linear(in_model, in_model * 2)
        self.norm1 = nn.LayerNorm(in_model * 2)
        self.dropout1 = nn.Dropout(p=dropout_rate)  # 添加Dropout层
        self.liner2 = nn.Linear(in_model * 2, out_model)
        self.norm2 = nn.LayerNorm(out_model)
        self.dropout2 = nn.Dropout(p=dropout_rate)  # 添加Dropout层
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.liner1(x)
        x = self.norm1(x)
        x = self.dropout1(x)  # 添加Dropout
        x = self.liner2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.dropout2(x)  # 添加Dropout
        return x


class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.d3_encode = nn.Embedding(num_embeddings=202, embedding_dim=128, padding_idx=201)
        # 获取嵌入层的权重参数
        embedding_weights = self.d3_encode.weight
        # 将指定位置的嵌入向量初始化为零
        embedding_weights.data[201].fill_(0)
        self.transformer = build_transformer(args)
        self.d3_pos_emb = AbsolutePositionalEncoding(128, 3)
        self.input_encoder_pos_emb = AbsolutePositionalEncoding(args.hidden_dim, args.max_length)
        self.input_decoder_pos_emb = AbsolutePositionalEncoding(args.hidden_dim, args.max_words_length)
        self.classify = nn.Linear(args.hidden_dim, args.cls_num)
        self.line1 = LineNormLayers(384, 512)
        self.line2 = LineNormLayers(384, 512)
        self.line3 = LineNormLayers(768, 512)

    def forward(self, image_embedding_paded, src_padding_mask, batch_d3, tgt_embedding, tgt_attention_mask):
        # ------------------------------------------------------- #
        # image_qurey_paded shape=(batch_size,max_length,dim)
        # ------------------------------------------------------- #
        batch_size = image_embedding_paded.shape[0]

        # ------------------------------------------------------- #
        # d3_embedding_paded shape=(batch_size,max_length,3,128)
        # ------------------------------------------------------- #
        d3_embedding_paded = self.d3_encode(batch_d3)

        # ------------------------------------------------------- #
        # d3_pos_embedding shape=(max_length,3,128)
        # ------------------------------------------------------- #
        d3_pos_embedding = self.d3_pos_emb(self.args.max_length, 3)
        d3_embedding_paded += d3_pos_embedding
        # d3_embedding_paded shape=(batch_size,max_length,3*128)
        d3_embedding_paded = d3_embedding_paded.flatten(2)
        image_embedding_paded = self.line1(image_embedding_paded)
        d3_embedding_paded = self.line2(d3_embedding_paded)

        # src_embedding shape=(barch_size,max_length,dim)
        src_embedding = image_embedding_paded + d3_embedding_paded
        # src_pos_embedding shape=(batch_size,max_length,dim)
        src_pos_embedding = self.input_encoder_pos_emb(batch_size, self.args.max_length)
        tgt_pos_embedding = self.input_decoder_pos_emb(batch_size, self.args.max_words_length)
        tgt_embedding = self.line3(tgt_embedding)
        hs = self.transformer(src_embedding.permute(1, 0, 2), src_padding_mask, tgt_attention_mask,
                              tgt_embedding.permute(1, 0, 2), src_pos_embedding.permute(1, 0, 2),
                              tgt_pos_embedding.permute(1, 0, 2))
        output = self.classify(hs.permute(1, 0, 2)[:, 0])
        return output
