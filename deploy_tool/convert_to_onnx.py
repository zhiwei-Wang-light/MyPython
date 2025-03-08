# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# File Name:        convert_to_onnx.py
# Author:           wzw
# Version:          0.1
# Created:          2024/12/12
# Description:      转换模型到onnx
# ------------------------------------------------------------------
from models.model_vit_small_patch16_224_in21k_with_finetune_BERT_zh_onnx_fill_zero_dropout import Model
import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--hidden_dim', default=512)
parser.add_argument('--dropout', default=0.1)
parser.add_argument('--nheads', default=8)
parser.add_argument('--dim_feedforward', default=2048)
parser.add_argument("--enc_layers", default=2)
parser.add_argument("--dec_layers", default=2)
parser.add_argument("--max_length", default=20)
parser.add_argument("--max_words_length", default=30)
parser.add_argument("--cls_num", default=20)
parser.add_argument("--pre_norm", default=True)
parser.add_argument("--epoches", default=300)
parser.add_argument("--batch_size", default=64)
parser.add_argument("--lr", default=1e-4)
args = parser.parse_args()
vqa_model = Model(args)
# 需要说明是否模型测试
vqa_model.eval()
batches_features = torch.zeros((1, 20, 384))
batch_mask = torch.zeros((1, 20))
batch_d3 = torch.zeros((1, 20, 3), dtype=torch.long)
tgt_embedding = torch.zeros((1, 30, 768))
tgt_attention_mask = torch.zeros((1, 30))
dummy_input = (batches_features, batch_mask, batch_d3, tgt_embedding, tgt_attention_mask)
model_name = "demo"
torch.onnx.export(vqa_model, dummy_input, f'{model_name}_script.onnx',
                  export_params=True,
                  do_constant_folding=True,
                  input_names=['input1', 'input2', 'input3', 'input4', 'input5'],
                  output_names=["output1"],
                  dynamic_axes={'input1': {0: 'batch_size', 1: 'max_length', 2: 'dim'},
                                'input2': {0: 'batch_size', 1: 'max_length'},
                                'input3': {0: 'batch_size', 1: 'max_length', 2: 'dim'},
                                'input4': {0: 'batch_size', 1: 'max_length', 2: 'dim'},
                                'input5': {0: 'batch_size', 1: 'max_length'}},
                  verbose=False,
                  opset_version=9)
