# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# File Name:        lstm.py
# Author:           wzw
# Version:          0.1
# Created:          2024/11/16
# Description:      lstm
# ------------------------------------------------------------------
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

dtype = torch.FloatTensor
# 句子
sentences = ["床 前 明 月 光", "疑 是 地 上 霜", "举 头 望 明 月","低 头 思 故 乡"]
# 分割句子
word_list = " ".join(sentences).split()
vocab = list(set(word_list))
# 映射
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for i, w in enumerate(vocab)}
# 单词数量
n_class = len(vocab)

# TextRNN Parameter
batch_size = 3
n_step = 4 # number of cells(= number of Step)
n_hidden = 5  # number of hidden units in one cell


def make_data(sentences):
    input_batch = []
    target_batch = []
    embedding = torch.nn.Embedding(n_class, 17)
    for sen in sentences:
        word = sen.split()
        # 单词索引
        input = [word2idx[n] for n in word[:-1]]
        input=torch.tensor(input)
        target = word2idx[word[-1]]
        input_batch.append(embedding(input))
        target_batch.append(target)

    return input_batch, target_batch


input_batch, target_batch = make_data(sentences)
# 设置batch size
input_batch, target_batch = torch.Tensor([item.cpu().detach().numpy() for item in input_batch]), torch.LongTensor(target_batch)
dataset = Data.TensorDataset(input_batch, target_batch)
loader = Data.DataLoader(dataset, batch_size, True)


class TextRNN(nn.Module):
    def __init__(self,embedding_shape):
        super(TextRNN, self).__init__()
        self.lstm = nn.LSTM(input_size=embedding_shape, hidden_size=n_hidden,num_layers=1)
        self.fc = nn.Linear(n_hidden, n_class)

    def forward(self, X,hc):
        # X:shape = (seq_len,batch_size,dim)
        X=X.transpose(0,1)
        # out:shape=(seq_len,batch_size, num_directions(=1) * n_hidden)
        out, hc_output= self.lstm(X,(hc[0],hc[1]))
        # hidden :shape= (num_layers(=1) * num_directions(=1), batch_size, n_hidden)
        # cell:shape=(num_layers(=1) * num_directions(=1), batch_size, n_hidden)
        print("out shape",out.shape)
        print("Lstm last step",out[-1])
        print("hidden layer",hc_output[0])
        print("cell layer",hc_output[1])
        # 最后一步的输出
        out = out[-1]
        model = self.fc(out)
        return model


model = TextRNN(17)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(1000):
    for x, y in loader:
        hidden = torch.zeros(1, x.shape[0], n_hidden)
        cell=torch.zeros(1, x.shape[0], n_hidden)
        # input_shape = [时间步数, 批量大小, 特征维度]
        pred = model(x,[hidden,cell])
        print("pred shape",pred.shape)
        print("y shape",y.shape)
        # pred : [batch_size, n_class], y : [batch_size] (LongTensor, not one-hot)
        loss = criterion(pred, y)
        if (epoch + 1) % 100 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

input = [sen.split()[:4] for sen in sentences]
# Predict
hidden = torch.zeros(1, len(input), n_hidden)
cell = torch.zeros(1, len(input), n_hidden)
predict = model(input_batch,[hidden,cell]).data.max(1, keepdim=True)[1]
print([sen.split()[:4] for sen in sentences], '->', [idx2word[n.item()] for n in predict.squeeze()])
