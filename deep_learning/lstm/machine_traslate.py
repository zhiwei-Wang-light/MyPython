# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# File Name:        lstm.py
# Author:           wzw
# Version:          0.1
# Created:          2024/11/16
# Description:      lstm,修改网络模型
# ------------------------------------------------------------------
import re
import numpy as np
import torch
from torch.optim import Adam
import torch.optim as optim
import torch.utils.data as Data

# 检查是否可以使用 GPU
if torch.cuda.is_available():
    device = torch.device("cuda")  # 使用 GPU
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")  # 使用 CPU
    print("CUDA is not available. Using CPU.")

lines = open('cmn.txt', encoding='utf-8').read().strip().split('\n')
words_re = re.compile(r'\w+')
MAX_LEN = 10
pairs = []
for l in lines:
    en_sent, cn_sent, _ = l.split('\t')
    pairs.append((words_re.findall(en_sent.lower()), list(cn_sent)))

# -------------------------------------------------------------------------------------------------------------------------
# 对于英文，会把全部英文都变成小写，并只保留英文的单词。
# 对于中文，为了简便起见，未做分词，按照字做了切分
# 为了后续的程序运行的更快，通过限制句子长度，和只保留部分英文单词开头的句子的方式，得到了一个较小的数据集。这样得到了一个有5508个句对的数据集。
# -------------------------------------------------------------------------------------------------------------------------
filtered_pairs = []

for x in pairs:
    if len(x[0]) < MAX_LEN and len(x[1]) < MAX_LEN and \
            x[0][0] in ('i', 'you', 'he', 'she', 'we', 'they'):
        filtered_pairs.append(x)

print("数据数量:", len(filtered_pairs))
for x in filtered_pairs[:10]: print(x)

# -------------------------------------------------------------------------------------------------------------------------
# 创建中英文的词表
# Note: 在实际的任务中，可能还需要通过<unk>（或者<oov>）特殊词来表示未在词表中出现的词。
# -------------------------------------------------------------------------------------------------------------------------
en_vocab = {}
cn_vocab = {}

# create special token for pad, begin of sentence, end of sentence
en_vocab['<pad>'], en_vocab['<bos>'], en_vocab['<eos>'] = 0, 1, 2
cn_vocab['<pad>'], cn_vocab['<bos>'], cn_vocab['<eos>'] = 0, 1, 2

en_idx, cn_idx = 3, 3
for en, cn in filtered_pairs:
    for w in en:
        if w not in en_vocab:
            en_vocab[w] = en_idx
            en_idx += 1
    for w in cn:
        if w not in cn_vocab:
            cn_vocab[w] = cn_idx
            cn_idx += 1

# print(len(list(en_vocab)))
# print(len(list(cn_vocab)))

# -------------------------------------------------------------------------------------------------------------------------
# 所有的句子都通过<pad>补充成为了长度相同的句子。
# 对于英文句子（源语言），将其反转了过来，这会带来更好的翻译的效果。
# 所创建的padded_cn_label_sents是训练过程中的预测的目标，即，每个中文的当前词去预测下一个词是什么词。
# -------------------------------------------------------------------------------------------------------------------------
padded_en_sents = []
padded_cn_sents = []
padded_cn_label_sents = []
for en, cn in filtered_pairs:
    # 填充eos与padding
    padded_en_sent = en + ['<eos>'] + ['<pad>'] * (MAX_LEN - len(en))
    padded_en_sent.reverse()
    # 填充bos,eos与padding
    padded_cn_sent = ['<bos>'] + cn + ['<eos>'] + ['<pad>'] * (MAX_LEN - len(cn))
    padded_cn_label_sent = cn + ['<eos>'] + ['<pad>'] * (MAX_LEN - len(cn) + 1)

    padded_en_sents.append([en_vocab[w] for w in padded_en_sent])
    padded_cn_sents.append([cn_vocab[w] for w in padded_cn_sent])
    padded_cn_label_sents.append([cn_vocab[w] for w in padded_cn_label_sent])

train_en_sents = np.array(padded_en_sents)
train_cn_sents = np.array(padded_cn_sents)
train_cn_label_sents = np.array(padded_cn_label_sents)

# print(train_en_sents.shape)
# print(train_cn_sents.shape)
# print(train_cn_label_sents.shape)
# -------------------------------------------------------------------------------------------------------------------------
# 网络参数
# -------------------------------------------------------------------------------------------------------------------------
embedding_size = 128
hidden_size = 256
num_encoder_lstm_layers = 1
en_vocab_size = len(list(en_vocab))
cn_vocab_size = len(list(cn_vocab))
epochs = 300
batch_size = 2048


# -------------------------------------------------------------------------------------------------------------------------
# encoder
# -------------------------------------------------------------------------------------------------------------------------
class Encoder(torch.nn.Module):
    def __init__(self, en_vocab_size, embedding_size, hidden_size, num_encoder_lstm_layers):
        super(Encoder, self).__init__()
        self.emb = torch.nn.Embedding(en_vocab_size, embedding_size).to(device)
        self.lstm = torch.nn.LSTM(input_size=embedding_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_encoder_lstm_layers)

    def forward(self, x):
        # x: shape=(batch_size,seq_len)
        x = self.emb(x)
        x = x.transpose(1, 0)
        # x: shape=(step_num,batch_size,dim)
        x, (h_n, c_n) = self.lstm(x)
        return x, (h_n, c_n)


class Decoder(torch.nn.Module):
    # cn_vocab_size 目标端的词汇表大小
    # emb_dim为词向量维度（我们将其设置与源端一样大小）
    # hidden_size 为目标端隐层维度（将其设置为与源端一样大小）
    # n_layers 网络层数（将其设置为一样大小）
    def __init__(self, cn_vocab_size, embedding_size, hidden_size, n_layers):
        super(Decoder, self).__init__()

        self.emb = torch.nn.Embedding(cn_vocab_size, embedding_size).to(device)
        self.lstm = torch.nn.LSTM(embedding_size, hidden_size, num_layers=n_layers)
        self.classify = torch.nn.Linear(hidden_size, cn_vocab_size)

    def forward(self, x, h_n, c_n):
        # x: shape=(batch_size,seq_len)
        x = self.emb(x)
        x = x.transpose(1, 0)
        # x: shape=(step_num,batch_size,dim)
        output, (h_n, c_n) = self.lstm(x, (h_n, c_n))  # 这里的lstm指定了h，c，因此其内部不会自己创建一个全为0的h，c
        # output: shape=(1 batch_size,dim)
        output = self.classify(output.squeeze())
        # output: shape=(batch_size,dim)
        return output, (h_n, c_n)  # 返回(h_n,c_n)是为了下一解码器继续使用


encoder = Encoder(en_vocab_size, embedding_size, hidden_size, num_encoder_lstm_layers).to(device)
decoder = Decoder(cn_vocab_size, embedding_size, hidden_size, num_encoder_lstm_layers).to(device)
encoder_pt = torch.load("encoder.pt", map_location=torch.device('cpu'))
decoder_pt = torch.load("decoder.pt", map_location=torch.device('cpu'))
encoder.load_state_dict(encoder_pt)
decoder.load_state_dict(decoder_pt)
# 冻结所有卷积层（将requires_grad设为False）
for param in decoder.parameters():
    param.requires_grad = False
for param in encoder.parameters():
    param.requires_grad = False
print("模型结构:", decoder)
classify_new = torch.nn.Linear(256, cn_vocab_size)

for old_layer_name, old_layer in decoder.named_children():
    if "classify" == old_layer_name and isinstance(classify_new, type(old_layer)):
        # 如果层的名称和类型匹配，替换参数
        classify_new.load_state_dict(old_layer.state_dict())
        print(f"Replaced {classify_new} with the parameters from the old module")
decoder.classify = classify_new
print("修改后的模型结构:", decoder)

optim = Adam([
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
], lr=0.001, )

criterion = torch.nn.CrossEntropyLoss()  # pad不参与损失函数的计算
for epoch in range(epochs):
    print("epoch:{}".format(epoch))

    # shuffle training data
    perm = np.random.permutation(len(train_en_sents))
    train_en_sents_shuffled = train_en_sents[perm]
    train_cn_sents_shuffled = train_cn_sents[perm]
    train_cn_label_sents_shuffled = train_cn_label_sents[perm]

    for iteration in range(train_en_sents_shuffled.shape[0] // batch_size):
        x_data = train_en_sents_shuffled[(batch_size * iteration):(batch_size * (iteration + 1))]
        sent = torch.tensor(x_data).to(device)
        en_repr, (hidden, cell) = encoder(sent)
        # print("hidden",hidden.shape)
        # print("en_repr",en_repr.shape)
        # print("en_repr shape",en_repr[-1].shape)
        x_cn_data = train_cn_sents_shuffled[(batch_size * iteration):(batch_size * (iteration + 1))]
        x_cn_label_data = train_cn_label_sents_shuffled[(batch_size * iteration):(batch_size * (iteration + 1))].astype(
            'int64')

        # # shape:  (num_layers(=1) * num_directions(=1), batch_size, n_hidden)
        # hidden = torch.zeros([1,batch_size, hidden_size])
        # cell = torch.zeros([1, batch_size, hidden_size])

        loss = torch.zeros([1]).to(device)
        # the decoder recurrent loop mentioned above
        # 一个字一个字按顺序输入
        for i in range(MAX_LEN + 2):
            cn_word = torch.tensor(x_cn_data[:, i:i + 1]).to(device)
            cn_word_label = torch.tensor(x_cn_label_data[:, i]).to(device)
            # print("cn_word",cn_word)
            logits, (hidden, cell) = decoder(cn_word, hidden, cell)
            # print("logits",logits)
            # print("cn_word_label",cn_word_label)
            step_loss = criterion(logits, cn_word_label)
            loss += step_loss

        loss = loss / (MAX_LEN + 2)
        if (iteration % 200 == 0):
            print("iter {}, loss:{}".format(iteration, loss))

        optim.zero_grad()
        loss.backward()
        optim.step()
    # torch.save(encoder.state_dict(),"encoder.pt")
    # torch.save(decoder.state_dict(), "decoder.pt")

# encoder.eval()
# decoder.eval()
# encoder_para=torch.load("encoder.pt")
# decoder_para=torch.load("decoder.pt")
# encoder.load_state_dict(encoder_para)
# decoder.load_state_dict(decoder_para)
# num_of_exampels_to_evaluate = 10
#
# indices = np.random.choice(len(train_en_sents), num_of_exampels_to_evaluate, replace=False)
# x_data = train_en_sents[indices]
# sent = torch.tensor(x_data).to(device)
# en_repr,(hidden,cell) = encoder(sent)
#
# word = np.array(
#     [[cn_vocab['<bos>']]] * num_of_exampels_to_evaluate
# )
# word = torch.tensor(word).to(device)
#
# # hidden = torch.zeros([1, num_of_exampels_to_evaluate, hidden_size])
# # cell = torch.zeros([1,num_of_exampels_to_evaluate, hidden_size])
#
# decoded_sent = []
# for i in range(MAX_LEN + 2):
#     logits, (hidden, cell) = decoder(word, hidden, cell)
#     word = torch.argmax(logits, dim=1)
#     decoded_sent.append(word.cpu().numpy())
#     word = torch.unsqueeze(word, dim=-1)
#
# results = np.stack(decoded_sent, axis=1)
# for i in range(num_of_exampels_to_evaluate):
#     en_input = " ".join(filtered_pairs[indices[i]][0])
#     ground_truth_translate = "".join(filtered_pairs[indices[i]][1])
#     model_translate = ""
#     for k in results[i]:
#         w = list(cn_vocab)[k]
#         if w != '<pad>' and w != '<eos>':
#             model_translate += w
#     print(en_input)
#     print("true: {}".format(ground_truth_translate))
#     print("pred: {}".format(model_translate))
