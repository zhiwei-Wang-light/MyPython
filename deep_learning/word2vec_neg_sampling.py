# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# File Name:        word2vec_neg_sampling.py
# Author:           wzw
# Version:          0.1
# Created:          2024/10/31
# Description:      负采样技术，将多分类转换为多个二分类,减少计算量,包括一个正类的loss与多个负类的loss
# ------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from nltk.tokenize import word_tokenize
import random
# 预处理文本
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return tokens

# 创建词汇表
def build_vocab(tokens):
    counter = Counter(tokens)
    word2idx = {word: i for i, (word, _) in enumerate(counter.items())}
    idx2word = {i: word for word, i in word2idx.items()}
    return word2idx, idx2word

# 生成Skip-gram训练数据
def create_skipgram_data(tokens, word2idx, window_size=2):
    data = []
    for i in range(len(tokens)):
        target = word2idx[tokens[i]]
        context_indices = range(max(0, i - window_size), min(len(tokens), i + window_size + 1))
        for j in context_indices:
            if j != i:
                data.append((target, word2idx[tokens[j]]))
    return data

# Skip-gram模型
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, target):
        return self.embeddings(target)

# 负采样
def negative_sampling(target, vocab_size, num_neg_samples=5):
    negatives = set()
    while len(negatives) < num_neg_samples:
        neg_sample = random.randint(0, vocab_size - 1)
        if neg_sample != target:
            negatives.add(neg_sample)
    return list(negatives)

# 训练模型
def train_skipgram(text, embedding_dim=100, window_size=2, epochs=1000, learning_rate=0.001, num_neg_samples=5):
    tokens = preprocess_text(text)
    word2idx, idx2word = build_vocab(tokens)
    skipgram_data = create_skipgram_data(tokens, word2idx, window_size)

    model = SkipGram(len(word2idx), embedding_dim)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0
        for target, context in skipgram_data:
            target_tensor = torch.tensor([target], dtype=torch.long)
            context_tensor = torch.tensor([context], dtype=torch.long)
            negatives = negative_sampling(context, len(word2idx), num_neg_samples)

            model.zero_grad()
            target_embedding = model(target_tensor)  # 获取目标词的嵌入
            context_embedding = model(context_tensor)  # 获取上下文词的嵌入
            neg_embeddings = model(torch.tensor(negatives, dtype=torch.long))  # 获取负样本的嵌入

            # 正样本损失
            pos_loss = -torch.log(torch.sigmoid(torch.matmul(target_embedding, context_embedding.T)))

            # 负样本损失
            neg_loss = -torch.sum(torch.log(torch.sigmoid(-torch.matmul(target_embedding, neg_embeddings.T))+1e-4))

            loss = pos_loss + neg_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {total_loss}')

    return model, word2idx, idx2word

# 示例文本
text = "Word embeddings are a type of word representation that allows words to be represented as vectors in a continuous vector space."
model, word2idx, idx2word = train_skipgram(text)

# 获取单词向量
word_vector = model.embeddings(torch.tensor(word2idx['word'], dtype=torch.long))
print("Word vector for 'word':", word_vector.detach().numpy())
