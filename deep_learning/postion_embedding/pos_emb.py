import torch
import math


class AbsolutePositionalEncoding(torch.nn.Module):
    def __init__(self, embedding_dim, max_seq_len):
        super(AbsolutePositionalEncoding, self).__init__()

        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        # 为每个位置编码生成一组正弦和余弦函数
        positional_encoding = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0).transpose(0, 1)

        # 为每个位置编码创建一个可学习的参数
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        batch_size, seq_len, embedding_dim = x.size()
        if seq_len > self.max_seq_len:
            raise ValueError("Sequence length exceeds maximum sequence length")

        # 从可学习的参数中提取每个位置的位置编码
        positional_encoding = self.positional_encoding[:seq_len, :]
        positional_encoding = positional_encoding.expand(batch_size, -1, -1).to(x.device)

        # 将位置编码添加到输入张量中
        x = x + positional_encoding
        return x
ThreeDimensionalPositionalEncoding=AbsolutePositionalEncoding(128,3)
