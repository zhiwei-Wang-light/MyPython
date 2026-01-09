# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# File Name:        infoNCE_loss.py
# Author:           wzw
# Version:          0.1
# Created:          2025/12/16
# Description:      simclr的loss
# ------------------------------------------------------------------
import torch
import torch.nn.functional as F
def simclr_info_nce(z, temperature=0.07):
    """
    z: (2N, D) 已经过 encoder + projection head
    """
    B = z.size(0)
    assert B % 2 == 0
    N = B // 2

    # 1. normalize
    z = F.normalize(z, dim=1)

    # 2. similarity
    sim = torch.matmul(z, z.T) / temperature  # (2N, 2N)

    # 3. mask self
    mask = torch.eye(B, device=z.device).bool()
    sim = sim.masked_fill(mask, -1e9)

    # 4. 正样本索引
    pos_idx = torch.arange(B, device=z.device)
    pos_idx = (pos_idx + N) % B

    # 5. InfoNCE
    loss = F.cross_entropy(sim, pos_idx)
    return loss
embedding=torch.rand((2*512,512))
loss=simclr_info_nce(embedding, temperature=0.07)
print(loss)