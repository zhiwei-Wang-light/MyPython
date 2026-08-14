import torch
import torch.nn as nn
def create_1d_absolute_trainable_embeddings(n_pos_vec,dim):
    postion_embedding=nn.Embedding(n_pos_vec,dim)
    nn.init.constant_(postion_embedding.weight,0.)
    return postion_embedding
n_pos_vec=torch.Tensor([[1,0,2],[2,0,3]]).to(torch.int)
p=create_1d_absolute_trainable_embeddings(5,5)(n_pos_vec)
