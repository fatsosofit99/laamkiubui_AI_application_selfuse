import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor)->Tuple[Tensor, Tensor]:
        #TODO
        K_T = K.transpose(-1,-2) #交换最后两个维度，即行列维度，实现转置
        d_model = Q.size(-1) #理论上应该匹配Q和K.T分别行与列是否序列维度是否相等上再取d_model,
        scale = torch.matmul(Q,K_T)/math.sqrt(d_model) #缩放:维持数据稳定
        #跳过mask
        output1 = self.softmax(scale)
        output2 = torch.matmul(output1,V)
        return output1,output2 #返回元组

class SelfAttention(nn.Module):
    def __init__(self, d_model: int=768):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        
    def forward(self, X: Tensor)->Tuple[Tensor, Tensor]:
        #TODO
        #矩阵变换
        Q,K,V = self.W_Q(X),self.W_K(X),self.W_V(X)
        return ScaledDotProductAttention()(Q,K,V)
        

if __name__ == '__main__':
    X = torch.rand((3, 4, 5)) # (batch_size, seq, hidden_dim)
    self_attn = SelfAttention(X.size(2))
    attn, output = self_attn(X)
    print(f'attention score shape: {attn.shape}') # torch.Size([3, 4, 4])
    print(f'output shape: {output.shape}') # torch.Size([3, 4, 5])