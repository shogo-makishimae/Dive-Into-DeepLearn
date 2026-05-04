import torch

def linreg(X, w, b):
    """
    线性回归模型的前向传播函数
    X: 输入特征 (形状如 [10, 2])
    w: 权重 (形状如 [2,])
    b: 偏置 (形状如 [1,])
    """
    return torch.matmul(X, w) + b