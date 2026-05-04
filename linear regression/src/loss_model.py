import torch

def squared_loss(y_hat, y):
    """
    均方损失函数
    y_hat: 模型的预测值（一维张量，如形状 [10]）
    y: 真实标签（可能需要调整形状以匹配 y_hat）
    """
    # 将真实值 y 的形状调整为和预测值 y_hat 完全一致，防止形状不匹配报错
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2