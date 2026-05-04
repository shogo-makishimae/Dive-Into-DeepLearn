import torch

def sgd(params, lr, batch_size):
    """
    小批量随机梯度下降优化算法
    params: 需要更新的模型参数列表，比如 [w, b]
    lr: 学习率 (learning rate)，控制每次更新的步长
    batch_size: 当前批次的大小
    """
    with torch.no_grad():  # 1. 告诉 PyTorch：更新参数时不需要计算梯度，也不需要记录计算图
        for param in params:
            # 2. 核心更新公式：参数 = 参数 - 学习率 * (参数的梯度 / 批次大小)
            # batch_size 是为了平均梯度，防止梯度过大导致更新过大
            param -= lr * param.grad / batch_size

    