import torch
# 生成数据集
def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声"""
    # 生成服从标准正态分布（均值为0，标准差为1）的随机样本特征 X
    X = torch.normal(0, 1, (num_examples, len(w)))
    # 计算线性部分 Xw + b
    y = torch.matmul(X, w) + b
    # 加上服从均值为0、标准差为0.01的正态分布噪声
    y += torch.normal(0, 0.01, y.shape)
    # 将 y 的形状转换为 (num_examples, 1) 的列向量并返回
    return X, y.reshape((-1, 1))

# 设定真实的权重 w、偏差 b 和样本数量
true_w = torch.tensor([2, -3.4])
true_b = 4.2
num_examples = 1000

# 生成数据集
features, labels = synthetic_data(true_w, true_b, num_examples)

