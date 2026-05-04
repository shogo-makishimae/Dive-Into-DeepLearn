import torch
import generate_data

# 根据生成的数据特征数初始化参数
num_inputs = generate_data.features.shape[1]

# 可训练参数 w（向量）和 b（标量）
w = torch.normal(0, 0.01, size=(num_inputs,), requires_grad=True)
b = torch.zeros(size=(1,), requires_grad=True)

# 注意：这里仅做参数初始化，不能对参数直接调用 backward()