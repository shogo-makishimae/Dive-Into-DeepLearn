import torch
import data
import random

# 定义一个手动分批的生成器函数
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    # 生成从 0 到 999 的索引列表
    indices = list(range(num_examples))
    random.shuffle(indices)
    
    # 每次步进 batch_size
    for i in range(0, num_examples, batch_size):
        # 获取当前批次的索引切片
        batch_indices = indices[i : min(i + batch_size, num_examples)]
        # 根据索引提取数据并 yield 返回
        yield features[batch_indices], labels[batch_indices]

# 使用生成器进行遍历读取
batch_size = 10
count = 0
for batch_X, batch_y in data_iter(batch_size, features, labels):
    print(f"生成器读取 - 特征形状: {batch_X.shape}, 标签形状: {batch_y.shape}")
    count += 1
    if count >= 3: # 仅打印前 3 个批量
        break