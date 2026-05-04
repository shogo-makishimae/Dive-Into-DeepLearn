import torch
import generate_data
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

