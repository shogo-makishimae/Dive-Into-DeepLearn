import random

"""手动实现数据分割，区别与dataset和dataloader的自动分批机制"""

def data_iter(x_data, y_data, batch_size=32):
    """通用的数据迭代器"""
    data_num = len(x_data)
    indices = list(range(data_num))
    random.shuffle(indices)  # 核心逻辑1：打乱索引
    
    for i in range(0, data_num, batch_size):
        # 核心逻辑2：分批提取
        batch_indices = indices[i : min(i + batch_size, data_num)]
        # 核心逻辑3：用 yield 返回当前批次的数据
        yield x_data[batch_indices], y_data[batch_indices]

# 使用时分别传入训练集和测试集
# for batch_x, batch_y in data_iter(x_train, y_train, batch_size=64):
#     print("训练批次形状:", batch_x.shape)

# for batch_x, batch_y in data_iter(x_test, y_test, batch_size=64):
#     print("测试批次形状:", batch_x.shape)