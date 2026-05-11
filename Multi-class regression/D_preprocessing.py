import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split # 用于划分数据集

def prepare_softmax_data(images, labels, batch_size=32, test_size=0.2):
    """
    改进版预处理：包含标准化、划分训练/测试集、以及 DataLoader 封装
    :param images: numpy数组，形状 (N, H, W) 或 (N, H, W, C)
    :param labels: numpy数组，形状 (N,)
    :param batch_size: 每次训练取多少张图
    :param test_size: 预留多少比例作为测试集 (0.2 表示 20%)
    """
    
    # --- 1. 数据标准化 (Standardization) ---
    # 先转为 float32
    images = images.astype(np.float32)
    
    # 将像素从 [0, 255] 归一化到 [0, 1]
    images = images / 255.0
    
    # 【关键改进】标准化：(x - mean) / std
    # 计算整个数据集的均值和标准差
    mean = images.mean()
    std = images.std()
    # 防止 std 为 0 的情况
    if std == 0: std = 1e-8 
    images = (images - mean) / std
    
    # --- 2. 数据展平 (Flatten) ---
    # Softmax 全连接层通常需要一维向量输入 (N, Features)
    # 如果是 CNN 则不需要这一步，保持 (N, C, H, W)
    images = images.reshape(images.shape[0], -1) 

    # --- 3. 标签处理 ---
    # 确保标签是 Long 类型，这是 PyTorch 计算交叉熵损失函数的要求
    labels = labels.astype(np.int64)

    # --- 4. 划分训练集和测试集 ---
    # 这一步对于评估模型真实效果至关重要
    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=test_size, random_state=42, stratify=labels
    )

    # --- 5. 转换为 PyTorch Tensor ---
    x_train_tensor = torch.from_numpy(x_train)
    y_train_tensor = torch.from_numpy(y_train)
    x_test_tensor = torch.from_numpy(x_test)
    y_test_tensor = torch.from_numpy(y_test)

    # --- 6. 封装为 DataLoader ---
    # 训练集需要打乱 (shuffle=True)，测试集不需要
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
