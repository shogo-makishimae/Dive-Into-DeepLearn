import numpy as np
from pathlib import Path

def load_fashionmnist_data(base_path):

    
    # --- 1. 原可视化脚本的路径拼接逻辑 ---
    raw_dir = Path(base_path) / 'raw'
    
    if not raw_dir.exists():
        raise FileNotFoundError(f"未找到 raw 数据目录: {raw_dir}")

    # --- 2. 复用原可视化脚本的读取函数逻辑 ---
    def read_idx_images(path):
        with open(path, 'rb') as f:
            magic = int.from_bytes(f.read(4), 'big')
            if magic != 2051:
                raise ValueError(f'Invalid magic number {magic} for images. Expected 2051.')
            n = int.from_bytes(f.read(4), 'big')
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')
            data = np.frombuffer(f.read(), dtype=np.uint8)
            data = data.reshape(n, rows, cols)
        return data

    def read_idx_labels(path):
        with open(path, 'rb') as f:
            magic = int.from_bytes(f.read(4), 'big')
            if magic != 2049:
                raise ValueError(f'Invalid magic number {magic} for labels. Expected 2049.')
            n = int.from_bytes(f.read(4), 'big')
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return data

    # --- 3. 构建文件路径 ---
    train_images_f = raw_dir / 'train-images-idx3-ubyte'
    train_labels_f = raw_dir / 'train-labels-idx1-ubyte'
    test_images_f = raw_dir / 't10k-images-idx3-ubyte'
    test_labels_f = raw_dir / 't10k-labels-idx1-ubyte'

    # --- 4. 检查文件是否存在 ---
    missing_files = [f for f in [train_images_f, train_labels_f] if not f.exists()]
    if missing_files:
        raise FileNotFoundError(f"缺失训练文件: {missing_files}")

    # --- 5. 执行读取 ---
    print(f"正在从 {raw_dir} 加载数据...")
    x_train = read_idx_images(train_images_f)
    y_train = read_idx_labels(train_labels_f)
    
    # 测试集存在则加载，不存在则返回 None (或者根据需要抛出异常)
    if test_images_f.exists() and test_labels_f.exists():
        x_test = read_idx_images(test_images_f)
        y_test = read_idx_labels(test_labels_f)
    else:
        print("警告: 未找到测试集文件，仅返回训练集。")
        x_test, y_test = None, None

    print(f"训练集图像形状: {x_train.shape}, 标签形状: {y_train.shape}")
    if x_test is not None:
        print(f"测试集图像形状: {x_test.shape}, 标签形状: {y_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)
