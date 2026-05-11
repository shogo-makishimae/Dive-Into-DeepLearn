import numpy as np
from pathlib import Path    
from data_loader import load_fashionmnist_data


# --- 使用示例 ---
if __name__ == "__main__":
    # 1. 设置 base_path，对应原可视化脚本中的 base
    # 请将此路径修改为你本地实际存放数据的路径
    DATA_ROOT = "datasets/zalando-research/fashionmnist/versions/4"
    
    # 2. 调用函数 (此时函数内部会自动处理 raw_dir = base_path / 'raw')
    (images_train, labels_train), (images_test, labels_test) = load_fashionmnist_data(DATA_ROOT)
    
    # 3. 后续即可传入预处理函数
    # train_loader, test_loader = prepare_softmax_data(images_train, labels_train)