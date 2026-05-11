"""
读取并检查 FashionMNIST 的原始 IDX 文件（不需要 .gz），
可视化若干样本并输出数据结构摘要，便于判断是否满足后续 softmax 训练要求。

期望数据位置：datasets/zalando-research/fashionmnist/versions/4/raw
文件名（示例）：train-images-idx3-ubyte, train-labels-idx1-ubyte
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import torch



def read_idx_images(path):
    """读取 IDX 图像文件，返回 numpy 数组 (N, H, W) dtype=uint8"""
    with open(path, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        if magic != 2051:
            raise ValueError(f'Unexpected magic number for image file: {magic}')
        n = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(n, rows, cols)
        return data


def read_idx_labels(path):
    """读取 IDX 标签文件，返回 numpy 数组 (N,) dtype=uint8"""
    with open(path, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        if magic != 2049:
            raise ValueError(f'Unexpected magic number for label file: {magic}')
        n = int.from_bytes(f.read(4), 'big')
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data


def summarize(images, labels, name='train'):
    print(f"--- {name} summary ---")
    print('images shape:', images.shape)
    print('images dtype:', images.dtype)
    print('pixel min/max:', images.min(), images.max())
    print('pixel mean/std:', float(images.mean()), float(images.std()))
    print('labels shape:', labels.shape)
    print('labels dtype:', labels.dtype)
    print('unique labels:', np.unique(labels))
    cnt = Counter(labels.tolist())
    print('label counts:', dict(cnt))
    # 是否适合 softmax：labels 是否从 0..K-1
    labels_ok = np.array_equal(np.unique(labels), np.arange(np.max(labels) + 1))
    print('labels consecutive 0..K-1?:', labels_ok)
    print('flattened sample shape (for softmax input):', (images.shape[0], images.shape[1] * images.shape[2]))


def show_samples(images, labels, n=8):
    n = min(n, images.shape[0])
    cols = min(8, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.6))
    axes = np.array(axes).reshape(-1)
    for i in range(n):
        ax = axes[i]
        ax.imshow(images[i], cmap='gray')
        ax.set_title(str(int(labels[i])))
        ax.axis('off')
    for j in range(n, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()

def main():
    base = Path('datasets/zalando-research/fashionmnist/versions/4')
    raw = base / 'raw'
    train_images_f = raw / 'train-images-idx3-ubyte'
    train_labels_f = raw / 'train-labels-idx1-ubyte'
    test_images_f = raw / 't10k-images-idx3-ubyte'
    test_labels_f = raw / 't10k-labels-idx1-ubyte'

    if not train_images_f.exists() or not train_labels_f.exists():
        raise FileNotFoundError(f'Missing train files under {raw}')

    train_images = read_idx_images(train_images_f)
    train_labels = read_idx_labels(train_labels_f)
    summarize(train_images, train_labels, name='train')
    show_samples(train_images, train_labels, n=8)

    if test_images_f.exists() and test_labels_f.exists():
        test_images = read_idx_images(test_images_f)
        test_labels = read_idx_labels(test_labels_f)
        summarize(test_images, test_labels, name='test')
        show_samples(test_images, test_labels, n=8)

    # 注意：脚本的目标是“仅读取与可视化，且不对数据做任何修改”。
    # 如果需要将数据转换为训练格式，请参考上方注释的 `to_torch_dataset` 示例并在训练脚本中执行转换。
    print('已完成读取与可视化检查（未对数据进行任何变换或保存）。')


if __name__ == '__main__':
    main()