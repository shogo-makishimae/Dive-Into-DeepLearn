from torchvision import datasets

# 你只需要指定一个路径，比如 "./data"
# 这里的 "./data" 就是 root
train_dataset = datasets.FashionMNIST(
    root="./autoAI/data",   # 当前目录中切换到 autoAI/data    
    train=True,
    download=False,      # 已经下载好了，不需要再下载了    
    transform=None
)

# --- 可视化
import os
import numpy as np
import matplotlib.pyplot as plt

out_dir = os.path.join(os.path.dirname(__file__), "visualizations")
os.makedirs(out_dir, exist_ok=True)

# 类别名称（FashionMNIST 官方顺序）
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# 尝试直接使用 dataset 属性（如果存在），否则回退到索引访问
def get_sample_data(dataset, idx):
    if hasattr(dataset, 'data'):
        img = dataset.data[idx]
        if isinstance(img, np.ndarray):
            arr = img
        else:
            arr = img.numpy()
        label = int(dataset.targets[idx]) if hasattr(dataset, 'targets') else int(dataset[idx][1])
        return arr, label
    else:
        img, label = dataset[idx]
        return np.array(img), int(label)

# 保存前 9 张样例图
fig, axes = plt.subplots(3, 3, figsize=(6, 6))
for i, ax in enumerate(axes.flatten()):
    arr, lbl = get_sample_data(train_dataset, i)
    ax.imshow(arr, cmap='gray')
    ax.set_title(class_names[lbl])
    ax.axis('off')
sample_path = os.path.join(out_dir, 'samples_grid.png')
plt.tight_layout()
plt.savefig(sample_path, dpi=150)
plt.close(fig)

# 绘制类别分布直方图（使用 targets 或遍历标签）
if hasattr(train_dataset, 'targets'):
    labels = np.array(train_dataset.targets)
else:
    labels = np.array([int(train_dataset[i][1]) for i in range(len(train_dataset))])

counts = np.bincount(labels, minlength=len(class_names))
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.bar(range(len(class_names)), counts, color='tab:blue')
ax2.set_xticks(range(len(class_names)))
ax2.set_xticklabels(class_names, rotation=45, ha='right')
ax2.set_ylabel('Count')
ax2.set_title('Class distribution')
dist_path = os.path.join(out_dir, 'class_distribution.png')
plt.tight_layout()
plt.savefig(dist_path, dpi=150)
plt.close(fig2)

print('Saved visualizations:')
print(' -', sample_path)
print(' -', dist_path)

# --- 终端数据结构摘要
def summarize_dataset(dataset, labels, class_names=None):
    print('\n--- Dataset summary ---')
    # images
    images_available = hasattr(dataset, 'data')
    try:
        if images_available:
            imgs = np.array(dataset.data)
        else:
            # 回退：仅检查第一个样本的形状并不遍历全部以节省时间
            sample_img, _ = get_sample_data(dataset, 0)
            imgs = np.empty((len(dataset),) + sample_img.shape, dtype=sample_img.dtype)
    except Exception:
        imgs = None

    if imgs is not None:
        print('images shape:', imgs.shape)
        print('images dtype:', imgs.dtype)
        try:
            print('pixel min/max:', int(imgs.min()), int(imgs.max()))
            print('pixel mean/std:', float(imgs.mean()), float(imgs.std()))
        except Exception:
            pass
        try:
            print(f'images memory: {imgs.nbytes / (1024**2):.2f} MB')
        except Exception:
            pass
    else:
        print('images: not materialized')

    # labels
    labels = np.array(labels)
    print('labels shape:', labels.shape)
    print('labels dtype:', labels.dtype)
    uniq = np.unique(labels)
    print('unique labels:', uniq)
    from collections import Counter
    cnt = Counter(labels.tolist())
    print('label counts (sample):', dict(list(cnt.items())[:20]))
    labels_ok = np.array_equal(uniq, np.arange(np.max(labels) + 1))
    print('labels consecutive 0..K-1?:', labels_ok)
    if imgs is not None:
        print('flattened sample shape (for softmax input):', (imgs.shape[0], imgs.shape[1] * imgs.shape[2]))

    # 前若干标签与类别分布（fraction）
    try:
        preview_n = min(10, labels.shape[0])
        print('first labels:', labels[:preview_n].tolist())
    except Exception:
        pass
    try:
        total = float(len(labels))
        pct = {int(k): f"{v/total:.3f}" for k, v in cnt.items()}
        print('label distribution (fraction):', pct)
    except Exception:
        pass


# 执行摘要打印
summarize_dataset(train_dataset, labels, class_names=class_names)
