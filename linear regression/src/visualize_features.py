import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.pyplot as _plt
import numpy as _np


def plot_feature_vs_label(X, y, feature_idx, ax=None):
    if ax is None:
        ax = plt.gca()
    x = X[:, feature_idx]
    mask = np.isfinite(x) & np.isfinite(y)
    x2 = x[mask]
    y2 = y[mask]
    if x2.size == 0:
        ax.text(0.5, 0.5, 'no valid data', ha='center')
        return
    # linear fit (degree 1)
    coef = np.polyfit(x2, y2, 1)
    line_x = np.linspace(x2.min(), x2.max(), 100)
    line_y = np.polyval(coef, line_x)

    ax.scatter(x2, y2, s=20, alpha=0.6)
    ax.plot(line_x, line_y, color='red', lw=2)
    ax.set_xlabel(f'Feature {feature_idx + 1}')
    ax.set_ylabel('Label')
    ax.set_title(f'Feature {feature_idx + 1} vs Label — slope={coef[0]:.3f}')


def load_data_from_csv(path: Path):
    # try to load numeric CSV, assume last column is label
    try:
        data = np.genfromtxt(path, delimiter=',', skip_header=0)
        if data.ndim == 1:
            data = data.reshape(-1, data.size)
        if data.shape[1] < 2:
            raise ValueError('need at least one feature and one label')
        X = data[:, :-1]
        y = data[:, -1]
        return X, y
    except Exception:
        return None, None


def example_synthetic(n=200, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 2)
    # make label linearly dependent on features with noise
    true_coef = np.array([2.0, -1.0])
    y = X @ true_coef + 0.5 * rng.randn(n)
    return X, y


def main():
    p = Path('data.csv')
    X, y = (None, None)
    if p.exists():
        X, y = load_data_from_csv(p)
        if X is None:
            print('Failed to parse data.csv — will use synthetic example.')
    if X is None:
        X, y = example_synthetic()

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    plot_feature_vs_label(X, y, 0, ax=axs[0])
    if X.shape[1] > 1:
        plot_feature_vs_label(X, y, 1, ax=axs[1])
    else:
        axs[1].axis('off')

    plt.tight_layout()
    out = Path('images') / 'features_vs_label.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f'Saved plot to {out}')
    plt.show()


if __name__ == '__main__':
    main()


def plot_training_results(true_w, learned_w, losses, out_path: Path = Path('training_results.png')):
    """绘制训练结果：左图为真实权重 vs 学到的权重，右图为训练损失曲线。

    参数可为 PyTorch 张量或 NumPy 数组/序列。
    """
    tw = _np.asarray(true_w).flatten()
    lw = _np.asarray(learned_w).flatten()
    losses = _np.asarray(losses)

    n = max(tw.size, lw.size)
    idx = _np.arange(n)

    fig, axs = _plt.subplots(1, 2, figsize=(12, 4))

    width = 0.35
    axs[0].bar(idx - width/2, tw, width=width, label='true')
    axs[0].bar(idx + width/2, lw, width=width, label='learned')
    axs[0].set_xticks(idx)
    axs[0].set_xticklabels([f'w{i+1}' for i in idx])
    axs[0].set_title('True vs Learned Weights')
    axs[0].legend()

    axs[1].plot(_np.arange(1, losses.size + 1), losses, marker='o')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Avg Loss')
    axs[1].set_title('Training Loss Curve')

    _plt.tight_layout()
    out_path = Path('images') / Path(out_path).name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f'Saved training results to {out_path}')
    _plt.show()
