import torch
from generate_data import features, labels, true_w, true_b
from readdata import data_iter
from init import w, b
from model import linreg
from loss_model import squared_loss
from SGD import sgd
from visualize_features import plot_training_results

# --- 超参数设置 ---
batch_size = 10
lr = 0.03
num_epochs = 100

epoch_losses = []

# --- 训练主循环 ---
for epoch in range(num_epochs):
    total_loss = 0
    num_batches = 0

    for X, y in data_iter(batch_size, features, labels):
        # --- 1. 前向传播 ---
        y_hat = linreg(X, w, b)
        loss = squared_loss(y_hat, y).mean() # 转为标量

        # --- 2. 梯度清零 (关键：因为 SGD.py 里删掉了，这里必须加) ---
        # PyTorch 默认会累加梯度 (accumulate)，所以每次 backward 前必须手动清零
        # 否则本次梯度 = 本次计算 + 上次残留，会导致参数乱飞
        if w.grad is not None:
            w.grad.zero_()
        if b.grad is not None:
            b.grad.zero_()
        
        # --- 3. 反向传播 ---
        loss.backward() # 自动计算梯度，填入 w.grad 和 b.grad
        
        # --- 4. 参数更新 ---
        # 调用 SGD，它只做减法，不再清零
        sgd([w, b], lr, batch_size)
        
        total_loss += loss.item()
        num_batches += 1

    # --- 记录与打印 ---
    avg_loss = total_loss / num_batches
    epoch_losses.append(avg_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')

print('\n训练完成！')
print(f'真实权重 w: {true_w}')
learned_w = w.detach().numpy().reshape(-1)
print(f'学习到的 w: {learned_w}')
print(f'真实偏置 b: {true_b}')
print(f'学习到的 b: {b.item():.4f}')

# 绘图
plot_training_results(true_w.numpy(), learned_w, epoch_losses)