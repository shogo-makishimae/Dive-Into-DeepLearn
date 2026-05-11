# 1. 导入必要的库和你自己的预处理函数
from data_loader import load_fashionmnist_data
from D_preprocessing import prepare_softmax_data
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# --- 数据加载 ---
# 这里需要你提供原始的 images 和 labels 数据
# 例如，如果你是从 npz 文件加载的：
# data = np.load('fashion_mnist.npz')
# images, labels = data['x_train'], data['y_train']

# 调用你之前的预处理函数
train_loader, test_loader = prepare_softmax_data(images, labels)

# --- 模型定义 ---
class SoftmaxRegression(nn.Module):
    def __init__(self, input_dim=784, num_classes=10):
        super(SoftmaxRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out

# 实例化模型
model = SoftmaxRegression()

# --- 损失函数和优化器 ---
criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(model.parameters(), lr=0.01) 

# --- 训练循环 ---
def train_model():
    num_epochs = 100
    loss_list = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for data, target in train_loader:
            outputs = model(data)
            loss = criterion(outputs, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        loss_list.append(avg_loss)
        
        if (epoch+1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    plt.plot(loss_list)
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

# --- 评估函数 ---
def evaluate_model():
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f'\nAccuracy on test set: {accuracy:.2f}%')

# --- 主程序入口 ---
if __name__ == "__main__":
    # 请在这里加载你的原始数据

    base_path = "datasets\\zalando-research\\fashionmnist\\versions\\4"
    images, labels = load_fashionmnist_data(base_path)  
    train_loader, test_loader = prepare_softmax_data(images, labels)
    
    train_model()
    evaluate_model()