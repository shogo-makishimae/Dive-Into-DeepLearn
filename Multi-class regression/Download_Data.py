import os

# 1. 在导入 kagglehub 之前，先设置环境变量
os.environ['KAGGLEHUB_CACHE'] = 'D:\python_Project_new\Multi-class regression'  # 将路径替换为你想要的下载位置

# 2. 然后再导入库并下载
import kagglehub

print("正在下载到:", os.environ.get('KAGGLEHUB_CACHE'))
path = kagglehub.dataset_download("zalando-research/fashionmnist")

print("实际文件路径:", path)