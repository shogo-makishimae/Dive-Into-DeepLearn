# import torch
# # 下面代码不是自动生成图的写法（这是我自己写的）
# x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
# y1 = x**2
# y2 = x
# for i in range(len(x)):
#     # 判断元素奇偶时应使用标量值
#     if int(x[i].item()) % 2 == 0:
#         print(y1)
#         # 每次反向传播前清零/重置梯度，避免累加
#         x.grad = None
#         y1.backward(torch.ones_like(x))
#         print(x.grad)
#     else:
#         print(y2)
#         x.grad = None
#         y2.backward(torch.ones_like(x))
#         print(x.grad)

# 上面代码的问题在于：每次循环迭代时，y1 和 y2 的计算图都被覆盖了，导致反向传播时无法正确追踪梯度。



# import torch
# 下面是正确的写法，使用一个新的张量 y 来存放结果（矩阵和张量都可以）
# x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# # 创建一个空的 y 用来存放结果，保留梯度追踪能力
# # y = torch.zeros_like(x) # y = [y[0] = 0, y[1] = 0, y[2] = 0]
# y = torch.zeros(len(x),len(x))
# print("初始 y:", y)

# # 【前向传播阶段】：控制流在这里发挥作用，动态建图
# for i in range(len(x)):
#     if int(x[i].item()) % 2 == 0:
#         # y[i] = x[i] ** 2  # 如果是偶数，为 y[i] 建立平方的计算图分支
#         for j in range(len(x)):
#             y[i][j] = x[j] ** 2
#         j = 0
#     else:
#         for j in range(len(x)):
#             y[i][j] = x[j]# y[i] = x[i] # 如果是奇数，为 y[i] 建立线性映射的计算图分支
#         j = 0

# print("前向传播计算结果 y:", y) 
# # 输出应为 tensor([1., 4., 3.], grad_fn=<CopySlices>)

# # 【反向传播阶段】：沿着刚才动态建立的、高低起伏的路径一次性回传
# # y.backward(torch.ones_like(x)) # 等价于 y.backward(torch.tensor([1.0, 1.0, 1.0]))
# for i in range(len(x)):
#     for j in range(len(x)):
#         y[i][j].backward(torch.tensor(1.0), retain_graph=True)
#     print(f"反向传播求导结果y[{i}]:", x.grad)
#     x.grad = None  # 每次反向传播后清零梯度，避免累加

# # print("反向传播求导结果 x.grad:", x.grad) 
# # 输出应为 tensor([1., 4., 1.])
# # 解释：
# # x[0]=1 (奇数)，导数是 1
# # x[1]=2 (偶数)，导数是 2*x = 4
# # x[2]=3 (奇数)，导数是 1


import torch
# 这个代码是矩阵的结果，上下两个代码灵活运用
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 创建一个空的 y 用来存放结果，保留梯度追踪能力
# y = torch.zeros_like(x) # y = [y[0] = 0, y[1] = 0, y[2] = 0]
y = torch.zeros(3, 3)
print("初始 y:", y)

# 【前向传播阶段】：控制流在这里发挥作用，动态建图
for i in range(len(x)):
    if int(x[i].item()) % 2 == 0:
        y[i] = x ** 2  # 如果是偶数，为 y[i] 建立平方的计算图分支,这里是直接把 x 的值赋给 y[i]，所以 y[i] 和 x 
                         # 是同一个张量，具有相同的计算图

    else:
            y[i] = x # 如果是奇数，为 y[i] 建立线性映射的计算图分支,这里是直接把 x 的值赋给 y[i]，所以 y[i] 和 x 
                         # 是同一个张量，具有相同的计算图
            
y.sum(dim=1)[0].backward(retain_graph=True)
print("反向传播计算结果 y:", x.grad)
x.grad = None  # 每次反向传播后清零梯度，避免累加

y.sum(dim=1)[1].backward(retain_graph=True)
print("反向传播计算结果 y:", x.grad)
x.grad = None  # 每次反向传播后清零梯度，避免累加

y.sum(dim=1)[2].backward(retain_graph=True)
print("反向传播计算结果 y:", x.grad)
x.grad = None  # 每次反向传播后清零梯度，避免累加s
