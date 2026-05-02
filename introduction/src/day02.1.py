import torch

# x = torch.tensor([1.0, 2.0], requires_grad=True)
# y = ( x + 2 ) ** 2
# z = y.mean() # 函数的线性组合 
# print( z ) # tensor(12.5000, grad_fn=<MeanBackward0>)
# z.backward()
# print( x.grad ) # tensor([3., 4.])
print("'''''''''''''''''")
      
# y_sum = y.sum() 
# y_sum.backward()
# print( x.grad ) # tensor([6., 8.])

# 矢量解决方法一：传入一个与z同形状的张量，表示每个元素的权重
x = torch.tensor([1.0,2.0,3.0], requires_grad=True)
y = ( x + 2 ) ** 2
z = y * 4
print( z ) # tensor([ 36.,  64., 100.], grad_fn=<MulBackward0>
z.backward(torch.tensor([1.0, 1.0, 1.0])) 
print( x.grad )    

# 矢量解决方法一（特殊情况）：求和后求导
x = torch.tensor([1.0,2.0,3.0], requires_grad=True)
y = ( x + 2 ) ** 2
z = y * 4
z_sum = z.sum() # 权重都为一的情况
z_sum.backward()
print( x.grad ) 