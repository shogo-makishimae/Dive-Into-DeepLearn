import numpy as np
import pandas as pd
import torch


x = np.arange(12)
x = x.reshape(3, 4)
y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

print(".........")
z = np.concatenate((x, y), axis=0) # axis=0表示在行的方向上进行连接
print("这是z")
print(z)

z1 = np.concatenate((x, y), axis=1) # axis=1表示在列的方向上进行连接
print("这是z1")
print(z1)   


z1=(x==x) # 比较x中的每个元素是否等于它自己，结果是一个布尔数组，
            # 所有元素都是True
print("这是z1")     
print(z1)

a1 = np.arange(0,3).reshape(1,3)
a2 = np.arange(3,6).reshape(3,1)
a3 = a1 + a2 # 广播机制：a1的形状是(1,3)，a2的形状是(3,1)，
                #它们可以通过广播机制进行加法运算，结果是一个形状为(3,3)的数组。
print("这是a3") 
print(a3)

b1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12],[13, 14, 15]])

print(b1[1,1],b1[0,1],b1[2,1])
print(b1[1:2]) # 这是切片操作，b1[1:2]表示取b1的第1行到第2行（不包括第2行）
print("''''''''''''''''''")
print(b1[0:2])
print("''''''''''''''''''")
print(b1[0::2])
print("''''''''''''''''''")
print(b1[1,:]) 
print("''''''''''''''''''")
b1[2:3] = 12
print(b1)

c1 = np.ones((3, 3))
print("这是c1") 
print(c1)

d1 = pd.DataFrame({'A': [10], 'B': [5], 'C': [15]})
print("这是d1")
print(d1)
d2 = d1.to_numpy()
print("这是d2")
print(d2)   

# 下面代码报错的原因是dim是torch的参数，numpy的参数是exis，所以需要修改为axis
# f1 = np.arange(12).reshape(3, 4)
# f2 = f1.sum(dim = 1)[0]
# print(f2)
