def simple_generator():
    print("--- 第一次执行开始 ---")
    yield 1  # 第一次暂停，产出 1
    print("--- 恢复执行，准备第二次产出 ---")
    yield 2  # 第二次暂停，产出 2
    print("--- 恢复执行，准备第三次产出 ---")
    yield 3  # 第三次暂停，产出 3
    print("--- 函数执行结束 ---")

# 1. 调用生成器函数，此时函数体代码不会执行，只是返回一个生成器对象
gen = simple_generator()
print(f"当前 gen 的类型是：{type(gen)}") 

# 2. 使用 next() 手动触发执行
print(next(gen))  # 执行到第一个 yield，打印日志并输出 1
print(next(gen))  # 从第一个 yield 后恢复，打印日志并输出 2
print(next(gen))  # 从第二个 yield 后恢复，打印日志并输出 3
# print(next(gen)) # 再次调用会抛出 StopIteration 异常，因为没有更多的 yield 了