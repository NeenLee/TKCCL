import numpy as np

# 读取npz文件
with np.load('mf.npz') as file:
    # 遍历文件中的所有数组
    for name, array in file.items():
        print(f"{name}: {array.shape}")
        # 执行进一步的操作，例如打印数组的内容
        print(array)

