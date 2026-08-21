"""
数据保存文件
"""
import torch
import numpy as np

"""
    tensor -> segms, tensor : [batch_size, in_channel, 200, 255] ? 测试的时候 batch_size 为 1
    将 tensor 存入 .segms 文件
"""
def save_to_segms(tensor, filename):
    section = tensor.squeeze(0).squeeze(0)
    x, z = section.shape
    y = 1

    with open(filename, 'w') as f:
        # 写入头部信息
        f.write(f"{x} {y} {z}\n")
        f.write("1\n")
        f.write("v\n")

        # 写入体素值 (注意遍历顺序)
        for k in range(z):
            for j in range(y):
                for i in range(x):
                    f.write(f"{section[i, k]:.6f}\n")

    print(f" 数据已写入: {filename}")

