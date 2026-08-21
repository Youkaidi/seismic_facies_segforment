"""
数据处理文件：数据集处理，数据格式转换
    三维模型->二维剖面
    npy->vti
"""
import numpy as np
from pyevtk.hl import imageToVTK
import os
from skimage.transform import resize

def get_slice_y(filepath, output_dir):
    """
    把三维模型(.npy)沿y轴切成二维剖面(.npy)
    :param filepath: 模型所在路径
    :param output_dir: 输出文件夹
    :return:
    """
    data = np.load(filepath)
    # 沿着中间轴切片并保存
    num_slices = data.shape[1]  # 701
    for i in range(num_slices):
        slice_i = data[:, i, :]  # 取第 i 个剖面 (200, 255)
        np.save(output_dir + 'slice_' + str(i) + '.npy', slice_i)

def npy_to_vti(filepath, output_dir):
    """
    将.npy 文件转为 .vti 文件
    :param filepath: npy 文件所在文件夹
    :param output_dir: 输出文件夹
    :return:
    """
    os.makedirs(output_dir, exist_ok=True)

    npy_files = [f for f in os.listdir(filepath) if f.endswith('.npy')]
    npy_files.sort()

    total = 0
    for i, fname in enumerate(npy_files):
        npy_path = os.path.join(filepath, fname)
        data = np.load(npy_path).astype(np.float32)

        # 保存为 VTI 格式
        # VTK 的坐标顺序是 (x, y, z)，而 numpy 通常是 (z, y, x)
        # 所以需要调整轴的顺序
        # 这块我保持怀疑
        # imageToVTK()要求data是一个三维 np 数组，这里扩展一个维度
        # 沿着谁切，就加在哪
        data_vtk = data[:, np.newaxis, :]
        filename = "volume"

        imageToVTK(output_dir + filename + '_' + str(i), pointData={"values": data_vtk})
        total += 1

    print(f" 文件已保存到: {filepath}，共 {total} 个文件")


def save_to_segms(tensor, filename):
    """
    tensor -> segms
    :param tensor:
    :param filename:
    :return:
    """
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

def Resize(data, target_H, target_W):
    """
    调整图像到指定尺寸：如需要扩充，复制上一行（列）的数据
    :param data: 需要处理的数据
    :param target_H: 目标图像高度
    :param target_W: 目标图像宽度
    :return: 调整尺寸后的图像
    """
    H, W = data.shape
    dtype = data.dtype  # 保留原始数据类型（如int64、float32）

    # 裁剪
    if target_H < H or target_W < W:
        # 用skimage的resize，指定order=0（最近邻），preserve_range=True（保留原始值范围）
        resized = resize(
            data,
            output_shape=(target_H, target_W),
            order=0,  # 0=最近邻，保证离散值不变
            preserve_range=True,  # 不归一化，保留原始数值（如0-4、0/1）
            anti_aliasing=False  # 关闭抗锯齿（离散数据无需平滑）
        )
        # 还原为原始数据类型（避免float污染整数标签）
        return resized.astype(dtype)

    # 扩充
    else:
        pad_H = target_H - H
        pad_W = target_W - W

        # 扩充高度：复制最后一行pad_H次
        if pad_H > 0:
            last_row = data[-1:, :]  # shape=(1, W)
            data = np.concatenate([data, np.repeat(last_row, pad_H, axis=0)], axis=0)

        # 扩充宽度：复制最后一列pad_W次
        if pad_W > 0:
            last_col = data[:, -1:]  # shape=(H_new, 1)
            data = np.concatenate([data, np.repeat(last_col, pad_W, axis=1)], axis=1)
        return data

if __name__ == '__main__':
    filepath = '../data/f3_model/test1_seismic.npy'
    output_dir = '../data/spilt_data_2d/test1_seismic/'
    get_slice_y(filepath, output_dir)
    # npy_to_vti(filepath, output_dir)

    # filepath = './data/2d_spilt_data/'
    # npy_files = [f for f in os.listdir(filepath) if f.endswith('.npy')]

    print("cuk")

