import numpy as np
import torch

"""
paraview 中是 xz，因为是沿着 y 轴切的
 tensor : [batch_size, in_channel, 200, 255]
 给 tensor 添加钻孔
"""
def get_drill_tensor(tensor):
    """
    提取条件数据（钻孔）
    :param tensor:
    :return:
    """
    # 沿着 x 轴打钻孔
    # 0 ~ 199 间，生成 20个 钻孔
    condition = torch.ones_like(tensor, dtype=torch.float32) * -1
    # 生成x轴上的20个钻孔位置 整数
    # 创建一个张量, 等间距取 20 个位置点 [start, end]
    x_positions = list(range(0, tensor.shape[2]-1, 10))
    condition[:, :, x_positions, :] = tensor[:, :, x_positions, :]
    return condition

def get_drill(section, num):
    """
    在完整剖面上均匀提取钻孔作为条件数据
    :param section: 完整剖面（numpy)
    :param num: 钻孔数量
    :return: 只保留钻孔数据的剖面（numpy）和掩码（numpy）
    """
    rows,cols = section.shape

    y_coords = np.linspace(0, rows-1,num,endpoint=True,dtype=int)
    y_coords = np.unique(y_coords)

    # 初始化剖面(全为nan)
    drill_section = np.full_like(section, -1, dtype=np.float32)
    for y in y_coords:
        drill_section[y, :] = section[y, :].astype(np.float32)

    # 生成掩码(已知区域为1，未知区域为0)
    mask = np.zeros((rows,cols),dtype=np.float32)
    for y in y_coords:
        mask[y,:] = 1.0

    return drill_section, mask

