import random
import numpy as np
from data_process.modelmethod import Model
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from scipy.interpolate import UnivariateSpline

def hard_constrained(pred, H_begin, H_end, W_begin, W_end):
    """
    对预测结果硬约束(纵向)
    :param pred: 网络预测结果（numpy）
    :return: 硬约束处理后的结果(numpy)
    """
    H, W = pred.shape
    for i in range(W):
        pred[0][i] = 0
    for i in range(W_begin,W_end):
        for j in range(H_begin,H_end- 1):
            m = int(pred[j][i])
            n = int(pred[j + 1][i])
            if (T[m][n] != 1):
                continue
            else:
                pred[j + 1][i] = m
    return pred


def majority_filter_numpy(pred, H_begin, W_begin,window_size=(5,5), min_count=None):
    """
    滤波器去除离散点
    :param pred: 预测的地层模型
    :param window_size: 窗口大小
    :param min_count: 异常像素阈值
    :return: 滤波后的地层模型
    """
    H, W = pred.shape
    kh, kw = window_size
    if min_count is None:
        min_count = (kh*kw)//2 + 1

    pad_h = kh // 2
    pad_w = kw // 2
    padded = np.pad(pred, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=-9999)
    views = sliding_window_view(padded, (kh, kw))  # shape [H, W, kh, kw]
    out = pred.copy()
    classes = np.unique(pred)
    # For each pixel, compute counts of each class and decide majority
    for i in range(H_begin,H):
        for j in range(W_begin,W):
            window = views[i, j].ravel()
            # ignore padded sentinel
            window = window[window != -9999]
            # compute mode
            vals, counts = np.unique(window, return_counts=True)
            maj_idx = np.argmax(counts)
            maj_val = vals[maj_idx]
            if counts[maj_idx] >= min_count and maj_val != pred[i,j]:
                out[i,j] = maj_val
    return out

def single_filter(pred, H_begin, H_end,W_begin,W_end, num):
    pred_temp = pred.copy()
    H, W = pred.shape
    for i in range(H_begin,H_end):
        for j in range(W_begin, W_end):
            if pred_temp[i][j] == num: pred_temp[i][j] = 4
    return pred_temp

def boundary_process(pred):
    """
    提取地层边界
    :param pred: 预测的地层模型
    :return:
    """
    H, W = pred.shape
    boundary_0 = []
    boundary_1 = []
    boundary_2 = []
    boundary_3 = []
    boundary_4 = []
    for i in range(W):
        b0_i, b1_i, b2_i, b3_i, b4_i = (True,) * 5
        for j in range(H-1):
            m = int(pred[j][i])
            n = int(pred[j + 1][i])
            if m!=n and (n - m)==1:
                if m == 0 and n == 1 and b0_i:
                    boundary_0.append([j, i])
                    b0_i = False
                elif m==1 and n==2 and b1_i:
                    boundary_1.append([j, i])
                    b1_i = False
                elif m==2 and n==3 and b2_i:
                    boundary_2.append([j, i])
                    b2_i = False
                elif m==3 and n==4 and b3_i:
                    boundary_3.append([j, i])
                    b3_i= False
                elif m==4 and n==5 and b4_i:
                    boundary_4.append([j, i])
                    b4_i = False
                else:
                    continue
    return [
        np.array(boundary_0),
        np.array(boundary_1),
        np.array(boundary_2),
        np.array(boundary_3),
        np.array(boundary_4)
    ]

def RBF_interpolation(boundary_list, pred, step=5, value = 4):
    """
    径向基函数插值
    :param boundary_list: 已知点序列
    :param pred:  待处理的地层模型
    :param step: 采样步长
    :return:
    """
    H,W = pred.shape
    pred_temp = pred.copy()
    boundary_list_sampled = boundary_list[::step]  # 根据步数采样
    x = []
    y = []
    # 提取x,y轴坐标值
    for i in range(len(boundary_list_sampled)):
        x.append(boundary_list_sampled[i][0])
        y.append(boundary_list_sampled[i][1])
    x_numpy = np.array(y)
    y_numpy = np.array(x)
    rbf = Rbf(x_numpy,y_numpy, function='multiquadric', epsilon=1,smooth=0.01)
    xx = np.arange(0,W)
    boundary_list_smooth = rbf(xx)  # 插值
    boundary_list_smooth = boundary_list_smooth.astype(int) # 取整

    for i in range(W):
        m = 255
        # 检索原边界位置，若有则赋值给m,若无则m置为255
        for y_temp in range(len(boundary_list)):
            if boundary_list[y_temp][1]==i:
                m = boundary_list[y_temp][0]
                break

        n = boundary_list_smooth[i]  # 插值后的边界位置
        if n > 255: n=255
        if m < n:
            for j in range(m, n):
                pred_temp[j][i] = value-1
            # pred_temp[n][i] = value - 1
        elif m > n:
            for j in range(n, m):
                pred_temp[j][i] = value
            pred_temp[m][i] = value
    return pred_temp

def RBF_interpolation_2(boundary_list, pred, step=5, value = 4):
    """
    径向基函数插值
    :param boundary_list: 已知点序列
    :param pred:  待处理的地层模型
    :param step: 采样步长
    :return:
    """
    H, W = pred.shape
    pred_temp = pred.copy()

    # 如果 boundary_list 是空的，直接返回
    if boundary_list is None or getattr(boundary_list, "size", 0) == 0:
        return pred_temp

    # 采样与 RBF 平滑（保留你原来的设置）
    boundary_list_sampled = boundary_list[::step]
    x = [p[0] for p in boundary_list_sampled]
    y = [p[1] for p in boundary_list_sampled]
    x_numpy = np.array(y)
    y_numpy = np.array(x)
    rbf = Rbf(x_numpy, y_numpy, function='multiquadric', epsilon=1, smooth=0.01)
    xx = np.arange(0, W)
    boundary_list_smooth = rbf(xx).astype(int)

    # 限制索引在合法范围内
    boundary_list_smooth = np.clip(boundary_list_smooth, 0, H - 1)

    # 提取已有边界列索引（整型）
    cols = np.array(boundary_list[:, 1], dtype=int) if boundary_list.size > 0 else np.array([], dtype=int)

    # 对每一列 i：如果该列已有边界（i 在 cols 中），就跳过；否则写入插值位置 n
    for i in range(W):
        n = int(boundary_list_smooth[i])
        # 若原边界存在于该列，跳过
        if cols.size > 0 and np.any(cols == i):
            continue
        # 否则写值（已经做了 clip，索引安全）
        pred_temp[n, i] = value

    return pred_temp

def RBF_interpolation_3(pred, b2, begin, end, value, depth):
    """
    处理b3断开的边界
    :param pred: 待处理的图像
    :param b2: b2界面的边界坐标
    :param begin:
    :param end:
    :param value: 填充的值
    :return:
    """
    pred_temp = pred.copy()

    for i in range(begin,end):
        x_pos = 0
        for j in range(len(b2)):
            if b2[j][1] == i: x_pos = b2[j][0]

        # n = m + random.randint(8,10)
        x_pos = x_pos + depth
        pred_temp[x_pos][i] = value

    return pred_temp


def Spline_interploation(boundary_list):
    x = []
    y = []
    for i in range(len(boundary_list)):
        x.append(boundary_list[i][0])
        y.append(boundary_list[i][1])
    x_numpy = np.array(y)
    y_numpy = np.array(x)

    spline = UnivariateSpline(x_numpy,y_numpy,k=3, s=50)
    xx = np.linspace(x_numpy.min(), x_numpy.max(), 192)
    yy = spline(xx)

    plt.scatter(y, x)
    plt.plot(xx, yy, label='RBF')
    plt.legend()

def plot_boundary_scatter(boundaries, H, W):
    """
    散点图可视化地层边界（支持原始/平滑对比）
    :param boundaries: 原始边界列表 [b0, b1, b2, b3, b4]
    :param H: 图像高度（用于设置y轴范围）
    :param W: 图像宽度（用于x轴范围）
    :param boundaries_smoothed: 平滑后边界列表（可选，传入则对比展示）
    :param plot_range: 横向展示范围（如[0, 200]，默认展示全部）
    """
    # 设置画布大小和样式
    plt.figure(figsize=(6, 8))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']  # 5种边界颜色（鲜明区分）
    boundary_names = ['C0→C1', 'C1→C2', 'C2→C3', 'C3→C4', 'C4→C5']  # 边界名称

    x_start, x_end = 0, W


    x = np.arange(x_start, x_end)  # x轴：横向列数

    # 绘制原始边界（散点）
    for idx, (boundary, color, name) in enumerate(zip(boundaries, colors, boundary_names)):
        # 筛选有效边界（排除无边界的位置，值≠H）
        valid_mask = (boundary[x_start:x_end] != H)
        if np.any(valid_mask):
            plt.scatter(
                x[valid_mask],  # 横向列数（x轴）
                boundary[x_start:x_end][valid_mask],  # 纵向位置（y轴）
                color=color,
                s=20,  # 散点大小
                alpha=0.7,  # 透明度（避免重叠遮挡）
                label=f'原始-{name}'
            )


    # 图表美化和标签设置
    plt.xlabel('横向列数（W）', fontsize=12)
    plt.ylabel('边界纵向位置（H）', fontsize=12)
    plt.title('地层边界横向分布散点图', fontsize=14, fontweight='bold')

    # y轴反转（因为图像坐标系中，y=0通常在顶部，反转后符合直观认知）
    plt.gca().invert_yaxis()
    plt.ylim(0, H)  # y轴范围：0~图像高度
    plt.xlim(x_start, x_end)  # x轴范围：设置的横向展示区间

    plt.legend(loc='best', fontsize=10)  # 图例（自动找最佳位置）
    plt.grid(True, alpha=0.3, linestyle='--')  # 网格线（辅助观察）
    plt.tight_layout()  # 自动调整布局，避免标签截断


T = np.array([[0,0,1,1,1,1],
              [1,0,0,1,1,1],
              [1,1,0,0,1,1],
              [1,1,1,0,0,0],
              [1,1,1,1,0,0],
              [1,1,1,1,1,0]])

H,W = 255,704  # 图像大小
save_path = "./hard_constrained_x_y_6.sgems"

model_path = f'./result/test1_test2_big/pred/U_Net-30-6.0000.sgems'
model = Model
model_numpy = model.loadmodel(model_path)
model_numpy = model_numpy.squeeze()
model_numpy = np.rot90(model_numpy,k=-1)
model.savemodel_data_with_position_2D(model_numpy,save_path)

# 去除部分离散点
pred_hard = majority_filter_numpy(model_numpy, H_begin=120, W_begin=0,window_size=(20,20), min_count=200)
model.savemodel_data_with_position_2D(pred_hard,save_path)

b0, b1, _, _, _ = boundary_process(pred_hard)

# b0 界面插值
pred_hard = RBF_interpolation_2(b0,pred_hard, step=1, value=1)
model.savemodel_data_with_position_2D(pred_hard,save_path)
# b1 界面插值
pred_hard = RBF_interpolation_2(b1,pred_hard, step=1, value=2)
model.savemodel_data_with_position_2D(pred_hard,save_path)

pred_hard = RBF_interpolation_3(pred_hard, b2=b1, begin=537,end=564, value=3, depth=90)
model.savemodel_data_with_position_2D(pred_hard,save_path)

_, _, b2, b3, _ = boundary_process(pred_hard)

pred_hard = RBF_interpolation_3(pred_hard, b2=b2, begin=468,end=647, value=4, depth=6)
model.savemodel_data_with_position_2D(pred_hard,save_path)

pred_hard = RBF_interpolation_3(pred_hard, b2=b3, begin=30,end=55, value=5, depth=10)
model.savemodel_data_with_position_2D(pred_hard,save_path)

pred_hard = single_filter(pred_hard, 50,100,0,700, num = 3)
model.savemodel_data_with_position_2D(pred_hard,save_path)

pred_hard = hard_constrained(pred_hard,0,256,0,704)
model.savemodel_data_with_position_2D(pred_hard,save_path)

pred_hard = single_filter(pred_hard, 0,256,200,700, num = 5)
model.savemodel_data_with_position_2D(pred_hard,save_path)
# # 提取地层边界
# b0, b1, b2, _, _ = boundary_process(pred_hard)

# # b2 界面插值
# pred_hard = RBF_interpolation_2(b2,pred_hard, step=1, value=3)
# model.savemodel_data_with_position_2D(pred_hard,save_path)
#
# # b3界面插值
# pred_hard = RBF_interpolation_3(pred_hard, b2=b2, begin=502,end=649, value=4)
# model.savemodel_data_with_position_2D(pred_hard,save_path)
#

#
# pred_hard = single_filter(pred_hard, 300,700, num = 5)
# model.savemodel_data_with_position_2D(pred_hard,save_path)
#
# pred_hard =majority_filter_numpy(pred_hard,window_size=(10,10), min_count=13)
# model.savemodel_data_with_position_2D(pred_hard,save_path)
#
# b0, b1, b2, b3, b4 = boundary_process(pred_hard)
# pred_hard = RBF_interpolation(b3,pred_hard, step=5, value=4)
# model.savemodel_data_with_position_2D(pred_hard,save_path)
#
# # pred_hard = RBF_interpolation(b4,pred_hard,step=5, value=5)
# pred_hard = majority_filter_numpy(pred_hard, min_count=13)
# model.savemodel_data_with_position_2D(pred_hard,save_path)
#
# pred_hard = majority_filter_numpy(pred_hard, min_count=13)
# model.savemodel_data_with_position_2D(pred_hard,save_path)
#
# # Spline_interploation(b3)
#
# # boundaries = [b0, b1, b2, b3, b4]
# # plot_boundary_scatter(boundaries, H, W)
# model.savemodel_data_with_position_2D(pred_hard,save_path)






