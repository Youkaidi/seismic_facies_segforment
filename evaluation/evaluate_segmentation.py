from evaluation.segmentation_metrics import runningScore
from data_process.modelmethod import Model
import numpy as np
import os
from data_process.data_process import Resize

folder_label = r"C:\Users\YouKaidi\Desktop\Result\Seismic\base\big\label"
# folder_pred = r"C:\Users\YouKaidi\Desktop\Result\Seismic\Unet_10\test1"
folder_pred = r"C:\Users\YouKaidi\Desktop\Result\Seismic\base\big\pred"

# 让 label 与 pred 文件自动按名字匹配
label_map = {os.path.splitext(f)[0]: f for f in os.listdir(folder_label)}
pred_map  = {os.path.splitext(f)[0]: f for f in os.listdir(folder_pred)}

print("label_map keys:", list(label_map.keys())[:20])
print("pred_map keys:", list(pred_map.keys())[:20])
common_keys = sorted(set(label_map.keys()) & set(pred_map.keys()))

model = Model
n_class = 6

# 用于保存每个样本的指标
pixel_acc_list = []
mean_class_acc_list = []
mean_iou_list = []
fw_iou_list = []
per_class_iou_list = []

for key in common_keys:
    # 加载一个样本
    label_path = os.path.join(folder_label, label_map[key])
    pred_path  = os.path.join(folder_pred,  pred_map[key])

    label = model.loadmodel(label_path)
    label = label.squeeze().astype(int)
    label = np.rot90(label, k=-1)

    pred = model.loadmodel(pred_path)
    pred = pred.squeeze().astype(int)
    pred = np.rot90(pred, k=-1)
    # pred = model.loadmodel_2d(pred_path)
    # pred = pred.astype(int)
    pred = Resize(pred, 255, 701)


    # 为每个样本建立新的 runningScore
    single_score = runningScore(n_class)
    single_score.update([label], [pred])

    scores, class_iou = single_score.get_scores()

    # 收集样本级指标
    pixel_acc_list.append(scores['Pixel Acc: '])
    mean_class_acc_list.append(scores['Mean Class Acc: '])
    mean_iou_list.append(scores['Mean IoU: '])
    fw_iou_list.append(scores['Freq Weighted IoU: '])
    per_class_iou_list.append(np.array(list(class_iou.values())))

# =============================================
#            计算样本平均指标
# =============================================

print("\n===== 样本级平均评价指标 =====")
print("Pixel Acc:       mean =", np.mean(pixel_acc_list),      ", std =", np.std(pixel_acc_list))
print("Mean Class Acc:  mean =", np.mean(mean_class_acc_list), ", std =", np.std(mean_class_acc_list))
print("Mean IoU:        mean =", np.mean(mean_iou_list),       ", std =", np.std(mean_iou_list))
print("FW IoU:          mean =", np.mean(fw_iou_list),         ", std =", np.std(fw_iou_list))

# 每类 IoU 平均与 std
per_class_iou_arr = np.stack(per_class_iou_list)   # shape: (N_samples, n_class)

print("\n===== 每类 IoU 的样本平均 =====")
for cls_idx in range(n_class):
    mean_val = np.nanmean(per_class_iou_arr[:, cls_idx])
    std_val  = np.nanstd(per_class_iou_arr[:, cls_idx])
    print(f"Class {cls_idx}: mean={mean_val:.4f}, std={std_val:.4f}")





