from torch.utils.data import Dataset
import numpy as np
import os
from data_process.condition import get_drill

class F3Dataset(Dataset):
    # 将 root_dir下的 npy 文件读入生成数据集
    def __init__(self, root_dir):
        self.npy_files = [f for f in os.listdir(root_dir) if f.endswith('.npy')]
        imgs = []
        for fname in self.npy_files:
            npy_path = os.path.join(root_dir, fname)
            data = np.load(npy_path).astype(np.float32) # 读取npy文件为numpy数组
            imgs.append(data)

        self.imgs = imgs
        self.len = len(self.npy_files)

    def __getitem__(self, idx):
        """
        通过索引访问数据集时触发
        return:
            section:完整剖面图（labels）
            section_drill:只有条件数据的剖面图
            mask:掩码
        """
        section = self.imgs[idx]
        section_drill, mask = get_drill(section,num=5)
        return section,section_drill,mask

    def __len__(self):
        return self.len

class F3DatasetSeismic(Dataset):
    # 将 root_dir下的 npy 文件读入生成数据集
    def __init__(self, root_dir):
        self.npy_files = [f for f in os.listdir(root_dir) if f.endswith('.npy')]
        imgs = []
        for fname in self.npy_files:
            npy_path = os.path.join(root_dir, fname)
            data = np.load(npy_path).astype(np.float32) # 读取npy文件为numpy数组
            imgs.append(data)

        self.imgs = imgs
        self.len = len(self.npy_files)

    def __getitem__(self, idx):
        """
        通过索引访问数据集时触发
        return:
            section:完整剖面图（labels）
        """
        section = self.imgs[idx]
        return section

    def __len__(self):
        return self.len

# train_path = r'D:\Learning\Knowledge_Net\Code\GAN_horizon\data\spilt_data_2d\test1_label\train'
# data_train = F3Dataset(train_path)
# section = data_train[0]
# section_drill = get_drill(section,10)
# # 查看第一个样本
# print(section_drill)
