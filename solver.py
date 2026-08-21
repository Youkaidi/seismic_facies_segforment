import os
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

from data_process.section_dataset import F3Dataset
from data_process.section_dataset import F3DatasetSeismic
from model.U_Net import U_Net,R2U_Net,AttU_Net,R2AttU_Net
from model.Unet2D import UNet2D
from data_process.save_tool import save_to_segms
from data_process.data_process import Resize
from model.Loss import cross_entropy_loss, vertical_transition_loss, EMALoss


class Solver(object):
    # def __init__(self, config, train_loader, valid_loader, test_loader):
    def __init__(self, config):

        # 数据加载器
        # self.train_loader = train_loader
        # self.valid_loader = valid_loader
        # self.test_loader = test_loader
        self.train_path = config.train_path
        self.test_path = config.test_path

        # 设置模型参数
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        # self.criterion = torch.nn.BCELoss()  # 损失函数

        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # 训练参数
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size

        # Step size
        self.log_step = config.log_step
        self.val_step = config.val_step

        # 路径信息
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.mode = config.mode

        # 指定设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config.model_type
        self.t = config.t
        self.build_model()  #  构建网络

        # 概率矩阵
        # self.W = torch.tensor([
        #             [0,1,1,1,1,1],
        #             [0,0,1,1,1,1],
        #             [1,0,0,1,1,1],
        #             [1,1,0,0,1,1],
        #             [1,1,0,0,0,1],
        #             [1,1,1,0,0,0]
        #         ], dtype=torch.float32).to(self.device)
        # self.W = torch.tensor([
        #             [0,1,1,1,1,1],
        #             [0,0,1,1,1,1],
        #             [1,0,0,1,1,1],
        #             [1,1,0,0,1,1],
        #             [1,1,1,0,0,1],
        #             [1,1,1,1,0,0]
        #         ], dtype=torch.float32).to(self.device)
        self.W = self.build_W(6)  # 构建转移矩阵
        self.lambda_trans_max = config.lambda_trans_max
        self.warmup_epochs = config.warmup_epochs

    def build_model(self):
        """Build generator and discriminator."""
        if self.model_type == 'U_Net':
            self.unet = U_Net(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'R2U_Net':
            self.unet = R2U_Net(img_ch=self.img_ch, output_ch=self.output_ch, t=self.t)
        elif self.model_type == 'AttU_Net':
            self.unet = AttU_Net(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'R2AttU_Net':
            self.unet = R2AttU_Net(img_ch=self.img_ch, output_ch=self.output_ch, t=self.t)
        elif self.model_type == 'Unet2D':
            self.unet = UNet2D(num_classes=5)

        self.optimizer = optim.Adam(list(self.unet.parameters()),
                                    self.lr, [self.beta1, self.beta2])
        self.unet.to(self.device)

    def build_W(self,C = 6):
        transition_matrix = torch.zeros((C, C))
        # 同类别转移（连续延伸）
        for i in range(C):
            transition_matrix[i, i] = 0.8
        # 相邻层序正向转移（如C0→C1, C1→C2, C2→C3）
        for i in range(C - 1):
            transition_matrix[i+1, i] = 0.2
        # 非相邻/逆序转移（如C0→C2, C1→C0）：极低概率
        transition_matrix = transition_matrix.clamp(min=1e-6)
        return transition_matrix

    # self.print_network(self.unet, self.model_type)

    def print_network(self, model, name):
        """打印网络信息"""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def update_lr(self, g_lr, d_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = g_lr

    def reset_grad(self):
        """Zero the gradient buffers. 梯度重置"""
        self.unet.zero_grad()

    # 硬约束
    def constrained_viterbi_decode(self, logits, transition_matrix, use_T_as_prior=False, forbidden_value=-1e9):
        """
        对 logits (B, C, H, W) 的每一列(r=0..H-1)做受限 Viterbi 解码，返回 labels_pred (B, H, W)
        Args:
            logits: torch.Tensor, shape [B, C, H, W]
            transition_matrix: torch.Tensor, shape [C, C]
                - if use_T_as_prior==False: treated as mask/weight M where M[i,j]>0 means forbidden (or high cost).
                  We will convert to log-prob style by setting forbidden transitions to large negative log-prob.
                - if use_T_as_prior==True: treated as transition probability matrix T (rows i -> cols j), non-negative rows.
            use_T_as_prior: bool, whether transition_matrix is T (probabilities)
            forbidden_value: large negative number to represent -inf (default -1e9)
        Returns:
            labels_pred: torch.LongTensor of shape [B, H, W], where each column sequence is the MAP path under constraints.
        Notes:
            - Complexity: loops over B and W; for typical image sizes (B small, W~<512) this is fine.
            - Works on GPU if logits on GPU.
        """
        assert logits.dim() == 4, "logits must be [B,C,H,W]"
        device = logits.device
        B, C, H, W = logits.shape

        # 转为概率
        logp = F.log_softmax(logits, dim=1)  # [B,C,H,W]

        # prepare transition log-probs matrix A where A[i,j] = log P(next=j | curr=i)
        # 转移矩阵为条件概率矩阵
        if use_T_as_prior:
            T = transition_matrix.to(device).float()
            T = T.clamp(min=1e-12)  # 防止0
            row_sums = T.sum(dim=1, keepdim=True)
            T = T / row_sums  # 归一化
            A = torch.log(T)  # 转为对数概率
        # 转移矩阵为掩码矩阵
        else:
            # 标记禁止的转移
            M = (transition_matrix.to(device).float() > 0).to(device)
            # 允许转移的矩阵
            allowed = (M == 0).float()
            # 允许转移log(1)=0 (无惩罚)， 禁止转移forbidden_value(近似负无穷)
            A = torch.where(allowed.bool(), torch.zeros_like(allowed), torch.full_like(allowed, forbidden_value))
            # shape [C,C]

        labels_pred = torch.zeros((B, H, W), dtype=torch.long, device=device)
        for b in range(B):
            for col in range(W):
                # unary scores for this column: seq_len x C
                # shape [H, C]
                unary = logp[b, :, :, col].transpose(0, 1).contiguous()  # from [C,H] -> [H,C]

                # dp: best score ending at class k at step t
                # backpointers: store argmax prev class
                dp = torch.empty((H, C), device=device)
                backp = torch.empty((H, C), dtype=torch.long, device=device)

                # init at t=0: dp[0,k] = unary[0,k]  (no prior)
                dp[0] = unary[0]
                backp[0] = -1

                # iterate t=1..H-1
                for t in range(1, H):
                    # dp[t-1] shape [C]
                    prev_scores = dp[t - 1].unsqueeze(1)  # [C,1]
                    # transition scores: prev_scores + A (broadcast) -> [C,C], row prev, col curr
                    # candidate_scores[i->j] = prev_scores[i] + A[i,j] + unary[t,j]
                    cand = prev_scores + A  # [C, C]
                    # maximize over prev i for each current j (i: prev class)
                    best_prev_scores, best_prev_idx = cand.max(dim=0)  # both [C], maximize over rows (prev)
                    dp[t] = best_prev_scores + unary[t]  # add unary at current step
                    backp[t] = best_prev_idx

                # termination: pick best final class
                last_best_score, last_best_class = dp[-1].max(dim=0)

                # backtrack
                seq = torch.empty((H,), dtype=torch.long, device=device)
                seq[H - 1] = last_best_class
                for t in range(H - 2, -1, -1):
                    seq[t] = backp[t + 1, seq[t + 1]]

                labels_pred[b, :, col] = seq
        return labels_pred

    def horizon_processing(self, pred):
        """
        对预测结果硬约束
        :param pred: 网络预测结果（numpy）
        :return: 硬约束处理后的结果(numpy)
        """
        H,W =  pred

    def decode_ordinal(self,prob):
        """
        prob: [B,5,H,W]  为 P(y>=1)...P(y>=5) 的 sigmoid 输出
        return: [B,H,W]  0~5 的整型标签
        """
        th = (prob >= 0.5).float()  # 阈值判断每个 k 是否成立
        cnt = th.sum(dim=1)  # [B,H,W] 满足多少个 y>=k
        return cnt.long()  # = predicted label 0~5

    def robust_norm(self, x):
        mean = x.mean()
        std = x.std() + 1e-6
        return (x - mean) / std

    def train(self):
        """Train encoder, generator and discriminator."""

        # ====================================== Training ===========================================#
        # ===========================================================================================#

        # unet_path = os.path.join(self.model_path, '%s-%d-%.4f.pkl' % (
        # self.model_type, self.num_epochs, self.lr))

        # 预训练模型路径
        unet_path = r'./models/seismic/U_Net-10-0.0001_seismic.pth'
        # U-Net Train
        if os.path.isfile(unet_path):
            # 加载预训练模型
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))

        # Train for Encoder
        # 读取标签数据集
        train_data = F3Dataset(self.train_path)
        # 读取地震数据集
        # train_data = F3DatasetSeismic(self.train_path)
        for epoch in range(self.num_epochs):
            # 逐轮训练
            self.unet.train(True)  # 切换到训练模式
            epoch_loss = 0
            target_size = (192, 256)
            for i in tqdm(range(len(train_data))):
                # 读取完整剖面(标签)和条件数据(钻孔)
                section, section_condition, mask = train_data[i]
                section = Resize(section, target_size[0], target_size[1])
                section_condition = Resize(section_condition, target_size[0], target_size[1])

                # 转为tensor
                section = torch.from_numpy(section).long()
                section_condition = torch.from_numpy(section_condition).float()
                # 输入数据(单通道)
                input = torch.stack([section_condition]).unsqueeze(0)

                input = input.to(self.device)
                section = section.to(self.device)

                # 输入网络
                logits, pred = self.unet(input)

                if section.dim()==2:
                    labels = section.unsqueeze(0)
                else :
                    labels = section
                loss_main = cross_entropy_loss(logits, labels)

                # 加入层序约束
                if epoch < self.warmup_epochs:
                    lambda_trans = self.lambda_trans_max*(epoch/max(1, self.warmup_epochs))
                else:
                    lambda_trans = self.lambda_trans_max

                trans_loss, metrics = vertical_transition_loss(
                    logits,self.W, lambda_trans=lambda_trans, reduction='mean', use_T_as_prior=False
                )

                loss = loss_main + trans_loss

                # 反向传播，优化
                self.reset_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * input.size(0)  # 由于batch_size为1，所以这里input.size(0)=1

            # 打印日志信息
            avg_loss = epoch_loss / len(train_data)
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.4f}")


            save_path = os.path.join(self.model_path, '%s-%d-%.4f.pth' % (
                self.model_type, self.num_epochs, self.lr))
            torch.save(self.unet.state_dict(), save_path)

    def train_seismic(self):
        """Train encoder, generator and discriminator."""

        # ====================================== Training ===========================================#
        # ===========================================================================================#

        # 加载预训练模型权重
        unet_path = os.path.join(self.model_path, '%s-%d-%.4f.pkl' % (
        self.model_type, self.num_epochs, self.lr))

        # U-Net Train
        if os.path.isfile(unet_path):
            # 加载预训练模型
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
        else:
            # Train for Encoder
            lr = self.lr
            best_unet_score = 0.

            # 读取地震数据集
            train_data = F3DatasetSeismic(self.train_path)
            # 读取标签数据
            path = './data/spilt_data_2d/test1_label/train'
            train_label = F3Dataset(path)

            # EMA 平滑
            ema_trans = EMALoss(alpha=0.05)  # 越小越平滑
            ema_ce = EMALoss(alpha=0.05)
            base_lambda = 0.1

            for epoch in range(self.num_epochs):
                # 逐轮训练
                self.unet.train(True)  # 切换到训练模式
                epoch_loss = 0
                epoch_loss_trans = 0
                epoch_loss_ce = 0
                target_size = (256, 192)
                for i in tqdm(range(len(train_data))):
                    # 读取地震数据
                    section = train_data[i]
                    # 顺时针旋转90度
                    section = np.rot90(section, k=-1)
                    section = Resize(section, target_size[0], target_size[1])

                    # 读取标签数据
                    section_label,_,_ = train_label[i]
                    # 顺时针旋转90度
                    section_label = np.rot90(section_label,k=-1)
                    section_label = Resize(section_label, target_size[0], target_size[1])
                    # 转为tensor
                    section_input = torch.from_numpy(section).float()
                    section_label = torch.from_numpy(section_label).long()
                    # 输入数据(2通道)
                    input = torch.stack([section_input]).unsqueeze(0)

                    input = input.to(self.device)
                    section_label = section_label.to(self.device)

                    # 输入网络
                    logits,pred = self.unet(input)

                    loss_ce = cross_entropy_loss(logits, section_label)



                    loss_trans, metrics = vertical_transition_loss(
                        logits, self.W, lambda_trans=self.lambda_trans_max, reduction='mean', use_T_as_prior=False
                    )

                    # EMA 平滑损失
                    ce = ema_ce.update(loss_ce)
                    trans = ema_trans.update(loss_trans)

                    # 动态调节权重
                    lambda_trans = (ce/trans).detach()* base_lambda
                    lambda_trans = torch.clamp(lambda_trans, 0.01, 1.0)

                    # 层序约束开关
                    # loss = loss_ce
                    loss = loss_ce + lambda_trans * loss_trans


                    # 反向传播，优化
                    self.reset_grad()
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()*input.size(0)  # 由于batch_size为1，所以这里input.size(0)=1
                    epoch_loss_trans += loss_trans.item()*input.size(0)
                    epoch_loss_ce += loss_ce.item()*input.size(0)

                # 打印日志信息
                avg_loss = epoch_loss/len(train_data)
                avg_loss_ce = epoch_loss_ce / len(train_data)
                avg_loss_trans = epoch_loss_trans / len(train_data)
                print(f"Epoch {epoch+1}/{self.num_epochs}, "
                      f"Loss: {avg_loss:.4f}, Loss_ce: {avg_loss_ce:.4f}, Loss_trans: {avg_loss_trans:.4f}")

            save_path = os.path.join(self.model_path, '%s-%d-%.4f_seismic_trans_2.pth' % (
                self.model_type, self.num_epochs, self.lr))
            torch.save(self.unet.state_dict(), save_path)

    def test(self):
        # ====================================== Testing ===========================================#
        # ===========================================================================================#
        # 加载预训练模型权重
        unet_path = os.path.join(self.model_path, '%s-%d-%.4f.pth' % (
        self.model_type, self.num_epochs, self.lr))

        unet_path = './models/seismic/U_Net-10-0.0001_pretrain.pth'

        # U-Net Train
        if os.path.isfile(unet_path):
            # 加载预训练模型
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
        else:
            print(f"未找到模型权重文件:{unet_path}")

        # 读取测试集数据
        test_data = F3Dataset(self.test_path)
        # self.unet.test(True)  # 切换到测试模式
        # epoch_loss = 0
        target_size = (192, 256)
        for i in range(len(test_data)):
            # 读取完整剖面(标签)和条件数据(钻孔)
            section, section_condition, mask = test_data[i]
            section = Resize(section, target_size[0], target_size[1])
            section_condition = Resize(section_condition, target_size[0], target_size[1])
            mask = Resize(mask, target_size[0], target_size[1])

            # 转为tensor
            section = torch.from_numpy(section).long()
            section_condition = torch.from_numpy(section_condition).float()
            mask = torch.from_numpy(mask).float()
            # 输入数据(2通道)
            # input = torch.stack([section_condition, mask], axis = 0).unsqueeze(0)
            input = torch.stack([section_condition]).unsqueeze(0)

            input = input.to(self.device)

            # 输入网络
            logits, pred = self.unet(input)

            label_save_path = os.path.join(self.result_path, f"label",'%s-%d-%.4f.sgems' % (
                self.model_type, self.num_epochs, i))
            condi_save_path = os.path.join(self.result_path, f"condition",'%s-%d-%.4f.sgems' % (
                self.model_type, self.num_epochs, i))
            pred_save_path = os.path.join(self.result_path, f"prediction",'%s-%d-%.4f.sgems' % (
                self.model_type, self.num_epochs, i))
            save_to_segms(section,label_save_path)
            save_to_segms(section_condition, condi_save_path)
            save_to_segms(pred,pred_save_path)

    def test_seismic(self):
        # ====================================== Testing ===========================================#
        # ===========================================================================================#
        # 加载预训练模型权重
        # unet_path = os.path.join(self.model_path, '%s-%d-%.4f_seismic_2.pth' % (
        # self.model_type, self.num_epochs, self.lr))

        unet_path = './models/seismic/best_11.ckpt'
        # U-Net Train
        if os.path.isfile(unet_path):
            # 加载预训练模型
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
        else:
            print(f"未找到模型权重文件:{unet_path}")
        self.unet.eval()
        # 读取测试集数据
        test_data = F3DatasetSeismic(self.test_path)
        # self.unet.test(True)  # 切换到测试模式
        # epoch_loss = 0
        target_size = (701, 255)
        for i in range(len(test_data)):
            # 读取完整剖面(标签)和条件数据(钻孔)
            section = test_data[i]
            section = self.robust_norm(section)
            section = np.expand_dims(section, axis=0)  # [1,H,W]
            section = np.expand_dims(section, axis=0)  # [1,1,H,W]
            # 顺时针旋转90度
            # section = np.rot90(section, k=-1)
            # section = np.rot90(section, k=-1)
            # section = Resize(section, target_size[0], target_size[1])

            # 转为tensor
            # section = torch.from_numpy(section).float()
            section = torch.from_numpy(section)
            # 输入数据
            # input = torch.stack([section]).unsqueeze(0)

            # input = input.to(self.device)
            input = section.to(self.device)
            # 输入网络
            # logits, pred = self.unet(input)
            pred = self.unet(input)
            # 加入硬约束
            # labels_hard = self.constrained_viterbi_decode(logits, self.W, use_T_as_prior=True)
            # pred_numpy = pred.cpu().numpy().squeeze()
            # labels_hard_numpy = labels_hard.cpu().numpy().squeeze()
            pred = torch.sigmoid(pred)  # [1,5,H,W]
            pred = self.decode_ordinal(pred)[0].cpu().numpy().astype(np.int16)
            # pred = pred.cpu().numpy().squeeze()
            pred_save_path = os.path.join(self.result_path, f"prediction",'%s-%d-%.4f.png' % (
                self.model_type, self.num_epochs, i))
            save_to_segms(labels_hard,pred_save_path)
            # save_to_segms(pred, pred_save_path)

            plt.figure(figsize=(6, 6))
            plt.imshow(pred.T, cmap="tab20", vmin=0, vmax=5)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(pred_save_path, dpi=150)
            plt.close()
            # # 保存硬约束后的结果
            # save_to_segms(pred, pred_save_path)




