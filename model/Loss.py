import torch.nn.functional as F
import torch


def cross_entropy_loss(logits, label):
    """
    全图多分类交叉熵损失：所有像素（条件区域+缺失区域）均参与损失计算
    :param logits: 模型输出的logits，shape=(batch,5,H,W)
    :param label: 完整岩性标签，shape=(batch,H,W)（值0-4）
    :param mask: 输入的掩码通道（此处仅为兼容原接口，实际不影响计算）
    :return: 全图平均损失
    """
    # 展平数据
    batch_size, num_classes, H, W = logits.shape
    logits_flat = logits.view(batch_size, num_classes, -1)  # (batch,num_class,H*W)
    label_flat = label.view(batch_size, -1)  # (batch,H*W)

    # 计算交叉熵损失
    ce_loss = F.cross_entropy(
        logits_flat, label_flat, reduction='none'  # 保留每个像素的损失
    ).view(batch_size, -1)  # (batch,H*W)

    total_loss = ce_loss  # 全图所有像素的损失均保留

    # 计算全图平均损失
    total_pixels = H * W  # 每个样本的总像素数（H*W）
    avg_loss = (total_loss.sum(dim=1) / total_pixels).mean()  # 先按样本归一化，再求批次平均

    return avg_loss


EPS = 1e-8

def vertical_transition_loss(logits, W, lambda_trans=1.0, reduction='mean',
                            use_T_as_prior=False):
    """
    计算竖直相邻像素的层序约束损失

    Args:
        logits: Tensor [B, C, H, W], 网络原始输出（未softmax）。
        W: 概率转移矩阵 Tensor [C, C]，
            如果 use_T_as_prior=False，W 被视为二值/权重掩码（W_ij 越大表示越惩罚）。
            如果 use_T_as_prior=True，W 被视为先验转移概率矩阵 T（每项为 P(j|i)），函数内部会转换为 -log(T+EPS)。
        lambda_trans: float, 该损失的权重系数。
        reduction: 'mean' or 'sum'，返回的 loss 聚合方式（默认 mean）。
        use_T_as_prior: 是否把 W 当作概率先验 T（True 时内部用 -log(T+EPS)）。
    Returns:
        loss (scalar tensor),
        dict metrics 包含:
            - 'forbidden_mass' : 标量（观测到的被惩罚的概率质量）
            - 'violation_map'  : [B, H-1, W] 每个竖向位置上的违例值（可用于可视化，未归一）
            - 'observed_R'    : [C, C] 该 batch 上观测到的软转移计数（未归一）
    """
    assert logits.dim() == 4, "logits 应为 [B,C,H,W]"
    B, C, H, W_img = logits.shape
    device = logits.device

    # p: [B, C, H, W]
    p = F.softmax(logits, dim=1)  # 将logits转为概率分布(dim表示每个像素的概率和为1)

    # 竖向相邻对：top 和 bottom 形状均为 [B, C, H-1, W]
    p_top = p[:, :, :-1, :]   # (r, c)
    p_bot = p[:, :, 1:, :]    # (r+1, c)

    # 若 W 是先验概率矩阵 T，则转换为代价矩阵 -log(T+eps)
    if use_T_as_prior:
        T = W
        W_cost = -torch.log(T.clamp(min=EPS))
    else:
        W_cost = W  # 直接作为权重/掩码

    # 计算每个竖向位置的 violation: sum_{i,j} p_top[i]*W_cost[i,j]*p_bot[j]
    # einsum: 'b c h w, c d, b d h w -> b h w'
    viol_map = torch.einsum('bchw,cd,bdhw->bhw', p_top, W_cost, p_bot)  # shape [B, H-1, W]

    # 聚合成标量损失
    if reduction == 'mean':
        loss_val = viol_map.mean()
    elif reduction == 'sum':
        loss_val = viol_map.sum()
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")

    # 层序约束权重
    loss = lambda_trans * loss_val

    # 监控项：observed_R = sum_{b,h,w} outer(p_top, p_bot) -> shape [C,C]
    # compute R by einsum: 'b c h w, b d h w -> c d'
    observed_R = torch.einsum('bchw,bdhw->cd', p_top, p_bot).detach()

    # 如果用户传入的是二值掩码 M（W>0 表示被惩罚），计算 forbidden_mass 为 R * M 之和
    if not use_T_as_prior:
        M = (W_cost > 0).float().to(device)
        forbidden_mass = (observed_R * M).sum().detach()
    else:
        # 若是概率先验，不能简单用 W>0 判断；用 -log(T) 作为权重来衡量总体代价
        forbidden_mass = (observed_R * W_cost).sum().detach()

    metrics = {
        'forbidden_mass': forbidden_mass,
        'violation_map': viol_map.detach(),   # 若 batch 较大，可选性返回
        'observed_R': observed_R
    }

    return loss, metrics

class EMALoss:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.ema = None

    def update(self, loss_value):
        loss_value = loss_value.detach()
        if self.ema is None:
            self.ema = loss_value
        else:
            self.ema = self.alpha * loss_value + (1 - self.alpha) * self.ema
        return self.ema



