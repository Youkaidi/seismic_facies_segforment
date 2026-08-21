"""
模型训练
"""
import argparse
import os
from torch.backends import cudnn
import random
from solver import Solver

def main(config):
    cudnn.benchmark = True
    if config.model_type not in ['U_Net', 'R2U_Net', 'AttU_Net', 'R2AttU_Net','Unet2D']:
        print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net')
        print('Your input for model_type was %s' % config.model_type)
        return

    # 创建文件保存路径
    # 模型保存路径
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    # 结果保存路径
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path, config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    # 随机生成训练超参数
    # decay_ratio = random.random() * 0.8
    # decay_epoch = int(epoch * decay_ratio)  # 学习率衰减轮数
    # # 更新参数到config中
    # config.num_epochs_decay = decay_epoch

    # 打印配置信息
    print(config)

    solver = Solver(config)

    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    if config.mode == 'train_seismic':
        solver.train_seismic()
    if config.mode == 'test':
        solver.test()
    elif config.mode == 'test_seismic':
        solver.test_seismic()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 模型超参数
    parser.add_argument('--image_size', type=int, default=255)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')

    # 训练超参数
    parser.add_argument('--img_ch', type=int, default=1)
    parser.add_argument('--output_ch', type=int, default=6)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--num_epochs_decay', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    parser.add_argument('--lambda_trans_max', type=float, default=1)  # momentum1 in Adam
    parser.add_argument('--warmup_epochs', type=int, default=5)

    # misc
    parser.add_argument('--mode', type=str, default='test_seismic')
    parser.add_argument('--model_type', type=str, default='Unet2D', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--model_path', type=str, default='./models/seismic')
    parser.add_argument('--train_path', type=str, default='./data/spilt_data_2d/test1_seismic/train/')
    parser.add_argument('--valid_path', type=str, default='./data/spilt_data_2d/test1_label/valid/')
    # parser.add_argument('--test_path', type=str, default='./data/spilt_data_2d/test1_seismic/test2/')
    parser.add_argument('--test_path', type=str, default='./data/spilt_data_2d/test1_seismic/test2/')
    parser.add_argument('--result_path', type=str, default='./result/Seismic')

    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()
    main(config)