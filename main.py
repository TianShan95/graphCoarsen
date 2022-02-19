# The code is partially adapted from https://github.com/RexYing/diffpool
# 单独 训练 图塌缩网络
# 每个图构成为 固定长度的 can 数据 长度为 {50, 100, 200, 300}
import warnings
import torch
from args import arg_parse
import os
from task import benchmark_task_val
import time
from graphModel.utils import setup_seed

warnings.filterwarnings('ignore')


def main():

    # 初始化参数
    prog_args = arg_parse()
    # 查看设备
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print('Using device-----', device)

    # 初始化 种子
    seed = 42
    setup_seed(seed)

    # 定义 并创建 此次实验的 log 文件夹
    time_mark = time.strftime("%Y%m%d_%H:%M:%S", time.localtime())
    log_out_dir = prog_args.out_dir + '/' + 'graphSize_' + str(prog_args.gs)  + '_Normlize_'+ str(bool(prog_args.normalize)) + '_' + time_mark + '_log/'
    if not os.path.exists(log_out_dir):
        os.makedirs(log_out_dir, exist_ok=True)
    # 定义 并创建 log 文件
    log_out_file = log_out_dir + 'graphSize_' + str(prog_args.gs)  + '_Normlize_'+ str(bool(prog_args.normalize)) + '_' + time_mark + '_shuffle_' + str(prog_args.shuffle) + '.txt'
    with open(log_out_file, 'w+') as f:
        f.write('Shuffle ' + str(prog_args.shuffle) + '====================================================================================\n')
        f.write(f'{prog_args}\n')
        f.write(f'Device: {device}\n')
        f.write(f'num_classes: {prog_args.num_classes}')
        f.write(f'batch_size: {prog_args.batch_size}')
        f.write(f'num_pool_matrix: {prog_args.num_pool_matrix}', )
        f.write(f'num_pool_final_matrix: {prog_args.num_pool_final_matrix}')
        f.write(f'epochs: {prog_args.num_epochs}')
        f.write(f'learning rate: {prog_args.lr}')
        f.write(f'num of gc layers: {prog_args.num_gc_layers}')
        f.write(f'output_dim: {prog_args.output_dim}')
        f.write(f'hidden_dim: {prog_args.hidden_dim}')
        f.write(f'pred_hidden: {prog_args.pred_hidden}')
        f.write(f'dropout: {prog_args.dropout}')
        f.write(f'weight_decay: {prog_args.weight_decay}')
        f.write(f'shuffle: {prog_args.shuffle}')
        f.write(f'Using batch normalize: {prog_args.bn}')
        f.write(f'Using feat: {prog_args.feat}')
        f.write(f'Using mask: {prog_args.mask}')
        f.write(f'Norm for eigens: {prog_args.norm}')
        f.write(f'With test: {prog_args.with_test}')


    pred_hidden_dims = [int(i) for i in prog_args.pred_hidden.split('_')]
    if prog_args.bmname is not None:
        benchmark_task_val(log_out_dir, log_out_file, prog_args, pred_hidden_dims=pred_hidden_dims, feat=prog_args.feat, device=device)


if __name__ == "__main__":
    main()
