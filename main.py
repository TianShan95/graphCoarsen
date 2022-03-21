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
from utils.logger import logger
import logging
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
    time_mark = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if prog_args.randGen:
        log_out_dir = prog_args.out_dir + '/'  + '_randGen_' + str(prog_args.seed)  + '_Normlize_'+ str(bool(prog_args.normalize))+'_concat_'+str(prog_args.concat) + time_mark + '_log/'
    else:
        log_out_dir = prog_args.out_dir + '/'  + '_graphSize_' + str(prog_args.gs)  + '_Normlize_'+ str(bool(prog_args.normalize))+'_concat_'+str(prog_args.concat) + time_mark + '_log/'
    if not os.path.exists(log_out_dir):
        os.makedirs(log_out_dir, exist_ok=True)
    # 定义 并创建 log 文件
    if prog_args.randGen:
        log_out_file = log_out_dir + '_randGen_' + str(prog_args.seed)  + '_Normlize_'+ str(bool(prog_args.normalize)) +'_concat_'+str(prog_args.concat) + '_' + '_shuffle_' + str(prog_args.shuffle) + time_mark + '.txt'

    else:
        log_out_file = log_out_dir + '_graphSize_' + str(prog_args.gs)  + '_Normlize_'+ str(bool(prog_args.normalize)) +'_concat_'+str(prog_args.concat) + '_' + '_shuffle_' + str(prog_args.shuffle) + time_mark + '.txt'



    # 配置日志 输出格式
    handler = logging.FileHandler(log_out_file)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # log 写入 参数
    logger.info(f'{prog_args}')

    logger.info(f'输出 log 文件路径: {log_out_file}')
    logger.info('Shuffle ' + str(prog_args.shuffle) + '====================================================================================\n')
    logger.info(f'{prog_args}\n')
    logger.info(f'{prog_args.add_log}\n')
    logger.info(f'{prog_args.ModelPara_dir}\n')
    logger.info(f'{prog_args.ModelPara_dir}\n')
    logger.info(f'Device: {device}\n')
    logger.info(f'num_classes: {prog_args.num_classes}\n')
    logger.info(f'batch_size: {prog_args.batch_size}\n')
    logger.info(f'num_pool_matrix: {prog_args.num_pool_matrix}\n', )
    logger.info(f'num_pool_final_matrix: {prog_args.num_pool_final_matrix}\n')
    logger.info(f'epochs: {prog_args.num_epochs}\n')
    logger.info(f'learning rate: {prog_args.lr}\n')
    logger.info(f'num of gc layers: {prog_args.num_gc_layers}\n')
    logger.info(f'output_dim: {prog_args.output_dim}\n')
    logger.info(f'hidden_dim: {prog_args.hidden_dim}\n')
    logger.info(f'pred_hidden: {prog_args.pred_hidden}\n')
    logger.info(f'dropout: {prog_args.dropout}\n')
    logger.info(f'weight_decay: {prog_args.weight_decay}\n')
    logger.info(f'shuffle: {prog_args.shuffle}\n')
    logger.info(f'Using batch normalize: {prog_args.bn}\n')
    logger.info(f'Using feat: {prog_args.feat}\n')
    logger.info(f'Using mask: {prog_args.mask}\n')
    logger.info(f'Norm for eigens: {prog_args.norm}\n')
    logger.info(f'With test: {prog_args.with_test}\n')


    pred_hidden_dims = [int(i) for i in prog_args.pred_hidden.split('_')]
    if prog_args.bmname is not None:
        benchmark_task_val(log_out_dir, log_out_file, prog_args, pred_hidden_dims=pred_hidden_dims, feat=prog_args.feat, device=device)


if __name__ == "__main__":
    main()
