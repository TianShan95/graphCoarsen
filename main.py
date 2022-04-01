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
import sys


def main():

    # 获取当地时间
    time_mark = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    # 记录本次实验的输入参数
    with open('../experiment/exp-record.txt', 'a') as f:
        f.write(time_mark + '\t' + '\t'.join(sys.argv) + '\n')
        f.close()

    # 初始化参数
    prog_args = arg_parse()

    # 定义 并创建 此次实验的 log 文件夹
    if prog_args.randGen:
        log_out_dir = prog_args.out_dir + '/'  + 'randGen_' + str(prog_args.seed)  + '_Normlize_'+ str(bool(prog_args.normalize))+'_concat_'+str(prog_args.concat) + time_mark + '_log/'
    else:
        log_out_dir = prog_args.out_dir + '/'  + 'graphSize_' + str(prog_args.gs)  + '_Normlize_'+ str(bool(prog_args.normalize))+'_concat_'+str(prog_args.concat) + time_mark + '_log/'
    if not os.path.exists(log_out_dir):
        os.makedirs(log_out_dir, exist_ok=True)
    # 定义 并创建 log 文件
    if prog_args.randGen:
        log_out_file = log_out_dir + 'randGen_' + str(prog_args.seed)  + '_Normlize_'+ str(bool(prog_args.normalize)) +'_concat_'+str(prog_args.concat) + '_' + '_shuffle_' + str(prog_args.shuffle) + time_mark + '.log'

    else:
        log_out_file = log_out_dir + 'graphSize_' + str(prog_args.gs)  + '_Normlize_'+ str(bool(prog_args.normalize)) +'_concat_'+str(prog_args.concat) + '_' + '_shuffle_' + str(prog_args.shuffle) + time_mark + '.log'



    # 配置日志 输出格式
    handler = logging.FileHandler(log_out_file)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # log 写入 参数
    logger.info('\t'.join(sys.argv))
    logger.info(f'{prog_args}')

    logger.info(f'输出 log 文件路径: {log_out_file}')
    logger.info('Shuffle ' + str(prog_args.shuffle))
    logger.info(f'{prog_args}')
    logger.info(f'{prog_args.add_log}')
    logger.info(f'{prog_args.ModelPara_dir}')
    logger.info(f'{prog_args.ModelPara_dir}')
    logger.info(f'num_classes: {prog_args.num_classes}')
    logger.info(f'batch_size: {prog_args.batch_size}')
    logger.info(f'num_pool_matrix: {prog_args.num_pool_matrix}')
    logger.info(f'num_pool_final_matrix: {prog_args.num_pool_final_matrix}')
    logger.info(f'epochs: {prog_args.num_epochs}')
    logger.info(f'learning rate: {prog_args.lr}')
    logger.info(f'num of gc layers: {prog_args.num_gc_layers}')
    logger.info(f'output_dim: {prog_args.output_dim}')
    logger.info(f'hidden_dim: {prog_args.hidden_dim}')
    logger.info(f'pred_hidden: {prog_args.pred_hidden}')
    logger.info(f'dropout: {prog_args.dropout}')
    logger.info(f'weight_decay: {prog_args.weight_decay}')
    logger.info(f'Using batch normalize: {prog_args.bn}')
    logger.info(f'Using feat: {prog_args.feat}')
    logger.info(f'Using mask: {prog_args.mask}')
    logger.info(f'Norm for eigens: {prog_args.norm}')
    logger.info(f'With test: {prog_args.with_test}')

    # 是否随机生成训练
    if prog_args.randGen:
        logger.info(f'随机生成图数据帧数 {prog_args.randGen}')
        logger.info(f'最小帧数 {prog_args.msg_smallest_num}')
        logger.info(f'最大帧数 {prog_args.msg_biggest_num}')

    # 查看设备
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    logger.info(f'Using device-----{device}')

    # 初始化 种子
    seed = 42
    setup_seed(seed)
    logger.info(f'seed-----{seed}')


    pred_hidden_dims = [int(i) for i in prog_args.pred_hidden.split('_')]
    if prog_args.bmname is not None:
        benchmark_task_val(log_out_dir, prog_args, pred_hidden_dims=pred_hidden_dims, feat=prog_args.feat, device=device)


if __name__ == "__main__":
    main()
