import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='Arguments.')
    parser.add_argument('--bmname', dest='bmname',
                        help='Name of the benchmark dataset')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int,
                        help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--lr', dest='lr', type=float,
                        help='Learning rate.')
    parser.add_argument('--clip', dest='clip', type=float,
                        help='Gradient clipping.')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
                        help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--train-ratio', dest='train_ratio', type=float,
                        help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--test-ratio', dest='test_ratio', type=float,
                        help='Ratio of number of graphs testing set to all graphs.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
                        help='Number of workers to load data.')
    parser.add_argument('--feature', dest='feature_type',
                        help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--input-dim', dest='input_dim', type=int,
                        help='Input feature dimension')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int,
                        help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int,
                        help='Output dimension')
    parser.add_argument('--num-classes', dest='num_classes', type=int,
                        help='Number of label classes')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int,
                        help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const',
                        const=False, default=True,
                        help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float,
                        help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const',
                        const=False, default=True,
                        help='Whether to add bias. Default to True.')
    parser.add_argument('--datadir', dest='datadir',
                        help='Directory where benchmark is located')

    parser.add_argument('--pool_sizes', type=str,
                        help='pool_sizes', default='10')
    parser.add_argument('--num_pool_matrix', type=int,
                        help='num_pooling_matrix', default=1)
    parser.add_argument('--min_nodes', type=int,
                        help='min_nodes', default=12)

    parser.add_argument('--weight_decay', type=float,
                        help='weight_decay', default=0.0)
    parser.add_argument('--num_pool_final_matrix', type=int,
                        help='number of final pool matrix', default=0)


    parser.add_argument('--pred_hidden', type=str,
                        help='pred_hidden', default='50')

    parser.add_argument('--out_dir', type=str,
                        help='out_dir', default='experiment')
    parser.add_argument('--num_shuffle', type=int,
                        help='total num_shuffle', default=10)
    parser.add_argument('--shuffle', type=int,
                        help='which shuffle, choose from 0 to 9', default=0)
    parser.add_argument('--concat', type=int,
                        help='whether concat', default=1)
    parser.add_argument('--feat', type=str,
                        help='which feat to use', default='node-label')
    parser.add_argument('--mask', type=int,
                        help='mask or not', default=1)
    parser.add_argument('--norm', type=str,
                        help='Norm for eigens', default='l2')

    parser.add_argument('--with_test', type=int,
                        help='with test or not', default=0)
    parser.add_argument('--con_final', type=int,
                        help='con_final', default=1)
    parser.add_argument('--device', type=str,
                        help='cpu or cuda', default='cpu')


    # 关于 图塌缩操作的 重要参数
    parser.add_argument('--normalize', type=int,
                        help='normalized laplacian or not', default=1)  # 图塌缩时 影响得到的池化矩阵


    # 使用 Car_Hacking_Challenge_Dataset_rev20Mar2021 数据集
    # 生成数据集需要修改的参数
    parser.add_argument('--ds', type=list,
                        help='dynamic or static', default=['D'])  # D or S 车辆动态报文 或者 车辆静止报文
    parser.add_argument('--csv_num', type=list,
                        help='csv num', default=[1, 2])  # 0 or 1 or 2  # csv文件标号
    parser.add_argument('--gs', type=int,
                        help='graph size', default=50)

    parser.add_argument('--regen_dataset', type=bool,
                        help='ReGenerate dataset', default=False)  # 如果图数据集有 就不重新生成数据集 只生成processed数据
    parser.add_argument('--dataset_name', type=str,
                        help='dynamic or static', default='Car_Hacking_Challenge_Dataset_rev20Mar2021')  # 0 or 1 or 2



    parser.set_defaults(max_nodes=100,
                        feature_type='default',
                        datadir='../data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/0_Training/',
                        lr=0.001,
                        clip=2.0,
                        batch_size=64,
                        num_epochs=20,
                        train_ratio=0.8,
                        test_ratio=0.1,
                        num_workers=0,
                        input_dim=10,
                        hidden_dim=20,
                        output_dim=20,
                        num_classes=2,  # 分两类 正常报文 异常报文
                        num_gc_layers=3,
                        dropout=0.0,
                        bmname='Pre_train',
                        )
    return parser.parse_args()


if __name__ == "__main__":
    p = arg_parse()
    print(p.bn)