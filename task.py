from graphModel.train import train
import os
import pickle
from processData.onlyGraphData import OnlyGraphData
from processData import load_data
import networkx as nx
from graphModel.coarsen_pooling_with_last_eigen_padding import Graphs as gp
import numpy as np
from processData.prepare_data import prepare_data
import hiddenlayer as hl
from graphModel import encoders
import torch


def benchmark_task_val(log_out_dir, log_out_file, args, feat='node-label', pred_hidden_dims=[50], device='cpu'):
    '''
    :param log_out_dir: 实验结果输出文件夹 保存实验数据 和 训练过的模型
    :param args:
    :param feat:
    :param pred_hidden_dims:
    :param device:
    :return:
    '''
    all_vals = []

    # ps: pool_size 每个簇的节点数量
    # gs: graph_size 样本每个图的大小
    # 定义数据输出文件夹
    data_out_dir = args.datadir
    data_out_dir += args.bmname + '_'
    # 如果是 Car_Hacking_Challenge_Dataset_rev20Mar2021 数据集
    # 指定是 动态车辆报文 或者是 静止车辆报文
    if args.dataset_name == 'Car_Hacking_Challenge_Dataset_rev20Mar2021':
        for i in args.ds:
            data_out_dir += i + '_'  # 表明是 车辆动态报文 或者 车辆动态报文
        for i in args.csv_num:
            data_out_dir += str(i)+'_'

    data_out_dir = data_out_dir + 'processed/ps_' + args.pool_sizes
    # 标记 邻接矩阵是否 拉普拉斯 归一化
    data_out_dir = data_out_dir + '_nor_' + str(bool(args.normalize)) + '_' + str(args.gs) + '/'
    print(f'查找数据集的文件位置是 {data_out_dir}')
    # 若 数据文件夹 不存在 则新建
    if not os.path.exists(data_out_dir):
        os.makedirs(data_out_dir)

    graph_list_file_name = data_out_dir + 'graphs_list.p'
    dataset_file_name = data_out_dir + 'dataset.p'

    if os.path.isfile(graph_list_file_name) and os.path.isfile(dataset_file_name):
        print('Files exist, reading from stored files....')
        print('Reading file from', data_out_dir)
        # 输出信息到log文件
        with open(log_out_file, 'a') as f:
            f.write(f'Files exist, reading from stored files....')
            f.write(f'Reading file from{data_out_dir}')
            f.close()
        with open(dataset_file_name, 'rb') as f:
            # 原始图的文件
            graphs = pickle.load(f)
        with open(graph_list_file_name, 'rb') as f:
            # 坍塌处理过的 图文件
            graphs_list = pickle.load(f)
        print('Data loaded!')
    else:
        print('No files exist, preprocessing datasets...')

        # 生成图数据集
        p = OnlyGraphData(args)

        graphs = load_data.read_graphfile(p.output_name_suffix, max_nodes=args.max_nodes)
        print('Data length before filtering: ', len(graphs))

        dataset_copy = graphs.copy()

        len_data = len(graphs)
        graphs_list = []
        pool_sizes = [int(i) for i in args.pool_sizes.split('_')]
        print('pool_sizes: ', pool_sizes)

        # 遍历每个图 得到坍塌图的池化矩阵
        # 把处理好的数据储存
        for i in range(len_data):
            adj = nx.adjacency_matrix(dataset_copy[i])
            # print('Adj shape',adj.shape)
            if adj.shape[0] < args.min_nodes or adj.shape[0] > args.max_nodes or adj.shape[0] != dataset_copy[i].number_of_nodes():
                graphs.remove(dataset_copy[i])
                # index_list.remove(i)
            else:
                # print('----------------------', i, adj.shape)
                number_of_nodes = dataset_copy[i].number_of_nodes()
                # if args.pool_ratios is not None:
                #     pool_sizes = []
                #     pre_layer_number_of_nodes = number_of_nodes
                #     for i in range(len(pool_ratios)):
                #         number_of_nodes_after_pool = int(pre_layer_number_of_nodes*pool_ratios[i])
                #         pool_sizes.append(number_of_nodes_after_pool)
                #         pre_layer_number_of_nodes = number_of_nodes_after_pool

                # print('Test pool_sizes:  ', pool_sizes)
                coarsen_graph = gp(adj.todense().astype(float), pool_sizes)
                # if args.method == 'wave':
                coarsen_graph.coarsening_pooling(args.normalize)
                graphs_list.append(coarsen_graph)
        print('Data length after filtering: ', len(graphs), len(graphs_list))
        print('Dataset preprocessed, dumping....')
        with open(dataset_file_name, 'wb') as f:
            pickle.dump(graphs, f)
        with open(graph_list_file_name, 'wb') as f:
            pickle.dump(graphs_list, f)
        print('Dataset dumped!')

    # 获取 图特征标识
    # if feat == 'node-feat' and 'feat_dim' in graphs[0].graph:
    #     print('Using node features')
    #     input_dim = graphs[0].graph['feat_dim']
    # elif feat == 'node-label' and 'label' in graphs[0].nodes[0]:
    print('Using node labels')
    # 使用节点标签作为图特征
    for G in graphs:
        for u in G.nodes():
            G.nodes[u]['feat'] = np.array(G.nodes[u]['label'])
    # else:
    #     print('Using constant labels')
    #     featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
    #     for G in graphs:
    #         featgen_const.gen_node_features(G)



    total_test_ac = 0
    total_test_best_ac = 0
    total_best_val_ac = 0
    for i in range(10):
        # 随机打乱的种子
        if i == args.shuffle:

            # 得到数据集
            if args.with_test:
                train_dataset, val_dataset, test_dataset, max_num_nodes, input_dim = \
                        prepare_data(log_out_file, graphs, graphs_list, args, test_graphs=None, max_nodes=args.max_nodes, seed=i)
            else:
                train_dataset, val_dataset, test_dataset, max_num_nodes, input_dim = \
                    prepare_data(log_out_file, graphs, graphs_list, args, test_graphs=[], max_nodes=args.max_nodes, seed=i)


            # out_dir = args.bmname + '/tar_' + '_graphSize_' +str(args.gs) + str(args.train_ratio) + '_ter_' + str(args.test_ratio) + '/'   +  'num_shuffle' + str(args.num_shuffle)  + '/' +  'numconv_' + str(args.num_gc_layers) + '_dp_' + str(args.dropout) + '_wd_' + str(args.weight_decay) + '_b_' + str(args.batch_size) + '_hd_' + str(args.hidden_dim) + '_od_' + str(args.output_dim)  + '_ph_' + str(args.pred_hidden) + '_lr_' + str(args.lr)  + '_concat_' + str(args.concat)
            # out_dir = out_dir + '_ps_' + '_graphSize_' +str(args.gs) + args.pool_sizes  + '_np_' + str(args.num_pool_matrix) + '_nfp_' + str(args.num_pool_final_matrix) + '_norL_' + str(args.normalize)  + '_mask_' + str(args.mask) + '_ne_' + args.norm  + '_cf_' + str(args.con_final)

            # results_out_dir = args.out_dir + '/'  + args.bmname + '_graphSize_' +str(args.gs) + '/with_test' + str(args.with_test) +  '/using_feat_' + args.feat + '/no_val_results/with_shuffles/' + out_dir + '/'
            # log_out_dir = args.out_dir  + '/' + args.bmname + '_graphSize_' + str(args.gs) + '/with_test' + str(args.with_test) + '/using_feat_' + args.feat + '/no_val_logs/with_shuffles/'+out_dir + '/'


            #
            # results_out_file = results_out_dir + 'shuffle' + str(args.shuffle) + '.txt'
            # log_out_file = log_out_dir + 'shuffle' + str(args.shuffle) + '.txt'
            # results_out_file_2 = results_out_dir + 'test_shuffle' + str(args.shuffle) + '.txt'
            # val_out_file = results_out_dir + 'val_result' + str(args.shuffle) + '.txt'
            # print(results_out_file)

            # with open(log_out_file, 'a') as f:
            #     f.write('Shuffle ' +str(i) + '====================================================================================\n')

            pool_sizes = [int(i) for i in args.pool_sizes.split('_')]

            # 如果指定了 模型参数路径 则 load 模型和 模型参数
            if args.ModelPara_dir:
                # 加载模型参数 因为模型命名和模型参数命名 是把model 改为 para
                # 则把模型路径 字符串 里最后一个 model 字符串 替换为 para
                match_string_len = len("model")
                last_char_index = args.ModelPara_dir.rfind("model")
                # 加载 指定模型
                # 加载在 cuda 上训练的 模型
                if device == 'cpu':
                    model = torch.load(args.ModelPara_dir, map_location=torch.device('cpu'))
                    model.load_state_dict(torch.load(args.ModelPara_dir[:last_char_index] + "para" + args.ModelPara_dir[last_char_index+match_string_len:],
                                                     map_location=torch.device('cpu')))
                elif device == 'cuda':
                    model = torch.load(args.ModelPara_dir)
                    print(f'模型文件是否存在：{os.path.isfile(args.ModelPara_dir)}')
                    print(f'模型参数文件是否存在：{os.path.isfile(args.ModelPara_dir[:last_char_index] + "para" + args.ModelPara_dir[last_char_index + match_string_len:])}')
                    print(f'type model {type(model)}')
                    model.load_state_dict(torch.load(args.ModelPara_dir[:last_char_index] + "para" + args.ModelPara_dir[last_char_index + match_string_len:]))

            else:
                model = encoders.WavePoolingGcnEncoder(input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers, args.num_pool_matrix, args.num_pool_final_matrix, pool_sizes=pool_sizes, pred_hidden_dims = pred_hidden_dims, concat = args.concat,bn=args.bn, dropout=args.dropout, mask = args.mask,args=args, device=device)

            # model = encoders.WavePoolingGcnEncoder(input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers, args.num_pool_matrix, args.num_pool_final_matrix,pool_sizes =  pool_sizes, pred_hidden_dims = pred_hidden_dims, concat = args.concat,bn=args.bn, dropout=args.dropout, mask = args.mask,args=args, device=device)
            # match_string_len = len("model")
            # last_char_index = args.ModelPara_dir.rfind("model")
            # model.load_state_dict(torch.load(args.ModelPara_dir[:last_char_index] + "para" + args.ModelPara_dir[last_char_index+match_string_len:], map_location=torch.device('cpu')))

            history = hl.History()
            canvas = hl.Canvas()
            # 把训练好的模型 保存到训练的数据集文件夹内 例如 Pre_train_D_1_2_processed/ps_10_nor_False_50
            # 模型名字里写入被训练的 epoch 数
            if args.with_test:
                _, val_accs, test_accs, best_val_result = train(log_out_dir, log_out_file, history, canvas, train_dataset, model, args, val_dataset=val_dataset, test_dataset=test_dataset,
                 log_dir = log_out_dir, device=device)
            else:
                _, val_accs, test_accs, best_val_result = train(log_out_dir, log_out_file, history, canvas, train_dataset, model, args, val_dataset=val_dataset, test_dataset=None,
                 log_dir = log_out_dir, device=device)

            print('Shuffle ', i, '--------- best val result', best_val_result )



    #
    #         if args.with_test:
    #             test_ac = test_accs[best_val_result['epoch']]
    #             print('Test accuracy: ', test_ac)
    #         best_val_ac =  best_val_result['acc']
    #
    #
    #
    #
    #         print('Best val on shuffle ', (args.shuffle), best_val_ac)
    #         if args.with_test:
    #             print('Test on shuffle', args.shuffle,' : ', test_ac)
    #
    #
    #
    #
    #
    # np.savetxt(val_out_file, val_accs)
    #
    # with open(results_out_file, 'w') as f:
    #     f.write('Best val on shuffle '+ str(args.shuffle) + ': ' + str(best_val_ac) + '\n')
    # if args.with_test:
    #     with open(results_out_file_2, 'w') as f:
    #         f.write('Test accuracy on shuffle ' + str( args.shuffle  ) +  ':' + str(test_ac) + '\n')
    #
    #
    # with open(log_out_file,'a') as f:
    #
    #
    #     f.write('Best val on shuffle ' + str(args.shuffle ) + ' : ' + str(best_val_ac) + '\n')
    #     if args.with_test:
    #         f.write('Test on shuffle ' + str( args.shuffle  ) +  ' : ' + str(test_ac) + '\n')
    #     f.write('------------------------------------------------------------------\n')
