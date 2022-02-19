# 单独 训练 图塌缩网络
# 每个图构成为 固定长度的 can 数据 长度为 {50, 100, 200, 300}

## The code is partially adapted from https://github.com/RexYing/diffpool
import warnings

warnings.filterwarnings('ignore')
from graphModel.train import train
import networkx as nx
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch.autograd import Variable
from args import arg_parse
from processData.prepare_data import prepare_data
import os
import pickle
import random
import time
from graphModel.coarsen_pooling_with_last_eigen_padding import Graphs as gp
import hiddenlayer as hl
from graphModel import encoders
from processData.onlyGraphData import OnlyGraphData

# import encoders as encoders
# import gen.feat as featgen
# from graph_sampler import GraphSampler
from processData import load_data
# from coarsen_pooling_with_last_eigen_padding import Graphs as gp
# import graph
# import time
#
#
# def evaluate(dataset, model, args, name='Validation', max_num_examples=None, device='cpu'):
#     model.eval()
#
#     labels = []
#     preds = []
#     for batch_idx, data in enumerate(dataset):
#         adj = Variable(data['adj'].float(), requires_grad=False).to(device)
#         h0 = Variable(data['feats'].float()).to(device)
#         labels.append(data['label'].long().numpy())
#         batch_num_nodes = data['num_nodes'].int().numpy()
#
#         adj_pooled_list = []
#         batch_num_nodes_list = []
#         pool_matrices_dic = dict()
#         pool_sizes = [int(i) for i in args.pool_sizes.split('_')]
#         for i in range(len(pool_sizes)):
#             ind = i + 1
#             adj_key = 'adj_pool_' + str(ind)
#             adj_pooled_list.append(Variable(data[adj_key].float(), requires_grad=False).to(device))
#             num_nodes_key = 'num_nodes_' + str(ind)
#             batch_num_nodes_list.append(data[num_nodes_key])
#
#             pool_matrices_list = []
#             for j in range(args.num_pool_matrix):
#                 pool_adj_key = 'pool_adj_' + str(i) + '_' + str(j)
#
#                 pool_matrices_list.append(Variable(data[pool_adj_key].float(), requires_grad=False).to(device))
#
#             pool_matrices_dic[i] = pool_matrices_list
#
#         pool_matrices_list = []
#         if args.num_pool_final_matrix > 0:
#
#             for j in range(args.num_pool_final_matrix):
#                 pool_adj_key = 'pool_adj_' + str(ind) + '_' + str(j)
#
#                 pool_matrices_list.append(Variable(data[pool_adj_key].float(), requires_grad=False).to(device))
#
#             pool_matrices_dic[ind] = pool_matrices_list
#
#         ypred = model(h0, adj, adj_pooled_list, batch_num_nodes, batch_num_nodes_list, pool_matrices_dic)
#
#         # else:
#         #     ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
#         _, indices = torch.max(ypred, 1)
#         preds.append(indices.cpu().data.numpy())
#
#         if max_num_examples is not None:
#             if (batch_idx + 1) * args.batch_size > max_num_examples:
#                 break
#
#     labels = np.hstack(labels)
#     preds = np.hstack(preds)
#
#     result = {'prec': metrics.precision_score(labels, preds, average='macro'),
#               'recall': metrics.recall_score(labels, preds, average='macro'),
#               'acc': metrics.accuracy_score(labels, preds),
#               'F1': metrics.f1_score(labels, preds, average="micro")}
#     return result
#
#
# def train(dataset, model, args, same_feat=True, val_dataset=None, test_dataset=None,
#           mask_nodes=True, log_dir=None, device='cpu'):
#     # writer_batch_idx = [0, 3, 6, 9]
#
#     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
#                                  weight_decay=args.weight_decay)
#     iter = 0
#     best_val_result = {
#         'epoch': 0,
#         'loss': 0,
#         'acc': 0}
#     test_result = {
#         'epoch': 0,
#         'loss': 0,
#         'acc': 0}
#     train_accs = []
#     train_epochs = []
#     best_val_accs = []
#     best_val_epochs = []
#     test_accs = []
#     test_epochs = []
#     val_accs = []
#     for epoch in range(args.num_epochs):
#         begin_time = time.time()
#         avg_loss = 0.0
#         model.train()
#         for batch_idx, data in enumerate(dataset):
#
#             time1 = time.time()
#             model.zero_grad()
#
#             adj = Variable(data['adj'].float(), requires_grad=False).to(device)
#             h0 = Variable(data['feats'].float(), requires_grad=False).to(device)
#             label = Variable(data['label'].long()).to(device)
#             batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
#             # assign_input = Variable(data['assign_feats'].float(), requires_grad=False).to(device)
#
#             # if args.method == 'wave':
#             adj_pooled_list = []
#             batch_num_nodes_list = []
#             pool_matrices_dic = dict()
#             pool_sizes = [int(i) for i in args.pool_sizes.split('_')]
#             for i in range(len(pool_sizes)):
#                 ind = i + 1
#                 adj_key = 'adj_pool_' + str(ind)
#                 adj_pooled_list.append(Variable(data[adj_key].float(), requires_grad=False).to(device))
#                 num_nodes_key = 'num_nodes_' + str(ind)
#                 batch_num_nodes_list.append(data[num_nodes_key])
#
#                 pool_matrices_list = []
#                 for j in range(args.num_pool_matrix):
#                     pool_adj_key = 'pool_adj_' + str(i) + '_' + str(j)
#
#                     pool_matrices_list.append(Variable(data[pool_adj_key].float(), requires_grad=False).to(device))
#
#                 pool_matrices_dic[i] = pool_matrices_list
#
#             pool_matrices_list = []
#             if args.num_pool_final_matrix > 0:
#
#                 for j in range(args.num_pool_final_matrix):
#                     pool_adj_key = 'pool_adj_' + str(ind) + '_' + str(j)
#
#                     pool_matrices_list.append(Variable(data[pool_adj_key].float(), requires_grad=False).to(device))
#
#                 pool_matrices_dic[ind] = pool_matrices_list
#
#             time2 = time.time()
#
#             ypred = model(h0, adj, adj_pooled_list, batch_num_nodes, batch_num_nodes_list, pool_matrices_dic)
#             # else:
#             #     ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
#             # if not args.method == 'soft-assign' or not args.linkpred:
#             loss = model.loss(ypred, label)
#             # else:
#             #     loss = model.loss(ypred, label, adj, batch_num_nodes)
#             loss.backward()
#
#             time3 = time.time()
#
#             nn.utils.clip_grad_norm(model.parameters(), args.clip)
#             optimizer.step()
#             iter += 1
#             avg_loss += loss
#
#         avg_loss /= batch_idx + 1
#         elapsed = time.time() - begin_time
#         # if writer is not None:
#         #     writer.add_scalar('loss/avg_loss', avg_loss, epoch)
#         #     if args.linkpred:
#         #         writer.add_scalar('loss/linkpred_loss', model.link_loss, epoch)
#
#         eval_time = time.time()
#         result = evaluate(dataset, model, args, name='Train', max_num_examples=100, device=device)
#         eval_time2 = time.time()
#         train_accs.append(result['acc'])
#         train_epochs.append(epoch)
#         if val_dataset is not None:
#             val_result = evaluate(val_dataset, model, args, name='Validation', device=device)
#             val_accs.append(val_result['acc'])
#         if val_result['acc'] > best_val_result['acc'] - 1e-7:
#             best_val_result['acc'] = val_result['acc']
#             best_val_result['epoch'] = epoch
#             best_val_result['loss'] = avg_loss
#             if test_dataset is not None:
#                 test_result = evaluate(test_dataset, model, args, name='Test', device=device)
#                 test_result['epoch'] = epoch
#
#         best_val_epochs.append(best_val_result['epoch'])
#         best_val_accs.append(best_val_result['acc'])
#         if test_dataset is not None:
#             test_epochs.append(test_result['epoch'])
#             test_accs.append(test_result['acc'])
#         if epoch % 50 == 0:
#             print('Epoch: ', epoch, '----------------------------------')
#             print('Train_result: ', result)
#             print('Val result: ', val_result)
#             print('Best val result', best_val_result)
#
#             if log_dir is not None:
#                 with open(log_dir, 'a') as f:
#                     f.write('Epoch: ' + str(epoch) + '-----------------------------\n')
#                     f.write('Train_result: ' + str(result) + '\n')
#                     f.write('Val result: ' + str(val_result) + '\n')
#                     f.write('Best val result: ' + str(best_val_result) + '\n')
#
#         end_time = time.time()
#     return model, val_accs, test_accs, best_val_result
#
#
# def prepare_data(graphs, graphs_list, args, test_graphs=None, max_nodes=0, seed=0):
#     zip_list = list(zip(graphs, graphs_list))
#     random.Random(seed).shuffle(zip_list)
#     graphs, graphs_list = zip(*zip_list)
#     print('Test ratio: ', args.test_ratio)
#     print('Train ratio: ', args.train_ratio)
#     test_graphs_list = []
#
#     if test_graphs is None:
#         train_idx = int(len(graphs) * args.train_ratio)
#         test_idx = int(len(graphs) * (1 - args.test_ratio))
#         train_graphs = graphs[:train_idx]
#         val_graphs = graphs[train_idx: test_idx]
#         test_graphs = graphs[test_idx:]
#         train_graphs_list = graphs_list[:train_idx]
#         val_graphs_list = graphs_list[train_idx: test_idx]
#         test_graphs_list = graphs_list[test_idx:]
#     else:
#         train_idx = int(len(graphs) * args.train_ratio)
#         train_graphs = graphs[:train_idx]
#         train_graphs_list = graphs_list[:train_idx]
#         val_graphs = graphs[train_idx:]
#         val_graphs_list = graphs_list[train_idx:]
#     print('Num training graphs: ', len(train_graphs),
#           '; Num validation graphs: ', len(val_graphs),
#           '; Num testing graphs: ', len(test_graphs))
#
#     print('Number of graphs: ', len(graphs))
#     print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
#     print('Max, avg, std of graph size: ',
#           max([G.number_of_nodes() for G in graphs]), ', '
#                                                       "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])),
#           ', '
#           "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))
#
#     test_dataset_loader = []
#
#     dataset_sampler = GraphSampler(train_graphs, train_graphs_list, args.num_pool_matrix, args.num_pool_final_matrix,
#                                    normalize=False, max_num_nodes=max_nodes,
#                                    features=args.feature_type, norm=args.norm)
#     train_dataset_loader = torch.utils.data.DataLoader(
#         dataset_sampler,
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=args.num_workers)
#
#     dataset_sampler = GraphSampler(val_graphs, val_graphs_list, args.num_pool_matrix, args.num_pool_final_matrix,
#                                    normalize=False, max_num_nodes=max_nodes,
#                                    features=args.feature_type, norm=args.norm)
#     val_dataset_loader = torch.utils.data.DataLoader(
#         dataset_sampler,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers)
#     if len(test_graphs) > 0:
#         dataset_sampler = GraphSampler(test_graphs, test_graphs_list, args.num_pool_matrix, args.num_pool_final_matrix,
#                                        normalize=False, max_num_nodes=max_nodes,
#                                        features=args.feature_type, norm=args.norm)
#         test_dataset_loader = torch.utils.data.DataLoader(
#             dataset_sampler,
#             batch_size=args.batch_size,
#             shuffle=False,
#             num_workers=args.num_workers)
#
#     return train_dataset_loader, val_dataset_loader, test_dataset_loader, \
#            dataset_sampler.max_num_nodes, dataset_sampler.feat_dim


def benchmark_task_val(args, feat='node-label', pred_hidden_dims=[50], device='cpu'):
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
                        prepare_data(graphs, graphs_list, args, test_graphs=None, max_nodes=args.max_nodes, seed=i)
            else:
                train_dataset, val_dataset, test_dataset, max_num_nodes, input_dim = \
                    prepare_data(graphs, graphs_list, args, test_graphs=[], max_nodes=args.max_nodes, seed=i)


            # out_dir = args.bmname + '/tar_' + '_graphSize_' +str(args.gs) + str(args.train_ratio) + '_ter_' + str(args.test_ratio) + '/'   +  'num_shuffle' + str(args.num_shuffle)  + '/' +  'numconv_' + str(args.num_gc_layers) + '_dp_' + str(args.dropout) + '_wd_' + str(args.weight_decay) + '_b_' + str(args.batch_size) + '_hd_' + str(args.hidden_dim) + '_od_' + str(args.output_dim)  + '_ph_' + str(args.pred_hidden) + '_lr_' + str(args.lr)  + '_concat_' + str(args.concat)
            # out_dir = out_dir + '_ps_' + '_graphSize_' +str(args.gs) + args.pool_sizes  + '_np_' + str(args.num_pool_matrix) + '_nfp_' + str(args.num_pool_final_matrix) + '_norL_' + str(args.normalize)  + '_mask_' + str(args.mask) + '_ne_' + args.norm  + '_cf_' + str(args.con_final)

            # results_out_dir = args.out_dir + '/'  + args.bmname + '_graphSize_' +str(args.gs) + '/with_test' + str(args.with_test) +  '/using_feat_' + args.feat + '/no_val_results/with_shuffles/' + out_dir + '/'
            # log_out_dir = args.out_dir  + '/' + args.bmname + '_graphSize_' + str(args.gs) + '/with_test' + str(args.with_test) + '/using_feat_' + args.feat + '/no_val_logs/with_shuffles/'+out_dir + '/'

            time_mark = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            # results_out_dir = args.out_dir + '/' + time_mark + '_result'
            log_out_dir = args.out_dir + '/' + time_mark + '_log'

            # if not os.path.exists(results_out_dir):
            #     os.makedirs(results_out_dir, exist_ok=True)
            if not os.path.exists(log_out_dir):
                os.makedirs(log_out_dir, exist_ok=True)
            #
            # results_out_file = results_out_dir + 'shuffle' + str(args.shuffle) + '.txt'
            # log_out_file = log_out_dir + 'shuffle' + str(args.shuffle) + '.txt'
            # results_out_file_2 = results_out_dir + 'test_shuffle' + str(args.shuffle) + '.txt'
            # val_out_file = results_out_dir + 'val_result' + str(args.shuffle) + '.txt'
            # print(results_out_file)

            # with open(log_out_file, 'a') as f:
            #     f.write('Shuffle ' +str(i) + '====================================================================================\n')

            pool_sizes = [int(i) for i in args.pool_sizes.split('_')]

            model = encoders.WavePoolingGcnEncoder(input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers, args.num_pool_matrix, args.num_pool_final_matrix,pool_sizes =  pool_sizes, pred_hidden_dims = pred_hidden_dims, concat = args.concat,bn=args.bn, dropout=args.dropout, mask = args.mask,args=args, device=device)

            history = hl.History()
            canvas = hl.Canvas()
            # 把训练好的模型 保存到训练的数据集文件夹内 例如 Pre_train_D_1_2_processed/ps_10_nor_False_50
            # 模型名字里写入被训练的 epoch 数
            if args.with_test:
                _, val_accs, test_accs, best_val_result = train(data_out_dir, history, canvas, train_dataset, model, args, val_dataset=val_dataset, test_dataset=test_dataset,
                 log_dir = log_out_dir, device=device)
            else:
                _, val_accs, test_accs, best_val_result = train(data_out_dir, history, canvas, train_dataset, model, args, val_dataset=val_dataset, test_dataset=None,
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


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    prog_args = arg_parse()
    seed = 42
    print(prog_args)
    setup_seed(seed)
    # random.seed(seed)
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    print('bmname: ', prog_args.bmname)
    print('num_classes: ', prog_args.num_classes)
    # print('method: ', prog_args.method)
    print('batch_size: ', prog_args.batch_size)
    print('num_pool_matrix: ', prog_args.num_pool_matrix)
    print('num_pool_final_matrix: ', prog_args.num_pool_final_matrix)
    print('epochs: ', prog_args.num_epochs)
    print('learning rate: ', prog_args.lr)
    print('num of gc layers: ', prog_args.num_gc_layers)
    print('output_dim: ', prog_args.output_dim)
    print('hidden_dim: ', prog_args.hidden_dim)
    print('pred_hidden: ', prog_args.pred_hidden)
    # print('if_transpose: ', prog_args.if_transpose)
    print('dropout: ', prog_args.dropout)
    print('weight_decay: ', prog_args.weight_decay)
    print('shuffle: ', prog_args.shuffle)
    print('Using batch normalize: ', prog_args.bn)
    print('Using feat: ', prog_args.feat)
    print('Using mask: ', prog_args.mask)
    print('Norm for eigens: ', prog_args.norm)
    # print('Combine pooling results: ', prog_args.pool_m)
    print('With test: ', prog_args.with_test)

    # writer = None
    # print('Using method: ', prog_args.method)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print('Using device-----', device)

    # if torch.cuda.is_available() and prog_args.device == 'cuda':
    #     device = 'cuda'
    # else:
    #     device = 'cpu'

    # print('Device: ', device)
    # pred_hidden_dims = [int(i) for i in prog_args.pred_hidden.split('_')]
    # if prog_args.bmname is not None:
    #     benchmark_task_val(prog_args, pred_hidden_dims=pred_hidden_dims, feat=prog_args.feat, device=device)
    time_mark = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    print('model_' + time_mark + '_epoch_' + str(prog_args.num_epochs) + '_ps_' + prog_args.pool_sizes + '_gs_' + str(prog_args.gs) + '_nor_' + str(prog_args.normalize) + '.pth')


if __name__ == "__main__":
    main()
