import random
from processData.graph_sampler import GraphSampler

import torch
import numpy as np


# 该函数 为 单独训练图网络准备数据
def prepare_data(graphs, graphs_list, args, test_graphs=None, max_nodes=0, seed=0):
    zip_list = list(zip(graphs, graphs_list))
    random.Random(seed).shuffle(zip_list)
    graphs, graphs_list = zip(*zip_list)
    print('Test ratio: ', args.test_ratio)
    print('Train ratio: ', args.train_ratio)
    test_graphs_list = []

    if test_graphs is None:
        train_idx = int(len(graphs) * args.train_ratio)
        test_idx = int(len(graphs) * (1 - args.test_ratio))
        train_graphs = graphs[:train_idx]
        val_graphs = graphs[train_idx: test_idx]
        test_graphs = graphs[test_idx:]
        train_graphs_list = graphs_list[:train_idx]
        val_graphs_list = graphs_list[train_idx: test_idx]
        test_graphs_list = graphs_list[test_idx:]
    else:
        train_idx = int(len(graphs) * args.train_ratio)
        train_graphs = graphs[:train_idx]
        train_graphs_list = graphs_list[:train_idx]
        val_graphs = graphs[train_idx:]
        val_graphs_list = graphs_list[train_idx:]
    print('Num training graphs: ', len(train_graphs),
          '; Num validation graphs: ', len(val_graphs),
          '; Num testing graphs: ', len(test_graphs))

    print('Number of graphs: ', len(graphs))
    print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
    print('Max, avg, std of graph size: ',
          max([G.number_of_nodes() for G in graphs]), ', '
                                                      "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])),
          ', '
          "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))

    test_dataset_loader = []

    dataset_sampler = GraphSampler(train_graphs, train_graphs_list, args.num_pool_matrix, args.num_pool_final_matrix,
                                   normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type, norm=args.norm)
    train_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)

    dataset_sampler = GraphSampler(val_graphs, val_graphs_list, args.num_pool_matrix, args.num_pool_final_matrix,
                                   normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type, norm=args.norm)
    val_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers)
    if len(test_graphs) > 0:
        dataset_sampler = GraphSampler(test_graphs, test_graphs_list, args.num_pool_matrix, args.num_pool_final_matrix,
                                       normalize=False, max_num_nodes=max_nodes,
                                       features=args.feature_type, norm=args.norm)
        test_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers)

    return train_dataset_loader, val_dataset_loader, test_dataset_loader, \
           dataset_sampler.max_num_nodes, dataset_sampler.feat_dim


