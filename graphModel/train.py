import torch
import time
from torch.autograd import Variable
import torch.nn as nn
from graphModel.evaluate import evaluate
import hiddenlayer as hl
import copy
import re
from utils.logger import logger


# 单独训练 图网络的 训练函数
def train(data_out_dir, history, canvas, dataset, model, args, same_feat=True, val_dataset=None, test_dataset=None,
          mask_nodes=True, device='cpu'):
    # writer_batch_idx = [0, 3, 6, 9]

    # 模型保存命名相关
    if not args.randGen:  # 如果不是随机生成则执行以下代码
        model_name_add = ''
        # 放在第一个 gs 位置的数字 表示第一次被训练是的 graph_size 如果是重训的则需要填上重新训练的 graph_size
        initial_gs = str(args.gs)
        # 模型最初的graphSize和现在要训练的数据的graphSize不一致 或者 名字里有多个 gs 字样 则都在模型名字后面追加 _gs_graphSize(此次图数据的can帧数)
        if args.ModelPara_dir:
            gs = re.findall(r"_gs_(\d+)_nor", args.ModelPara_dir)

            gs_list = re.findall(r'_gs_(\d+)', args.ModelPara_dir)[1:]  # 这里list 存放了 除了第一个的 gs 参数
            gs_list += str(args.gs)  # 再加上 本次的 gs
            gs_num = args.ModelPara_dir.count("_gs_")
            if int(gs[0]) != args.gs or gs_num > 1:  # 出现了不只一次 gs
                initial_gs = str(gs[0])  # 最初的 graphSize
                # 名字模型 加上 所有训练过的 历史 gs
                for gs in gs_list:
                    model_name_add += '_gs_' + str(gs)


    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    iter = 0
    best_val_result = {
        'epoch': 0,
        'loss': 0,
        'acc': 0}
    test_result = {
        'epoch': 0,
        'loss': 0,
        'acc': 0}
    train_accs = []
    train_epochs = []
    best_val_accs = []
    best_val_epochs = []
    test_accs = []
    test_epochs = []
    val_accs = []
    # sing_epoch_loss_list = []  # 记录每个epoch的loss值 画每个epoch的loss曲线
    # all_loss_list = []  # 记录每个
    train_start_time = time.time()  # 记录整个训练开始的时间

    print(f"### ### device {device}")

    for epoch in range(args.num_epochs):
        begin_time = time.time()
        avg_loss = 0.0
        model.train()
        for batch_idx, data in enumerate(dataset):

            model.zero_grad()

            adj = Variable(data['adj'].float(), requires_grad=False).to(device)
            h0 = Variable(data['feats'].float(), requires_grad=False).to(device)
            label = Variable(data['label'].long()).to(device)
            batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
            # assign_input = Variable(data['assign_feats'].float(), requires_grad=False).to(device)

            # if args.method == 'wave':
            adj_pooled_list = []
            batch_num_nodes_list = []
            pool_matrices_dic = dict()
            pool_sizes = [int(i) for i in args.pool_sizes.split('_')]
            for i in range(len(pool_sizes)):
                ind = i + 1
                adj_key = 'adj_pool_' + str(ind)
                adj_pooled_list.append(Variable(data[adj_key].float(), requires_grad=False).to(device))
                num_nodes_key = 'num_nodes_' + str(ind)
                batch_num_nodes_list.append(data[num_nodes_key])

                pool_matrices_list = []
                for j in range(args.num_pool_matrix):
                    pool_adj_key = 'pool_adj_' + str(i) + '_' + str(j)

                    pool_matrices_list.append(Variable(data[pool_adj_key].float(), requires_grad=False).to(device))

                pool_matrices_dic[i] = pool_matrices_list

            pool_matrices_list = []
            if args.num_pool_final_matrix > 0:

                for j in range(args.num_pool_final_matrix):
                    pool_adj_key = 'pool_adj_' + str(ind) + '_' + str(j)

                    pool_matrices_list.append(Variable(data[pool_adj_key].float(), requires_grad=False).to(device))

                pool_matrices_dic[ind] = pool_matrices_list

            ypred, _ = model(h0, adj, adj_pooled_list, batch_num_nodes, batch_num_nodes_list, pool_matrices_dic)
            # else:
            #     ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
            # if not args.method == 'soft-assign' or not args.linkpred:
            loss = model.loss(ypred, label)
            # else:
            #     loss = model.loss(ypred, label, adj, batch_num_nodes)
            loss.backward()

            # time3 = time.time()  # 一个数据 进行训练的时间

            nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()
            iter += 1
            avg_loss += loss
            # sing_epoch_loss_list.append(copy.deepcopy(loss))  # 记录下每个 loss

            # 记录 loss
            # if log_out_file:
            #     with open(log_out_file, 'a') as f:
            #         f.write('\nEpoch: ' + str(epoch) + 'loss: ' + str(loss.item()) + '\n')
                    # f.close()

            # 训练时 实时显示 loss 变化曲线
            history.log((epoch, batch_idx), train_loss=loss)
            # loss_list.append(loss)
            with canvas:
                canvas.draw_plot(history['train_loss'])
            if args.randGen:
                logger.info(f'Epoch: {epoch:4} step: {batch_idx:5}, loss: {loss:.6f} seed: {args.seed}, concat: {args.concat} Norm: {args.normalize}')
            else:
                logger.info(f'Epoch: {epoch:4} step: {batch_idx:5}, loss: {loss:.6f} graph_size: {args.gs}, concat: {args.concat} Norm: {args.normalize}')
            # print(epoch)


        avg_loss /= batch_idx + 1
        logger.info(f'epoch {epoch} 结束 平均loss是 {avg_loss}')
        elapsed = time.time() - begin_time  # 一个 epoch 耗费的时间
        # if writer is not None:
        #     writer.add_scalar('loss/avg_loss', avg_loss, epoch)
        #     if args.linkpred:
        #         writer.add_scalar('loss/linkpred_loss', model.link_loss, epoch)

        logger.info('正在评估在训练数据上的精度')
        result = evaluate(dataset, model, args, name='Train', max_num_examples=100, device=device)

        # 在训练集上的精度
        train_accs.append(result['acc'])
        logger.info(f"在训练集的精度是: {result['acc']}")
        train_epochs.append(epoch)
        if val_dataset is not None:
            val_result = evaluate(val_dataset, model, args, name='Validation', device=device)
            val_accs.append(val_result['acc'])
            logger.info(f"在验证集的精度是: {val_result['acc']}")
        else:
            logger.info(f'无 验证集')

        # 记录在 验证集最好的 精度 epoch 和 loss
        if val_result['acc'] > best_val_result['acc'] - 1e-7:
            best_val_result['acc'] = val_result['acc']
            best_val_result['epoch'] = epoch
            best_val_result['loss'] = avg_loss
            if test_dataset is not None:
                test_result = evaluate(test_dataset, model, args, name='Test', device=device)
                test_result['epoch'] = epoch
                logger.info(f"在测试集的精度是: {test_result['acc']}")
            else:
                logger.info(f'无 测试集')

            # 保存在验证集上精度最好的模型
            # 保存模型参数
            if args.randGen:
                model_para_name = str(format(val_result['acc'], '.2f')) + '_better_para_' + '_totalEpoch_' + \
                                  str(args.num_epochs) + '_epoch_' + str(epoch) + '_ps_' + args.pool_sizes +\
                                   '_nor_' + str(args.normalize) + '.pth'
                torch.save(model.state_dict(), data_out_dir + model_para_name)  # 保存模型参数
                # 保存模型
                model_name = str(format(val_result['acc'], '.2f')) + '_better_model_' + '_totalEpoch_' + \
                             str(args.num_epochs) + '_epoch_' + str(epoch) + '_ps_' + args.pool_sizes \
                              + '_nor_' + str(args.normalize) + '.pth'
                torch.save(model, data_out_dir + model_name)  # 保存 整个模型
            else:
                model_para_name = str(format(val_result['acc'], '.2f')) + '_better_para_' + '_totalEpoch_' + \
                                  str(args.num_epochs) + '_epoch_' + str(epoch) + '_ps_' + args.pool_sizes + '_gs_' + \
                                  initial_gs + '_nor_' + str(args.normalize) + model_name_add + '.pth'
                torch.save(model.state_dict(), data_out_dir + model_para_name)  # 保存模型参数
                # 保存模型
                model_name = str(format(val_result['acc'], '.2f')) + '_better_model_' + '_totalEpoch_' + \
                             str(args.num_epochs) + '_epoch_' + str(epoch) + '_ps_' + args.pool_sizes + '_gs_' + \
                             initial_gs + '_nor_' + str(args.normalize) + model_name_add + '.pth'
                torch.save(model, data_out_dir + model_name)  # 保存 整个模型

        best_val_epochs.append(best_val_result['epoch'])
        best_val_accs.append(best_val_result['acc'])
        if test_dataset is not None:
            test_epochs.append(test_result['epoch'])
            test_accs.append(test_result['acc'])

        logger.info('Epoch: ' + str(epoch) + '-----------------------------')
        logger.info(f'avg_loss {avg_loss}')
        logger.info('Train_result: ' + str(result))
        logger.info('Val result: ' + str(val_result))
        logger.info('Best val result: ' + str(best_val_result))
        logger.info(f'This Epoch consume time: {str(elapsed)} s')

        # 每 20 epoch 保存一次模型
        # if epoch % 20 == 0:
        #     time_mark = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        #     model_para_name = 'para_' + time_mark + '_totalEpoch_' + str(args.num_epochs) + '_epoch_' + str(epoch) + \
        #                       '_ps_' + args.pool_sizes + '_gs_' + initial_gs + '_nor_' + str(args.normalize) + model_name_add + '.pth'
        #     torch.save(model.state_dict(), data_out_dir + model_para_name)  # 保存模型参数
        #     model_name = 'model_' + time_mark + '_totalEpoch_' + str(args.num_epochs) + '_epoch_' + str(epoch) + \
        #                  '_ps_' + args.pool_sizes + '_gs_' + initial_gs + '_nor_' + str(args.normalize) + model_name_add + '.pth'
        #     torch.save(model, data_out_dir + model_name)  # 保存 整个模型
    # 训练完成
    logger.info('train step consume total time: ' + str(time.time()-train_start_time) + 's -----------------------------')
    logger.info('local time: ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    return model, val_accs, test_accs, best_val_result
