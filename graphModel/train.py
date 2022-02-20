import torch
import time
from torch.autograd import Variable
import torch.nn as nn
from graphModel.evaluate import evaluate
import hiddenlayer as hl
import copy


# 单独训练 图网络的 训练函数
def train(data_out_dir, log_out_file, history, canvas, dataset, model, args, same_feat=True, val_dataset=None, test_dataset=None,
          mask_nodes=True, log_dir=None, device='cpu'):
    # writer_batch_idx = [0, 3, 6, 9]

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
    for epoch in range(args.num_epochs):
        begin_time = time.time()
        avg_loss = 0.0
        model.train()
        for batch_idx, data in enumerate(dataset):

            time1 = time.time()
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

            # time2 = time.time() - begin_time # 准备一次数据的时间

            ypred,_ = model(h0, adj, adj_pooled_list, batch_num_nodes, batch_num_nodes_list, pool_matrices_dic)
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
            if log_out_file:
                with open(log_out_file, 'a') as f:
                    f.write('Epoch: ' + str(epoch) + '-----------------------------\n')
                    f.write('loss: ' + str(loss) + '\n')
                    f.close()

            # 训练时 实时显示 loss 变化曲线

            history.log((epoch, batch_idx), train_loss=loss)
            # loss_list.append(loss)
            with canvas:
                canvas.draw_plot(history['train_loss'])
            print(f'Epoch: {epoch} step: {batch_idx}, loss: {loss}:.8f graph_size: {args.gs}, Normalize: {args.normalize}')
            # print(epoch)



        avg_loss /= batch_idx + 1
        print(f'epoch {epoch} 结束 平均loss是 {avg_loss}')
        elapsed = time.time() - begin_time  # 一个 epoch 耗费的时间
        # if writer is not None:
        #     writer.add_scalar('loss/avg_loss', avg_loss, epoch)
        #     if args.linkpred:
        #         writer.add_scalar('loss/linkpred_loss', model.link_loss, epoch)

        eval_time = time.time()
        print('正在评估在训练数据上的精度')
        result = evaluate(dataset, model, args, name='Train', max_num_examples=100, device=device)
        eval_time2 = time.time()
        # 在训练集上的精度
        train_accs.append(result['acc'])
        print(f"在训练集的精度是: {result['acc']}")
        train_epochs.append(epoch)
        if val_dataset is not None:
            val_result = evaluate(val_dataset, model, args, name='Validation', device=device)
            val_accs.append(val_result['acc'])
            print(f"在验证集的精度是: {val_result['acc']}")
        else:
            print(f'无 验证集')

        # 记录在 验证集最好的 精度 epoch 和 loss
        if val_result['acc'] > best_val_result['acc'] - 1e-7:
            best_val_result['acc'] = val_result['acc']
            best_val_result['epoch'] = epoch
            best_val_result['loss'] = avg_loss
            if test_dataset is not None:
                test_result = evaluate(test_dataset, model, args, name='Test', device=device)
                test_result['epoch'] = epoch
                print(f"在测试集的精度是: {test_result['acc']}")
            else:
                print(f'无 测试集')

            # 保存在验证集上精度最好的模型
            time_mark = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            model_para_name = str(format(val_result['acc'], '.6f')) + '_better_para_' + time_mark + '_totalEpoch_' + str(args.num_epochs) + '_epoch_' + str(epoch) + '_ps_' + args.pool_sizes + '_gs_' + str(args.gs) + '_nor_' + str(args.normalize) + '.pth'
            torch.save(model.state_dict(), data_out_dir + model_para_name)  # 保存模型参数
            model_name = str(val_result['acc']) + '_better_model_' + time_mark + '_totalEpoch_' + str(args.num_epochs) + '_epoch_' + str(epoch) + '_ps_' + args.pool_sizes + '_gs_' + str(args.gs) + '_nor_' + str(args.normalize) + '.pth'
            torch.save(model, data_out_dir + model_name)  # 保存 整个模型

        best_val_epochs.append(best_val_result['epoch'])
        best_val_accs.append(best_val_result['acc'])
        if test_dataset is not None:
            test_epochs.append(test_result['epoch'])
            test_accs.append(test_result['acc'])
        # if epoch % 50 == 0:
        print('Epoch: ', epoch, '----------------------------------')
        print('avg_loss', avg_loss)
        print('Train_result: ', result)
        print('Val result: ', val_result)
        print('Best val result', best_val_result)


        with open(log_out_file, 'a') as f:
            f.write('Epoch: ' + str(epoch) + '-----------------------------\n')
            f.write(f'avg_loss {avg_loss}\n')
            f.write('Train_result: ' + str(result) + '\n')
            f.write('Val result: ' + str(val_result) + '\n')
            f.write('Best val result: ' + str(best_val_result) + '\n')
            f.write(f'This Epoch consume time: {str(elapsed)} s')
            f.close()

        end_time = time.time()


        # 每 20 epoch 保存一次模型
        if epoch % 20 == 0:
            time_mark = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            model_para_name = 'para_' + time_mark + '_totalEpoch_' + str(args.num_epochs) + '_epoch_' + str(epoch) + '_ps_' + args.pool_sizes + '_gs_' + str(args.gs) + '_nor_' + str(args.normalize) + '.pth'
            torch.save(model.state_dict(), data_out_dir + model_para_name)  # 保存模型参数
            model_name = 'model_' + time_mark + '_totalEpoch_' + str(args.num_epochs) + '_epoch_' + str(epoch) + '_ps_' + args.pool_sizes + '_gs_' + str(args.gs) + '_nor_' + str(args.normalize) + '.pth'
            torch.save(model, data_out_dir + model_name)  # 保存 整个模型

    with open(log_out_file, 'a') as f:
        f.write('train step consume total time: ' + str(time.time()-train_start_time) + 's -----------------------------\n')
        f.write('local time: ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        f.close()

    return model, val_accs, test_accs, best_val_result
