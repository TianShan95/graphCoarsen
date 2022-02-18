import torch
import time
from torch.autograd import Variable
import torch.nn as nn
from graphModel.evaluate import evaluate
import hiddenlayer as hl


# 单独训练 图网络的 训练函数
def train(data_out_dir, history, canvas, dataset, model, args, same_feat=True, val_dataset=None, test_dataset=None,
          mask_nodes=True, log_dir=None, device='cpu'):
    # writer_batch_idx = [0, 3, 6, 9]

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                 weight_decay=args.weight_decay)
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

            time2 = time.time()

            ypred,_ = model(h0, adj, adj_pooled_list, batch_num_nodes, batch_num_nodes_list, pool_matrices_dic)
            # else:
            #     ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
            # if not args.method == 'soft-assign' or not args.linkpred:
            loss = model.loss(ypred, label)
            # else:
            #     loss = model.loss(ypred, label, adj, batch_num_nodes)
            loss.backward()

            time3 = time.time()

            nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()
            iter += 1
            avg_loss += loss

            # 训练时 实时显示 loss 变化曲线

            history.log((epoch, batch_idx), train_loss=loss)
            # loss_list.append(loss)
            with canvas:
                canvas.draw_plot(history['train_loss'])
                print(f'step: {batch_idx}, graph_size: {args.gs}')
            # print(epoch)

        print('epoch 结束')
        avg_loss /= batch_idx + 1
        elapsed = time.time() - begin_time
        # if writer is not None:
        #     writer.add_scalar('loss/avg_loss', avg_loss, epoch)
        #     if args.linkpred:
        #         writer.add_scalar('loss/linkpred_loss', model.link_loss, epoch)

        eval_time = time.time()
        print('正在评估在训练数据上的精度')
        result = evaluate(dataset, model, args, name='Train', max_num_examples=100, device=device)
        eval_time2 = time.time()
        train_accs.append(result['acc'])
        train_epochs.append(epoch)
        if val_dataset is not None:
            val_result = evaluate(val_dataset, model, args, name='Validation', device=device)
            val_accs.append(val_result['acc'])
        if val_result['acc'] > best_val_result['acc'] - 1e-7:
            best_val_result['acc'] = val_result['acc']
            best_val_result['epoch'] = epoch
            best_val_result['loss'] = avg_loss
            if test_dataset is not None:
                test_result = evaluate(test_dataset, model, args, name='Test', device=device)
                test_result['epoch'] = epoch

        best_val_epochs.append(best_val_result['epoch'])
        best_val_accs.append(best_val_result['acc'])
        if test_dataset is not None:
            test_epochs.append(test_result['epoch'])
            test_accs.append(test_result['acc'])
        if epoch % 50 == 0:
            print('Epoch: ', epoch, '----------------------------------')
            print('Train_result: ', result)
            print('Val result: ', val_result)
            print('Best val result', best_val_result)

            if log_dir is not None:
                with open(log_dir, 'a') as f:
                    f.write('Epoch: ' + str(epoch) + '-----------------------------\n')
                    f.write('Train_result: ' + str(result) + '\n')
                    f.write('Val result: ' + str(val_result) + '\n')
                    f.write('Best val result: ' + str(best_val_result) + '\n')

        end_time = time.time()
        model_name = args.pool_sizes + '_' + str(args.gs) + '.pth'
        torch.save(model.state_dict(), data_out_dir + model_name)
    return model, val_accs, test_accs, best_val_result


