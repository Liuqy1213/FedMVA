# import copy
# import numpy as np
# import sys
# import torch
# # from sklearn.metrics import accuracy_score as acc, precision_score as pr, recall_score as rc, f1_score as f1
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
#
#
# from torch.optim.lr_scheduler import StepLR
# from sklearn.metrics import roc_auc_score as auc ,confusion_matrix, matthews_corrcoef as mcc
# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt
# from tsne import plot_embedding
#
#
# # def train(args, model, dataset):
# #     model.train()
# #     print('Start Training', file=sys.stderr)
# #     #assert isinstance(model, FEDModel)
# #     best_f1 = 0
# #     best_model = None
# #     if args.optimizer == 'adam':
# #         optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
# #     else:
# #         optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum=0.9, weight_decay=args.weight_decay)
# #     lr_step = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
# #
# #     for epoch_count in range(args.Epoch):
# #         batch_losses = []
# #         num_batches = dataset.initialize_train_batches()
# #         print("num_batches",num_batches)
# #         model.len = num_batches
# #         output_batches_generator = range(num_batches)
# #         for _ in output_batches_generator:
# #             model.zero_grad()
# #             features, targets = dataset.get_next_train_batch()
# #             probabilities, representation, batch_loss = model(example_batch=features, targets=targets)
# #
# #             optimizer.zero_grad()
# #             batch_loss.backward()
# #             optimizer.step()
# #             batch_losses.append(batch_loss.detach().cpu().item())
# #
# #         lr_step.step()
# #         epoch_loss = np.mean(batch_losses).item()
# #
# #         model.train()
# #
# #         print('=' * 100, file=sys.stderr)
# #         print('After epoch %2d Train loss : %10.4f' % (epoch_count, epoch_loss), file=sys.stderr)
# #         print('=' * 100, file=sys.stderr)
# #         if epoch_count % 1 == 0:
# #             vacc, vpr, vrc, vf1, vauc = evaluate(model, dataset)
# #             if vf1 > best_f1:
# #                 best_f1 = vf1
# #                 best_model = copy.deepcopy(model)
# #             if sys.stderr is not None:
# #                 print('Validation Set: Acc: %6.3f\tPr: %6.3f\tRc %6.3f\tF1: %6.3f \tauc: %6.3f' % \
# #                       (vacc, vpr, vrc, vf1, vauc), file=sys.stderr)
# #                 print('-' * 100, file=sys.stderr)
# #     return best_model
# def train(args, model, dataset):
#     model.train()
#     print('Start Training', file=sys.stderr)
#     best_f1 = 0
#     best_model = None
#
#     if args.optimizer == 'adam':
#         optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#     else:
#         optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
#
#     lr_step = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
#
#     # for epoch_count in range(args.Epoch):
#     for epoch_count in range(1):  # 将 Epoch 设置为 1
#         batch_losses = []
#         num_batches = dataset.initialize_train_batches()
#         print("num_batches", num_batches)
#         model.len = num_batches
#         output_batches_generator = range(num_batches)
#
#         for _ in output_batches_generator:
#             model.zero_grad()
#             features, targets = dataset.get_next_train_batch()
#             probabilities, representation, batch_loss = model(example_batch=features, targets=targets)
#
#             optimizer.zero_grad()
#             batch_loss.backward()
#             optimizer.step()
#             batch_losses.append(batch_loss.detach().cpu().item())
#
#         lr_step.step()
#         epoch_loss = np.mean(batch_losses).item()
#         model.train()
#
#         print('=' * 100, file=sys.stderr)
#         print('After epoch %2d Train loss : %10.4f' % (epoch_count, epoch_loss), file=sys.stderr)
#         print('=' * 100, file=sys.stderr)
#
#         if epoch_count % 1 == 0:
#             vacc, vpr, vrc, vf1, vauc = evaluate(model, dataset, average='macro')  # 指定平均方式为 macro
#             if vf1 > best_f1:
#                 best_f1 = vf1
#                 best_model = copy.deepcopy(model)
#             if sys.stderr is not None:
#                 print('Validation Set: Acc: %6.3f\tPr: %6.3f\tRc %6.3f\tF1: %6.3f \tauc: %6.3f' % \
#                       (vacc, vpr, vrc, vf1, vauc), file=sys.stderr)
#                 print('-' * 100, file=sys.stderr)
#
#     return best_model
#
#
# # def test(model,dataset):
# #     model.eval()
# #     with torch.no_grad():
# #         predictions = []
# #         expectations = []
# #         _batch_count = dataset.initialize_test_batches()
# #         batch_generator = range(_batch_count)
# #         for _ in batch_generator:
# #             features, targets = dataset.get_next_test_batch()
# #
# #             probs, _ = model(example_batch=features)
# #             batch_pred = np.argmax(probs.detach().cpu().numpy(), axis=-1).tolist()
# #             batch_tgt = targets.detach().cpu().numpy().tolist()
# #             predictions.extend(batch_pred)
# #             expectations.extend(batch_tgt)
# #         cm_matrix = confusion_matrix(expectations, predictions)
# #
# #         class_names = [0, 1]  # name  of classes
# #         fig, ax = plt.subplots()
# #         tick_marks = np.arange(len(class_names))
# #         plt.xticks(tick_marks, class_names)
# #         plt.yticks(tick_marks, class_names)
# #         # create heatmap
# #         sns.heatmap(pd.DataFrame(cm_matrix), annot=True, cmap="YlGnBu", fmt='g')
# #         ax.xaxis.set_label_position("top")
# #         plt.tight_layout()
# #         plt.title('Confusion matrix', y=1.1)
# #         plt.ylabel('Actual label')
# #         plt.xlabel('Predicted label')
# #         plt.savefig('.pdf')
# #         plt.show()
# #         tn, fp, fn, tp = cm_matrix.ravel()
# #
# #         if (fp + tn) == 0:
# #             fpr = -1.0
# #         else:
# #             fpr = float(fp) / (fp + tn)
# #
# #         if (tp + fn) == 0:
# #             fnr = -1.0
# #         else:
# #             fnr = float(fn) / (tp + fn)
# #
# #         model.train()
# #
# #         print('acc:', acc(expectations, predictions) * 100,
# #              'pre:',pr(expectations, predictions) * 100,
# #             'rec:',rc(expectations, predictions) * 100,
# #             'F1:',f1(expectations, predictions) * 100,
# #             'auc:',auc(expectations, predictions) * 100,
# #             'mcc:',mcc(expectations, predictions) * 100,
# #             "tp", tp,
# #             'fpr:',fpr * 100,
# #             'fnr:',fnr * 100)
#
#
#         print('acc:', acc(expectations, predictions) * 100,
#               'pre:', pr(expectations, predictions, average='macro') * 100,
#               'rec:', rc(expectations, predictions, average='macro') * 100,
#               'F1:', f1(expectations, predictions, average='macro') * 100,
#               'auc:', auc(expectations, predictions) * 100,
#               'mcc:', mcc(expectations, predictions) * 100,
#               "tp", tp,
#               'fpr:', fpr * 100,
#               'fnr:', fnr * 100)
#
#
#
# def evaluate(model, dataset, average='macro'):
#     model.eval()
#     with torch.no_grad():
#         predictions = []
#         expectations = []
#         probabilities = []  # 存储概率分布
#         _batch_count = dataset.initialize_valid_batches()
#         batch_generator = range(_batch_count)
#         for _ in batch_generator:
#             features, targets = dataset.get_next_valid_batch()
#             probs, _ = model(example_batch=features)
#             batch_pred = np.argmax(probs.detach().cpu().numpy(), axis=-1).tolist()
#             batch_probs = probs.detach().cpu().numpy().tolist()  # 取概率分布
#             batch_tgt = targets.detach().cpu().numpy().tolist()
#
#             predictions.extend(batch_pred)
#             expectations.extend(batch_tgt)
#             probabilities.extend(batch_probs)  # 存储每个样本的概率分布
#
#         # 计算评估指标，传递 average 参数
#         accuracy = accuracy_score(expectations, predictions)
#         precision = precision_score(expectations, predictions, average=average)
#         recall = recall_score(expectations, predictions, average=average)
#         f1 = f1_score(expectations, predictions, average=average)
#
#         # 使用概率分布计算 AUC
#         auc = roc_auc_score(expectations, probabilities, average=average, multi_class='ovr')
#
#     model.train()
#     return accuracy, precision, recall, f1, auc
#
#
#
#

import copy
import numpy as np
import sys
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    matthews_corrcoef
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import roc_auc_score as auc, matthews_corrcoef as mcc
from sklearn.metrics import matthews_corrcoef as mcc
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tsne import plot_embedding
import xlrd

from xlutils.copy import copy as copy_xl
import os



# Training Function
def train(args, model, dataset):
    model.train()




    print('Start Training', file=sys.stderr)
    best_f1 = 0
    best_model = None

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay) if args.optimizer == 'adam' else torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    lr_step = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # for epoch_count in range(args.Epoch):
    for epoch_count in range(50):  # Limit epoch to 1 as requested
        batch_losses = []
        num_batches = dataset.initialize_train_batches()
        print("num_batches", num_batches)
        model.len = num_batches
        output_batches_generator = range(num_batches)

        for _ in output_batches_generator:
            model.zero_grad()
            features, targets = dataset.get_next_train_batch()
            probabilities, representation, batch_loss = model(example_batch=features, targets=targets)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            batch_losses.append(batch_loss.detach().cpu().item())

        lr_step.step()
        epoch_loss = np.mean(batch_losses).item()
        model.train()

        print('=' * 100, file=sys.stderr)
        print('After epoch %2d Train loss : %10.4f' % (epoch_count, epoch_loss), file=sys.stderr)
        print('=' * 100, file=sys.stderr)

        if epoch_count % 1 == 0:
            vacc, vpr, vrc, vf1, mcc = evaluate(model, dataset, average='macro')  # Macro average for multi-class
            if vf1 > best_f1:
                best_f1 = vf1
                best_model = copy.deepcopy(model)
            print(
                'Validation Set: Acc: %6.3f\tPr: %6.3f\tRc %6.3f\tF1: %6.3f \tMCC: %6.3f' % (vacc, vpr, vrc, vf1, mcc),
                file=sys.stderr)
            print('-' * 100, file=sys.stderr)

    return best_model


# Evaluation Function
def evaluate(model, dataset, average='macro'):
    model.eval()
    with torch.no_grad():
        predictions, expectations, probabilities = [], [], []
        _batch_count = dataset.initialize_valid_batches()
        batch_generator = range(_batch_count)
        for _ in batch_generator:
            features, targets = dataset.get_next_valid_batch()
            probs, _ = model(example_batch=features)
            batch_pred = np.argmax(probs.detach().cpu().numpy(), axis=-1).tolist()
            batch_probs = probs.detach().cpu().numpy().tolist()
            batch_tgt = targets.detach().cpu().numpy().tolist()

            predictions.extend(batch_pred)
            expectations.extend(batch_tgt)
            probabilities.extend(batch_probs)

        accuracy = accuracy_score(expectations, predictions)
        precision = precision_score(expectations, predictions, average=average)
        recall = recall_score(expectations, predictions, average=average)
        f1 = f1_score(expectations, predictions, average=average)
        mcc_value = matthews_corrcoef(expectations, predictions)

    model.train()
    return accuracy, precision, recall, f1, mcc_value

def append_to_excel(words, filename):
    '''
    追加数据到excel
    :param words: 【item】 [{},{}]格式
    :param filename: 文件名
    :return:
    '''
    try:
        # 打开excel
        word_book = xlrd.open_workbook(filename)
        # 获取所有的sheet表单。
        sheets = word_book.sheet_names()
        # 获取第一个表单
        work_sheet = word_book.sheet_by_name(sheets[0])
        # 获取已经写入的行数
        old_rows = work_sheet.nrows
        # 获取表头信息
        heads = work_sheet.row_values(0)
        # 将xlrd对象变成xlwt
        new_work_book = copy_xl(word_book)
        # 添加内容
        new_sheet = new_work_book.get_sheet(0)
        i = old_rows
        for item in words:
            for j in range(len(heads)):
                new_sheet.write(i, j, item[heads[j]])
            i += 1
        new_work_book.save(filename)
        print('追加成功！')
    except Exception as e:
        print('追加失败！', e)

# Testing Function with Multi-Class Support
def test(model, dataset):
    model.eval()
    # # 输出模型结构
    # print(model)
    with torch.no_grad():
        predictions, probabilities, expectations = [], [], []
        _batch_count = dataset.initialize_test_batches()
        batch_generator = range(_batch_count)

        for _ in batch_generator:
            features, targets = dataset.get_next_test_batch()

            # 获取模型的预测概率
            probs, _ = model(example_batch=features)
            probabilities.extend(probs.detach().cpu().numpy())  # 概率输出
            batch_pred = np.argmax(probs.detach().cpu().numpy(), axis=-1).tolist()  # 类别预测
            batch_tgt = targets.detach().cpu().numpy().tolist()

            predictions.extend(batch_pred)
            expectations.extend(batch_tgt)

        # 混淆矩阵（多分类）
        cm_matrix = confusion_matrix(expectations, predictions)
        print("Confusion Matrix:\n", cm_matrix)

        # 绘制混淆矩阵
        class_names = [0, 1, 2]  # 根据实际类别调整
        fig, ax = plt.subplots()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
        sns.heatmap(pd.DataFrame(cm_matrix), annot=True, cmap="YlGnBu", fmt='g')
        ax.xaxis.set_label_position("top")
        plt.tight_layout()
        plt.title('Confusion Matrix', y=1.1)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.savefig('confusion_matrix.pdf')
        plt.show()

        vacc, vpr, vrc, vf1, mcc = evaluate(model, dataset, average='macro')
        print(
            'Validation Set: Acc: %6.3f\tPr: %6.3f\tRc %6.3f\tF1: %6.3f \tMCC: %6.3f' % (vacc, vpr, vrc, vf1, mcc),
            file=sys.stderr)
        words = [{ 'vacc': vacc*100, 'vpr': vpr*100, 'vrc': vrc*100, 'vf1': vf1*100, 'mcc': mcc*100}]
        # 追加内容
        append_to_excel(words=words, filename='data7.xls')



        # # 计算多分类的指标
        # acc_value = accuracy_score(expectations, predictions) * 100
        # pr_value = precision_score(expectations, predictions, average='macro') * 100
        # rc_value = recall_score(expectations, predictions, average='macro') * 100
        # f1_value = f1_score(expectations, predictions, average='macro') * 100
        # # auc_value = roc_auc_score(expectations, probabilities, multi_class='ovr') * 100  # 使用概率计算AUC
        # mcc_value = mcc(expectations, predictions) * 100
        #
        # print(
        #     f'Accuracy: {acc_value:.2f}, Precision: {pr_value:.2f}, Recall: {rc_value:.2f}, F1: {f1_value:.2f}, MCC: {mcc_value:.2f}')


def show_representation(model,dataset,k):
    model.eval()
    with torch.no_grad():
        representations = []
        expected_targets = []
        _batch_count = dataset.initialize_train_batches()
        batch_generator = range(_batch_count)
        for _ in batch_generator:
            iterator_values = dataset.get_next_train_batch()
            features, targets = iterator_values[0], iterator_values[1]
            _, repr= model(example_batch=features)
            repr = repr.detach().cpu().numpy()
            #print(repr.shape)
            representations.extend(repr.tolist())
            expected_targets.extend(targets.numpy().tolist())
        model.train()
        print(np.array(representations).shape)
        print(np.array(expected_targets).shape)
        plot_embedding(representations, expected_targets, title=str(k))