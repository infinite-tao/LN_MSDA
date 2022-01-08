import torch
import torch.nn as nn
import networks
from data_load import ImageList
from torch.utils.data import DataLoader
import utils
from sklearn.metrics import *
import numpy as np
import os
from Wasserstein import SinkhornDistance
# from main_IAGCN import DEVICE
import Loss_function

# DEVICE = torch.device('cpu')
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# os.environ['CUDA_VISIBLE_DEVICE'] = '1'

def evaluate(i, config, base_network1, base_network2, base_network3, classifier_gnn, target_test_dset_dict, AUC_tSNE = False, W = None, dynamic_threshold = None):
    base_network1.eval()
    base_network2.eval()
    base_network3.eval()
    classifier_gnn.eval()

    test_res = eval_domain(i, config, target_test_dset_dict, base_network1, base_network2, base_network3,  classifier_gnn, AUC_tSNE, W, dynamic_threshold)
    mlp_f1_score, mlp_f1_score1, mlp_f1_score2, mlp_f1_score3, gnn_f1_score = test_res['mlp_f1_score'], test_res['mlp_f1_score1'], test_res['mlp_f1_score2'], test_res['mlp_f1_score3'], test_res['gnn_f1_score']
    mlp_AUC, mlp_AUC1, mlp_AUC2, mlp_AUC3, gnn_AUC = test_res['mlp_AUC'], test_res['mlp_AUC1'], test_res['mlp_AUC2'], test_res['mlp_AUC3'], test_res['gnn_AUC']
    mlp_Pre,  mlp_Pre1,  mlp_Pre2,  mlp_Pre3,  gnn_Pre = test_res['mlp_pre'], test_res['mlp_pre1'], test_res['mlp_pre2'], test_res['mlp_pre3'], test_res['gnn_pre']
    mlp_Recall, mlp_Recall1, mlp_Recall2, mlp_Recall3, gnn_Recall = test_res['mlp_Recall'], test_res['mlp_Recall1'], test_res['mlp_Recall2'], test_res['mlp_Recall3'], test_res['gnn_Recall']
    mlp_Spec,  mlp_Spec1,  mlp_Spec2,  mlp_Spec3,  gnn_Spec = test_res['mlp_Spec'], test_res['mlp_Spec1'], test_res['mlp_Spec2'], test_res['mlp_Spec3'], test_res['gnn_Spec']
    mlp_accuracy, mlp_accuracy1, mlp_accuracy2, mlp_accuracy3, gnn_accuracy = test_res['mlp_accuracy'], test_res['mlp_accuracy1'], test_res['mlp_accuracy2'], test_res['mlp_accuracy3'], test_res['gnn_accuracy']

    # print out test f1_score for domain
    log_str1 = 'Test f1_score mlp %.4f\tTest f1_score mlp1 %.4f\tTest f1_score mlp2 %.4f\tTest f1_score mlp3 %.4f\tTest f1_score gnn %.4f' \
              % (mlp_f1_score * 100, mlp_f1_score1 * 100, mlp_f1_score2 * 100, mlp_f1_score3 * 100, gnn_f1_score * 100)
    config['out_file'].write(log_str1 + '\n')
    config['out_file'].flush()
    print(log_str1)
    # print out test AUC for domain
    log_str2 = 'Test AUC mlp %.4f\tTest AUC mlp1 %.4f\tTest AUC mlp2 %.4f\tTest AUC mlp3 %.4f\tTest AUC gnn %.4f' \
               % (mlp_AUC * 100, mlp_AUC1 * 100, mlp_AUC2 * 100, mlp_AUC3 * 100, gnn_AUC * 100)
    config['out_file'].write(log_str2 + '\n')
    config['out_file'].flush()
    print(log_str2)
    # print out test Pre for domain
    log_str3 = 'Test Pre mlp %.4f\tTest Pre mlp1 %.4f\tTest Pre mlp2 %.4f\tTest Pre mlp3 %.4f\tTest Pre gnn %.4f' \
               % (mlp_Pre * 100, mlp_Pre1 * 100, mlp_Pre2 * 100, mlp_Pre3 * 100, gnn_Pre * 100)
    config['out_file'].write(log_str3 + '\n')
    config['out_file'].flush()
    print(log_str3)
    # print out test Recall for domain
    log_str4 = 'Test Recall mlp %.4f\tTest Recall mlp1 %.4f\tTest Recall mlp2 %.4f\tTest Recall mlp3 %.4f\tTest Recall gnn %.4f' \
               % (mlp_Recall * 100, mlp_Recall1 * 100, mlp_Recall2 * 100, mlp_Recall3 * 100, gnn_Recall * 100)
    config['out_file'].write(log_str4 + '\n')
    config['out_file'].flush()
    print(log_str4)
    # print out test Spec for domain
    log_str5 = 'Test Spec mlp %.4f\tTest Spec mlp1 %.4f\tTest Spec mlp2 %.4f\tTest Spec mlp3 %.4f\tTest Spec gnn %.4f' \
               % (mlp_Spec * 100, mlp_Spec1 * 100, mlp_Spec2 * 100, mlp_Spec3 * 100, gnn_Spec * 100)
    config['out_file'].write(log_str5 + '\n')
    config['out_file'].flush()
    print(log_str5)
    # print out test accuracy for domain
    log_str = 'Test Accuracy mlp %.4f\tTest Accuracy mlp1 %.4f\tTest Accuracy mlp2 %.4f\tTest Accuracy mlp3 %.4f\tTest Accuracy gnn %.4f' \
              % (mlp_accuracy * 100, mlp_accuracy1 * 100, mlp_accuracy2 * 100, mlp_accuracy3 * 100, gnn_accuracy * 100)
    config['out_file'].write(log_str + '\n')
    config['out_file'].flush()
    print(log_str)

    base_network1.train()
    base_network2.train()
    base_network3.train()
    classifier_gnn.train()


def eval_domain(i, config, test_loader, base_network1, base_network2, base_network3, classifier_gnn, AUC_tSNE, W, dynamic_threshold):
    logits_mlp_all, logits_mlp1_all, logits_mlp2_all, logits_mlp3_all, logits_gnn_all, confidences_mlp1, confidences_mlp2, confidences_mlp3, confidences_mlp, confidences_gnn_numpy, confidences_label, confidences_gnn_all, labels_all = [], [], [], [], [], [], [], [], [], [], [], [], []
    Iter_num = i
    with torch.no_grad():
        iter_test = iter(test_loader)
        for j in range(len(test_loader)):
            data = iter_test.next()
            inputs = data['img'].to(DEVICE)
            # labels = data['target'].to(DEVICE)
            logits_one = torch.ones((len(test_loader), 2), dtype=torch.float32).to(DEVICE)
            feature_one = torch.ones((len(test_loader), 64), dtype=torch.float32).to(DEVICE)
            # forward pass
            feature1, logits_mlp10 = base_network1(inputs)
            feature2, logits_mlp20 = base_network2(inputs)
            feature3, logits_mlp30 = base_network3(inputs)
            if AUC_tSNE:
                feature = torch.div((feature1 + feature2 + feature3), (feature_one + feature_one + feature_one))
                logits_mlp = torch.div((W[0] * logits_mlp10 + W[1] * logits_mlp20 + W[2] * logits_mlp30),
                                       (W[0] * logits_one + W[1] * logits_one + W[2] * logits_one))
                pseudo_labels_m2g = nn.Softmax(dim=1)(logits_mlp).max(1)[0]
                for k in range(len(pseudo_labels_m2g)):
                    if pseudo_labels_m2g[k] > dynamic_threshold:
                        pseudo_labels_m2g[k] = 1
                    else:
                        pseudo_labels_m2g[k] = 0
            else:
                feature = torch.div((feature1 + feature2 + feature3), (feature_one + feature_one + feature_one))
                logits_mlp = torch.div((logits_mlp10 + logits_mlp20 + logits_mlp30),
                                       (logits_one + logits_one + logits_one))
                pseudo_labels_m2g = nn.Softmax(dim=1)(logits_mlp).max(1)[0]
                for k in range(len(pseudo_labels_m2g)):
                    if pseudo_labels_m2g[k] > 0.7:
                        pseudo_labels_m2g[k] = 1
                    else:
                        pseudo_labels_m2g[k] = 0
            pseudo_labels_m2g = torch.unsqueeze(pseudo_labels_m2g, dim=1)
            # check if number of samples is greater than 1

            logits_gnn, _ = classifier_gnn(feature, pseudo_labels_m2g)
            logits_mlp1_all.append(logits_mlp10.cpu())
            logits_mlp2_all.append(logits_mlp20.cpu())
            logits_mlp3_all.append(logits_mlp30.cpu())
            logits_mlp_all.append(logits_mlp.cpu())
            logits_gnn_all.append(logits_gnn.cpu())

            confidences_mlp1.append(nn.Softmax(dim=1)(logits_mlp1_all[-1]).cpu().numpy())
            confidences_mlp2.append(nn.Softmax(dim=1)(logits_mlp2_all[-1]).cpu().numpy())
            confidences_mlp3.append(nn.Softmax(dim=1)(logits_mlp3_all[-1]).cpu().numpy())
            confidences_mlp.append(nn.Softmax(dim=1)(logits_mlp_all[-1]).cpu().numpy())
            confidences_gnn_numpy.append(nn.Softmax(dim=1)(logits_gnn_all[-1]).cpu().numpy())

            confidences_gnn_all.append(nn.Softmax(dim=1)(logits_gnn_all[-1]).max(1)[0])
            labels_all.append(data['target'])
    pred_mlp1 = confidences_mlp1[-1].argmax(1)
    pred_mlp2 = confidences_mlp2[-1].argmax(1)
    pred_mlp3 = confidences_mlp3[-1].argmax(1)
    pred_mlp = confidences_mlp[-1].argmax(1)
    pred_gnn = confidences_gnn_numpy[-1].argmax(1)
    label = torch.squeeze(labels_all[-1]).cpu().numpy()

    confidences_mlp1 = confidences_mlp1[-1][:, 1]
    confidences_mlp2 = confidences_mlp2[-1][:, 1]
    confidences_mlp3 = confidences_mlp3[-1][:, 1]
    confidences_mlp = confidences_mlp[-1][:, 1]
    confidences_gnn_numpy = confidences_gnn_numpy[-1][:, 1]
    # for ROC Curve and tSNE
    if AUC_tSNE:
        np.savetxt(os.path.join(config['output_path'], 'Iterative_{}_tSNE.txt'.format(Iter_num)), logits_gnn_all[-1].cpu().numpy())
        result = []
        for j in range(len(pred_gnn)):
            result0 = []
            result0.append(label[j])
            result0.append(pred_gnn[j])
            result0.append(confidences_gnn_numpy[j])
            result.append(result0)
        output_result = open(
            os.path.join(config['output_path'], 'Iterative_{}_result.xlsx'.format(Iter_num)), 'w', encoding='gbk')
        for hh in range(len(result)):
            for kk in range(len(result[hh])):
                output_result.write(str(result[hh][kk]))
                output_result.write('\t')
            output_result.write('\n')
        output_result.close()

    mlp1_f1_score = f1_score(label, pred_mlp1)
    fpr1, tpr1, thresholds1 = roc_curve(label, pred_mlp1)
    mlp1_AUC = auc(fpr1, tpr1)
    mlp1_Pre = average_precision_score(label, pred_mlp1)
    mlp1_Recall = recall_score(label, pred_mlp1)
    mlp1_confusion_matrix = confusion_matrix(label, pred_mlp1)
    mlp1_TN = mlp1_confusion_matrix[0, 0]
    mlp1_FP = mlp1_confusion_matrix[0, 1]
    mlp1_Spec = mlp1_TN / float(mlp1_TN + mlp1_FP)

    mlp2_f1_score = f1_score(label, pred_mlp2)
    fpr2, tpr2, thresholds2 = roc_curve(label, pred_mlp2)
    mlp2_AUC = auc(fpr2, tpr2)
    mlp2_Pre = average_precision_score(label, pred_mlp2)
    mlp2_Recall = recall_score(label, pred_mlp2)
    mlp2_confusion_matrix = confusion_matrix(label, pred_mlp2)
    mlp2_TN = mlp2_confusion_matrix[0, 0]
    mlp2_FP = mlp2_confusion_matrix[0, 1]
    mlp2_Spec = mlp2_TN / float(mlp2_TN + mlp2_FP)

    mlp3_f1_score = f1_score(label, pred_mlp3)
    fpr3, tpr3, thresholds3 = roc_curve(label, pred_mlp3)
    mlp3_AUC = auc(fpr3, tpr3)
    mlp3_Pre = average_precision_score(label, pred_mlp3)
    mlp3_Recall = recall_score(label, pred_mlp3)
    mlp3_confusion_matrix = confusion_matrix(label, pred_mlp3)
    mlp3_TN = mlp3_confusion_matrix[0, 0]
    mlp3_FP = mlp3_confusion_matrix[0, 1]
    mlp3_Spec = mlp3_TN / float(mlp3_TN + mlp3_FP)

    mlp_f1_score = f1_score(label, pred_mlp)
    fpr, tpr, thresholds = roc_curve(label, pred_mlp)
    mlp_AUC = auc(fpr, tpr)
    mlp_Pre = average_precision_score(label, pred_mlp)
    mlp_Recall = recall_score(label, pred_mlp)
    mlp_confusion_matrix = confusion_matrix(label, pred_mlp)
    mlp_TN = mlp_confusion_matrix[0, 0]
    mlp_FP = mlp_confusion_matrix[0, 1]
    mlp_Spec = mlp_TN / float(mlp_TN + mlp_FP)

    gnn_f1_score = f1_score(label, pred_gnn)
    fpr_gnn, tpr_gnn, thresholds_gnn = roc_curve(label, pred_gnn)
    gnn_AUC = auc(fpr_gnn, tpr_gnn)
    gnn_Pre = average_precision_score(label, pred_gnn)
    gnn_Recall = recall_score(label, pred_gnn)
    gnn_confusion_matrix = confusion_matrix(label, pred_gnn)
    gnn_TN = gnn_confusion_matrix[0, 0]
    gnn_FP = gnn_confusion_matrix[0, 1]
    gnn_Spec = gnn_TN / float(gnn_TN + gnn_FP)

    # concatenate data
    logits_mlp = torch.cat(logits_mlp_all, dim=0)
    logits_mlp1 = torch.cat(logits_mlp1_all, dim=0)
    logits_mlp2 = torch.cat(logits_mlp2_all, dim=0)
    logits_mlp3 = torch.cat(logits_mlp3_all, dim=0)
    logits_gnn = torch.cat(logits_gnn_all, dim=0)
    confidences_gnn = torch.cat(confidences_gnn_all, dim=0)
    labels = torch.cat(labels_all, dim=0)
    # predict class labels
    _, predict_mlp = torch.max(logits_mlp, 1)
    _, predict_mlp1 = torch.max(logits_mlp1, 1)
    _, predict_mlp2 = torch.max(logits_mlp2, 1)
    _, predict_mlp3 = torch.max(logits_mlp3, 1)
    _, predict_gnn = torch.max(logits_gnn, 1)
    labels = torch.squeeze(labels)
    mlp_accuracy = torch.sum(predict_mlp == labels).item() / labels.size(0)
    mlp_accuracy1 = torch.sum(predict_mlp1 == labels).item() / labels.size(0)
    mlp_accuracy2 = torch.sum(predict_mlp2 == labels).item() / labels.size(0)
    mlp_accuracy3 = torch.sum(predict_mlp3 == labels).item() / labels.size(0)
    gnn_accuracy = torch.sum(predict_gnn == labels).item() / labels.size(0)

    # compute mask for high confident samples
    sample_masks_bool = (confidences_gnn > config['lthreshold'])
    sample_masks_idx = torch.nonzero(sample_masks_bool, as_tuple=True)[0].numpy()
    # compute accuracy of pseudo labels
    total_pseudo_labels = len(sample_masks_idx)
    if len(sample_masks_idx) > 0:
        correct_pseudo_labels = torch.sum(predict_gnn[sample_masks_bool] == labels[sample_masks_bool]).item()
        pseudo_label_acc = correct_pseudo_labels / total_pseudo_labels
    else:
        correct_pseudo_labels = -1.
        pseudo_label_acc = -1.
    out = {
        'mlp_f1_score': mlp_f1_score,
        'mlp_f1_score1': mlp1_f1_score,
        'mlp_f1_score2': mlp2_f1_score,
        'mlp_f1_score3': mlp3_f1_score,
        'gnn_f1_score': gnn_f1_score,
        'mlp_AUC': mlp_AUC,
        'mlp_AUC1': mlp1_AUC,
        'mlp_AUC2': mlp2_AUC,
        'mlp_AUC3': mlp3_AUC,
        'gnn_AUC': gnn_AUC,
        'mlp_pre': mlp_Pre,
        'mlp_pre1': mlp1_Pre,
        'mlp_pre2': mlp2_Pre,
        'mlp_pre3': mlp3_Pre,
        'gnn_pre': gnn_Pre,
        'mlp_Recall': mlp_Recall,
        'mlp_Recall1': mlp1_Recall,
        'mlp_Recall2': mlp2_Recall,
        'mlp_Recall3': mlp3_Recall,
        'gnn_Recall': gnn_Recall,
        'mlp_Spec': mlp_Spec,
        'mlp_Spec1': mlp1_Spec,
        'mlp_Spec2': mlp2_Spec,
        'mlp_Spec3': mlp3_Spec,
        'gnn_Spec': gnn_Spec,
        'mlp_accuracy': mlp_accuracy,
        'mlp_accuracy1': mlp_accuracy1,
        'mlp_accuracy2': mlp_accuracy2,
        'mlp_accuracy3': mlp_accuracy3,
        'gnn_accuracy': gnn_accuracy,
        'confidences_gnn': confidences_gnn,
        'pred_cls': predict_gnn.numpy(),
        'sample_masks': sample_masks_idx,
        'sample_masks_cgct': sample_masks_bool.float(),
        'pseudo_label_acc': pseudo_label_acc,
        'correct_pseudo_labels': correct_pseudo_labels,
        'total_pseudo_labels': total_pseudo_labels,
    }
    return out

def train_source(config, base_network1, base_network2, base_network3, classifier_gnn, dset_loaders):
    # define loss functions
    ce_edge = nn.BCELoss(reduction='mean')
    ce_criterion1 = Loss_function.FocalLoss(gamma=0.2, alpha=0.72, size_average=False)
    ce_criterion2 = Loss_function.FocalLoss(gamma=0.2, alpha=0.64, size_average=False)
    ce_criterion3 = Loss_function.FocalLoss(gamma=0.2, alpha=0.78, size_average=False)
    ce_criterion_all = Loss_function.FocalLoss(gamma=0.2, alpha=0.69, size_average=False)

    # configure optimizer
    optimizer_config = config['optimizer']
    parameter_list = base_network1.get_parameters() + base_network2.get_parameters() + base_network3.get_parameters() +\
                     [{'params': classifier_gnn.parameters(), 'lr_mult': 10, 'decay_mult': 2}]
    optimizer = optimizer_config['type'](parameter_list, **(optimizer_config['optim_params']))

    # configure learning rates
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group['lr'])
    schedule_param = optimizer_config['lr_param']

    # start train loop
    base_network1.train()
    base_network2.train()
    base_network3.train()
    classifier_gnn.train()
    len_train_source1 = len(dset_loaders["source1"])
    len_train_source2 = len(dset_loaders["source2"])
    len_train_source3 = len(dset_loaders["source3"])
    for i in range(config['source_iters']):
        optimizer = utils.inv_lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()

        # get input data
        if i % len_train_source1 == 0:
            iter_source1 = iter(dset_loaders["source1"])
        if i % len_train_source2 == 0:
            iter_source2 = iter(dset_loaders["source2"])
        if i % len_train_source3 == 0:
            iter_source3 = iter(dset_loaders["source3"])

        batch_source1 = iter_source1.next()
        inputs_source1, labels_source1 = batch_source1['img'].to(DEVICE), batch_source1['target'].to(DEVICE)
        batch_source2 = iter_source2.next()
        inputs_source2, labels_source2 = batch_source2['img'].to(DEVICE), batch_source2['target'].to(DEVICE)
        batch_source3 = iter_source3.next()
        inputs_source3, labels_source3 = batch_source3['img'].to(DEVICE), batch_source3['target'].to(DEVICE)

        # make forward pass for aggregator and mlp head
        features_source1, logits_mlp1 = base_network1(inputs_source1)
        mlp_loss1 = ce_criterion1(logits_mlp1, labels_source1)
        features_source2, logits_mlp2 = base_network2(inputs_source2)
        mlp_loss2 = ce_criterion1(logits_mlp2, labels_source2)
        features_source3, logits_mlp3 = base_network3(inputs_source3)
        mlp_loss3 = ce_criterion3(logits_mlp3, labels_source3)

        # make forward pass for gnn head
        features_source = torch.cat((features_source1, features_source2, features_source3), 0)
        labels_source = torch.cat((labels_source1, labels_source2, labels_source3), 0)
        logits_gnn, edge_sim = classifier_gnn(features_source, labels_source)
        gnn_loss = ce_criterion_all(logits_gnn, labels_source)

        # total loss and backpropagation
        loss = 0.5*mlp_loss1 + 0.5*mlp_loss2 + 0.5*mlp_loss3 + gnn_loss
        loss.backward()
        optimizer.step()

        # printout train loss
        if i % 1 == 0 or i == config['source_iters'] - 1:
            log_str = 'Iters:(%4d/%d)\tMLP loss1:%.4f\tMLP loss2:%.4f\tMLP loss3:%.4f\tGNN loss:%.4f' % (i,
                                                                                        config['source_iters'],
                                                                                        mlp_loss1.item(),
                                                                                        mlp_loss2.item(),
                                                                                        mlp_loss3.item(),
                                                                                        gnn_loss.item())
            utils.write_logs(config, log_str)
        # evaluate network every test_interval
        if i % config['test_interval'] == config['test_interval'] - 1:
            evaluate(i, config, base_network1, base_network2, base_network3, classifier_gnn, dset_loaders['target_test'])

    return base_network1, base_network2, base_network3, classifier_gnn

def adapt_target_UDS(config, base_network1, base_network2, base_network3, classifier_gnn, dset_loaders):
    # define loss functions
    sinkhornDistance = SinkhornDistance(eps=0.1, max_iter=100)
    ce_criterion1 = Loss_function.FocalLoss(gamma=0.2, alpha=0.72, size_average=False)
    ce_criterion2 = Loss_function.FocalLoss(gamma=0.2, alpha=0.64, size_average=False)
    ce_criterion3 = Loss_function.FocalLoss(gamma=0.2, alpha=0.78, size_average=False)
    ce_criterion_all = Loss_function.FocalLoss(gamma=0.2, alpha=0.69, size_average=False)

    # configure optimizer
    optimizer_config = config['optimizer']
    parameter_list = base_network1.get_parameters() + base_network2.get_parameters() + base_network3.get_parameters() + \
                     [{'params': classifier_gnn.parameters(), 'lr_mult': 10, 'decay_mult': 2}]
    optimizer = optimizer_config['type'](parameter_list, **(optimizer_config['optim_params']))
    # configure learning rates
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group['lr'])
    schedule_param = optimizer_config['lr_param']

    # start train loop
    len_train_source1 = len(dset_loaders['source1'])
    len_train_source2 = len(dset_loaders['source2'])
    len_train_source3 = len(dset_loaders['source3'])
    len_train_target = len(dset_loaders['target_train'])
    # set nets in train mode
    base_network1.train()
    base_network2.train()
    base_network3.train()
    classifier_gnn.train()

    h_t1 = config['hthreshold']
    l_t1 = config['lthreshold']

    for i in range(config['adapt_iters']):
        optimizer = utils.inv_lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()
        # get input data
        if i % len_train_source1 == 0:
            iter_source1 = iter(dset_loaders['source1'])
        if i % len_train_source2 == 0:
            iter_source2 = iter(dset_loaders['source2'])
        if i % len_train_source3 == 0:
            iter_source3 = iter(dset_loaders['source3'])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders['target_train'])

        batch_source1 = iter_source1.next()
        batch_source2 = iter_source2.next()
        batch_source3 = iter_source3.next()
        batch_target = iter_target.next()
        inputs_source1, inputs_source2, inputs_source3, inputs_target = batch_source1['img'].to(DEVICE), batch_source2[
            'img'].to(DEVICE), batch_source3['img'].to(DEVICE), batch_target['img'].to(DEVICE)
        labels_source1, labels_source2, labels_source3, labels_target = batch_source1['target'].to(DEVICE), \
                                                                        batch_source2['target'].to(DEVICE), \
                                                                        batch_source3['target'].to(DEVICE), \
                                                                        batch_target['target'].to(DEVICE)

        # make forward pass for aggregator and mlp head
        features_source1, logits_mlp_source1 = base_network1(inputs_source1)
        features_source2, logits_mlp_source2 = base_network2(inputs_source2)
        features_source3, logits_mlp_source3 = base_network3(inputs_source3)
        features_target1, logits_mlp_target1 = base_network1(inputs_target)
        features_target2, logits_mlp_target2 = base_network2(inputs_target)
        features_target3, logits_mlp_target3 = base_network3(inputs_target)

        logits_one = torch.ones((config['target_batch'], 2), dtype=torch.float32).to(DEVICE)
        feature_one = torch.ones((config['target_batch'], 64), dtype=torch.float32).to(DEVICE)
        WD1, P1, C1 = sinkhornDistance(features_source1, features_target1)
        WD2, P2, C2 = sinkhornDistance(features_source2, features_target2)
        WD3, P3, C3 = sinkhornDistance(features_source3, features_target3)
        print(WD1, WD2, WD3)
        W1 = torch.div(torch.exp(-WD1 ** 2), (torch.exp(-WD1 ** 2) + torch.exp(-WD2 ** 2) + torch.exp(-WD3 ** 2))).to(
            DEVICE)
        W2 = torch.div(torch.exp(-WD2 ** 2), (torch.exp(-WD1 ** 2) + torch.exp(-WD2 ** 2) + torch.exp(-WD3 ** 2))).to(
            DEVICE)
        W3 = torch.div(torch.exp(-WD3 ** 2), (torch.exp(-WD1 ** 2) + torch.exp(-WD2 ** 2) + torch.exp(-WD3 ** 2))).to(
            DEVICE)
        print(W1, W2, W3)
        features_target = torch.div((features_target1 + features_target2 + features_target3),
                                    (feature_one + feature_one + feature_one))
        logits_mlp_target = torch.div((W1 * logits_mlp_target1 + W2 * logits_mlp_target2 + W3 * logits_mlp_target3),
                                      (W1 * logits_one + W2 * logits_one + W3 * logits_one))

        features = torch.cat((features_source1, features_source2, features_source3, features_target), dim=0)
        # compute pseudo-labels for affinity matrix by mlp classifier
        pseudo_labels_m2g = nn.Softmax(dim=1)(logits_mlp_target).max(1)[0]
        dynamic_threshold = h_t1 - ((h_t1 - l_t1) / config['adapt_iters'])
        h_t1 = h_t1 - ((h_t1 - l_t1) / config['adapt_iters'])
        l_t1 = l_t1 + ((h_t1 - l_t1) / config['adapt_iters'])
        print('dynamic_threshold', dynamic_threshold)
        for j in range(len(pseudo_labels_m2g)):
            if pseudo_labels_m2g[j] > dynamic_threshold:
                pseudo_labels_m2g[j] = 1
            else:
                pseudo_labels_m2g[j] = 0
        pseudo_labels_m2g = torch.unsqueeze(pseudo_labels_m2g, dim=1)
        # combine source labels and target pseudo labels for edge_net
        labels_all = torch.cat((labels_source1, labels_source2, labels_source3, pseudo_labels_m2g), dim=0)
        # *** GNN at work ***
        # make forward pass for gnn head
        logits_gnn, edge_sim = classifier_gnn(features, labels_all)
        # compute pseudo-labels for mlp classifier by gcn
        pseudo_labels_g2m = nn.Softmax(dim=1)(logits_gnn[96:, ]).max(1)[0]
        for k in range(len(pseudo_labels_g2m)):
            if pseudo_labels_g2m[k] > dynamic_threshold:
                pseudo_labels_g2m[k] = 1
            else:
                pseudo_labels_g2m[k] = 0
        pseudo_labels_g2m = torch.unsqueeze(pseudo_labels_g2m, dim=1)

        # focal loss for MLP head
        mlp_loss1 = ce_criterion1(torch.cat((logits_mlp_source1, logits_mlp_target1), dim=0),
                                  torch.cat((labels_source1, pseudo_labels_g2m), dim=0))
        mlp_loss2 = ce_criterion2(torch.cat((logits_mlp_source2, logits_mlp_target2), dim=0),
                                  torch.cat((labels_source2, pseudo_labels_g2m), dim=0))
        mlp_loss3 = ce_criterion3(torch.cat((logits_mlp_source3, logits_mlp_target3), dim=0),
                                  torch.cat((labels_source3, pseudo_labels_g2m), dim=0))

        # focal loss for GNN head
        gnn_loss = ce_criterion_all(logits_gnn, labels_all)

        # total loss and backpropagation
        loss = 0.5 * mlp_loss1 + 0.5 * mlp_loss2 + 0.5 * mlp_loss3 + gnn_loss
        loss.backward()
        optimizer.step()
        # printout train loss
        if i % 1 == 0 or i == config['adapt_iters'] - 1:
            log_str = 'Iters:(%4d/%d)\tMLP loss1:%.4f\tMLP loss2:%.4f\tMLP loss3:%.4f\tGNN loss:%.4f' % (i,
                                                                                                         config[
                                                                                                             'adapt_iters'],
                                                                                                         mlp_loss1.item(),
                                                                                                         mlp_loss2.item(),
                                                                                                         mlp_loss3.item(),
                                                                                                         gnn_loss.item())
            utils.write_logs(config, log_str)
        # evaluate network every test_interval
        if i % config['test_interval'] == config['test_interval'] - 1:
            evaluate(i, config, base_network1, base_network2, base_network3, classifier_gnn,
                     dset_loaders['target_test'], AUC_tSNE=True, W=[W1, W2, W3], dynamic_threshold=dynamic_threshold)

    return base_network1, base_network2, base_network3, classifier_gnn
