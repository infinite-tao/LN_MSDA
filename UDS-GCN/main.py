import os
import random
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

import graph_net
import utils
import trainer
import networks
import data_load


DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# os.environ['CUDA_VISIBLE_DEVICE'] = '1'

parser = argparse.ArgumentParser(description='Unsupervised domain selective graph convolutional network')
# model args
parser.add_argument('--rand_proj', type=int, default=1024, help='random projection dimension')
parser.add_argument('--edge_features', type=int, default=32, help='graph edge features dimension')
parser.add_argument('--save_models', action='store_true', help='whether to save encoder, mlp and gnn models')
# dataset args
parser.add_argument('--dataset', type=str, default='Task4', help='dataset used')
parser.add_argument('--list_path', default='/home/ubuntu/zhangyongtao/UDS-GCN/splits_final.pkl', help='list of all task')

parser.add_argument('--data_root', type=str, default='/home/ubuntu/zhangyongtao/UDS-GCN/Stage_two/Task4', help='path to dataset root')
# training args
parser.add_argument('--source_iters', type=int, default=100, help='number of source pre-train iters')
parser.add_argument('--adapt_iters', type=int, default=500, help='number of iters for a curriculum adaptation')
parser.add_argument('--test_interval', type=int, default=1, help='interval of two continuous test phase')

parser.add_argument('--output_dir', type=str, default='/home/ubuntu/zhangyongtao/UDS-GCN/result/task4', help='output directory')
parser.add_argument('--source_batch', type=int, default=32)
parser.add_argument('--target_batch', type=int, default=32)
# optimization args
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
parser.add_argument('--lambda_node', default=1.0, type=float, help='node classification loss weight')
parser.add_argument('--hthreshold', type=float, default=0.9, help='threshold for pseudo labels')
parser.add_argument('--lthreshold', type=float, default=0.5, help='threshold for pseudo labels')
parser.add_argument('--seed', type=int, default=0, help='random seed for training')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataloaders')


def main(args):
    # fix random seed
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # create train configurations
    args.use_UDS_mask = True  # used in UDS-GCN for pseudo label mask in target datasets
    config = utils.build_config(args)
    # prepare data
    dsets, dset_loaders = utils.build_data(config)

    # set domain-specific feature extract network
    Aggregator1 = networks.Feature_Aggregator1()
    Aggregator1 = Aggregator1.to(DEVICE)
    Aggregator2 = networks.Feature_Aggregator2()
    Aggregator2 = Aggregator2.to(DEVICE)
    Aggregator3 = networks.Feature_Aggregator3()
    Aggregator3 = Aggregator3.to(DEVICE)

    # set GNN classifier
    classifier_gnn = graph_net.ClassifierGNN(in_features=64,
                                             edge_features=config['edge_features'],
                                             nclasses=2,
                                             device=DEVICE)
    classifier_gnn = classifier_gnn.to(DEVICE)
    print(classifier_gnn)

    # train on source domain
    log_str = '==> Step 1: Pre-training on the source dataset ...'
    utils.write_logs(config, log_str)

    base_network1, base_network2, base_network3, classifier_gnn = trainer.train_source(config, Aggregator1, Aggregator2, Aggregator3, classifier_gnn, dset_loaders)

    log_str = '==> Finished pre-training on source!\n'
    utils.write_logs(config, log_str)

    # run adaptation episodes
    log_str = '==> Starting the adaptation'
    utils.write_logs(config, log_str)
    ######### Step 2: adaptation stage##########
    base_network1, base_network2, base_network3, classifier_gnn = trainer.adapt_target_UDS(config, base_network1, base_network2, base_network3, classifier_gnn,
                                                             dset_loaders)

    log_str = '==> Finishing adaptation episode!\n'
    utils.write_logs(config, log_str)

    # save models
    if args.save_models:
        torch.save(base_network1.cpu().state_dict(), os.path.join(config['output_path'], 'base_network1.pth.tar'))
        torch.save(base_network2.cpu().state_dict(), os.path.join(config['output_path'], 'base_network2.pth.tar'))
        torch.save(base_network3.cpu().state_dict(), os.path.join(config['output_path'], 'base_network3.pth.tar'))
        torch.save(classifier_gnn.cpu().state_dict(), os.path.join(config['output_path'], 'classifier_gnn.pth.tar'))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
