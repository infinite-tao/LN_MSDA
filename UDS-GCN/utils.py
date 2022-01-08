import os
import torch
from torch.utils.data import DataLoader

import networks
import data_load
from data_load import ImageList
import pickle as pc

def inv_lr_scheduler(optimizer, iter_num, gamma, power, lr=0.001, weight_decay=0.0005):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = lr * (1 + gamma * iter_num) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = weight_decay * param_group['decay_mult']
        i += 1
    return optimizer

def all_list(path):
    with open(path, 'rb') as f:
        all_list = pc.load(f)

    return all_list

def build_config(args):
    config = {
        # 'aggregator': args.aggregator,
        'ndomains': 2,
        'list_path': args.list_path,
        'hthreshold': args.hthreshold,
        'lthreshold': args.lthreshold,
        'edge_features': args.edge_features,
        'source_iters': args.source_iters,
        'adapt_iters': args.adapt_iters,
        'test_interval': args.test_interval,
        'source_batch' : args.source_batch,
        'target_batch' : args.target_batch,
        'num_workers': args.num_workers,
        'lambda_node': args.lambda_node,
        'random_dim': args.rand_proj,
        'use_UDS_mask': args.use_UDS_mask if 'use_UDS_mask' in args else False,
    }
    # optimizer params
    config['optimizer'] = {
        'type': torch.optim.SGD,
        'optim_params': {
            'lr': args.lr,
             'momentum': 0.9,
             'weight_decay': args.wd,
             'nesterov': True,
             },
        'lr_type': 'inv',
        'lr_param': {
            'lr': args.lr,
            'gamma': 0.001,
            'power': 0.75,
        },
    }
    # dataset params
    config['dataset'] = args.dataset
    list_all = all_list(config['list_path'])

    # set task
    if config['dataset'] == 'Task1':
        config['data_root'] = '/home/ubuntu/zhangyongtao/UDS-GCN/Stage_two/Task1'
        config['output_path'] = '/home/ubuntu/zhangyongtao/UDS-GCN/result/task1'
        config['source1'] = list_all[0]['train'][0:74]
        config['source2'] = list_all[0]['train'][74:113]
        config['source3'] = list_all[0]['train'][113:223]
        config['target'] = list_all[0]['val']
        config['test'] = list_all[0]['val']

    elif config['dataset'] == 'Task2':
        config['data_root'] = '/home/ubuntu/zhangyongtao/UDS-GCN/Stage_two/Task2'
        config['output_path'] = '/home/ubuntu/zhangyongtao/UDS-GCN/result/task2'
        config['source1'] = list_all[1]['train'][0:74]
        config['source2'] = list_all[1]['train'][74:113]
        config['source3'] = list_all[1]['train'][113:164]
        config['target'] = list_all[1]['val']
        config['test'] = list_all[1]['val']

    elif config['dataset'] == 'Task3':
        config['data_root'] = '/home/ubuntu/zhangyongtao/UDS-GCN/Stage_two/Task3'
        config['output_path'] = '/home/ubuntu/zhangyongtao/UDS-GCN/result/task3'
        config['source1'] = list_all[2]['train'][0:74]
        config['source2'] = list_all[2]['train'][74:184]
        config['source3'] = list_all[2]['train'][184:235]
        config['target'] = list_all[2]['val']
        config['test'] = list_all[2]['val']

    elif config['dataset'] == 'Task4':
        config['data_root'] = '/home/ubuntu/zhangyongtao/UDS-GCN/Stage_two/Task4'
        config['output_path'] = '/home/ubuntu/zhangyongtao/UDS-GCN/result/task4'
        config['source1'] = list_all[3]['train'][0:39]
        config['source2'] = list_all[3]['train'][39:149]
        config['source3'] = list_all[3]['train'][149:200]
        config['target'] = list_all[3]['val']
        config['test'] = list_all[3]['val']
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')

    # create output folder and log file
    if not os.path.exists(config['output_path']):
        os.system('mkdir -p '+config['output_path'])
    config['out_file'] = open(os.path.join(config['output_path'], 'log.txt'), 'w')

    # print pout config values
    config['out_file'].write(str(config)+'\n')
    config['out_file'].flush()

    return config


def build_data(config):
    dsets = {
        'target_train': {},
        'target_test': {},
    }
    dset_loaders = {
        'target_train': {},
        'target_test': {},
    }
    train_bs = config['source_batch']
    target_bs = config['target_batch']
    if config['dataset'] == 'Task1':
        test_bs = 51
    elif config['dataset'] == 'Task2':
        test_bs = 110
    elif config['dataset'] == 'Task3':
        test_bs = 39
    elif config['dataset'] == 'Task4':
        test_bs = 74

    # source dataloader
    dsets['source1'] = ImageList(image_root=config['data_root'], image_list_root=config['source1'])
    dset_loaders['source1'] = DataLoader(dsets['source1'], batch_size=train_bs, shuffle=True,
                                        num_workers=config['num_workers'], drop_last=True, pin_memory=True)
    dsets['source2'] = ImageList(image_root=config['data_root'], image_list_root=config['source2'])
    dset_loaders['source2'] = DataLoader(dsets['source2'], batch_size=train_bs, shuffle=True,
                                         num_workers=config['num_workers'], drop_last=True, pin_memory=True)
    dsets['source3'] = ImageList(image_root=config['data_root'], image_list_root=config['source3'])
    dset_loaders['source3'] = DataLoader(dsets['source3'], batch_size=train_bs, shuffle=True,
                                         num_workers=config['num_workers'], drop_last=True, pin_memory=True)

    # target dataloader
    # create train and test datasets for a target domain
    dsets['target_train'] = ImageList(image_root=config['data_root'],
                                                 image_list_root=config['target'],
                                                 use_UDS_mask=config['use_UDS_mask'])
    dsets['target_test'] = ImageList(image_root=config['data_root'],
                                                image_list_root=config['test'],
                                                use_UDS_mask=config['use_UDS_mask'])
    # create train and test dataloaders for a target domain
    dset_loaders['target_train'] = DataLoader(dataset=dsets['target_train'],
                                                         batch_size=target_bs, shuffle=True,
                                                         num_workers=config['num_workers'], drop_last=True)
    dset_loaders['target_test'] = DataLoader(dataset=dsets['target_test'],
                                                        batch_size=test_bs, num_workers=config['num_workers'],
                                                        pin_memory=True)

    return dsets, dset_loaders


def write_logs(config, log_str):
    config['out_file'].write(log_str + '\n')
    config['out_file'].flush()
    print(log_str)
