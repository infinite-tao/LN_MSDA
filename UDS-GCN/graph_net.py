import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class NodeNet(nn.Module):
    def __init__(self, in_features, num_features, device, ratio=(2, 1)):
        super(NodeNet, self).__init__()
        num_features_list = [num_features * r for r in ratio]
        self.device = device
        # define layers
        layer_list = OrderedDict()
        for l in range(len(num_features_list)):
            layer_list['conv{}'.format(l)] = nn.Conv2d(
                in_channels=num_features_list[l - 1] if l > 0 else in_features * 2,
                out_channels=num_features_list[l],
                kernel_size=1, bias=False
            )
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=num_features_list[l])
            if l < (len(num_features_list) - 1):
                layer_list['relu{}'.format(l)] = nn.LeakyReLU()
        self.network = nn.Sequential(layer_list)

    def forward(self, node_feat, edge_feat):
        node_feat = node_feat.unsqueeze(dim=0)
        num_tasks = node_feat.size(0)
        num_data = node_feat.size(1)
        # get eye matrix (batch_size x node_size x node_size) only use inter dist.
        diag_mask = 1.0 - torch.eye(num_data).unsqueeze(0).repeat(num_tasks, 1, 1).to(self.device)
        # set diagonal as zero and normalize
        edge_feat = F.normalize(edge_feat * diag_mask, p=1, dim=-1)
        # compute attention and aggregate
        aggr_feat = torch.bmm(edge_feat.squeeze(1), node_feat)
        node_feat = torch.cat([node_feat, aggr_feat], -1).transpose(1, 2)
        # non-linear transform
        node_feat = self.network(node_feat.unsqueeze(-1)).transpose(1, 2)
        node_feat = node_feat.squeeze(-1).squeeze(0)
        return node_feat


class EdgeNet(nn.Module):
    def __init__(self, in_features, num_features, device, ratio=(2, 1)):
        super(EdgeNet, self).__init__()
        num_features_list = [num_features * r for r in ratio]
        self.device = device
        # define layers
        layer_list = OrderedDict()
        for l in range(len(num_features_list)):
            layer_list['conv{}'.format(l)] = nn.Conv2d(
                in_channels=num_features_list[l-1] if l > 0 else in_features,
                out_channels=num_features_list[l], kernel_size=1, bias=False
            )
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=num_features_list[l])
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()
        # add final similarity kernel
        layer_list['conv_out'] = nn.Conv2d(in_channels=num_features_list[-1],
                                           out_channels=1, kernel_size=1)
        self.sim_network = nn.Sequential(layer_list)

    def forward(self, node_feat, node_label):
        node_feat = node_feat.unsqueeze(dim=0)
        num_tasks = node_feat.size(0)
        num_data = node_feat.size(1)
        x_i = node_feat.unsqueeze(2)
        x_j = torch.transpose(x_i, 1, 2)
        x_ij = torch.abs(x_i - x_j)
        x_ij = torch.transpose(x_ij, 1, 3)
        node_edge = torch.zeros((node_label.size(0), node_label.size(0))).to(DEVICE)
        for i in range(node_label.size(0)):
            for j in range(node_label.size(0)):
                if node_label[i, 0] == node_label[j, 0]:
                    node_edge[i][j] = 1
        # compute similarity/dissimilarity (batch_size x feat_size x num_samples x num_samples)
        sim_val = torch.sigmoid(self.sim_network(x_ij)).squeeze(1).squeeze(0)
        # normalize affinity matrix
        force_edge_feat = torch.eye(num_data).unsqueeze(0).repeat(num_tasks, 1, 1).to(self.device)
        edge_feat = node_edge * sim_val + force_edge_feat
        edge_feat = edge_feat + 1e-6
        edge_feat = edge_feat / torch.sum(edge_feat, dim=1).unsqueeze(1)
        return edge_feat, sim_val


class ClassifierGNN(nn.Module):
    def __init__(self, in_features, edge_features, nclasses, device):
        super(ClassifierGNN, self).__init__()

        self.edge_net = EdgeNet(in_features=in_features,
                                num_features=edge_features,
                                device=device)
        # set edge to node
        self.node_net = NodeNet(in_features=in_features,
                                num_features=nclasses,
                                device=device)
        # mask value for no-gradient edges
        self.mask_val = -1


    def forward(self, init_node_feat, init_node_label):
        #  compute normalized and not normalized affinity matrix
        edge_feat, edge_sim = self.edge_net(init_node_feat, init_node_label)
        # compute node features and class logits
        logits_gnn = self.node_net(init_node_feat, edge_feat)
        return logits_gnn, edge_sim
