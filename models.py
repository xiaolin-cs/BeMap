import os
from os.path import *
import random
import json
import dgl
import numpy as np
import torch
from dgl.nn.pytorch.conv import GraphConv, GATConv, SAGEConv
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from scipy import sparse
import time
from copy import deepcopy
from tqdm import trange
import tqdm


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(GCN, self).__init__()

        self.gc1 = GraphConv(nfeat, nhid)
        self.gc2 = GraphConv(nhid, nclass)
        # self.fc = nn.Linear(nhid, nclass)
        # self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc2 = GraphConvolution(nhid, nclass)
        # self.gc2 = GraphConvolution(nhid, 2 * nhid)
        # self.gc3 = GraphConvolution(2 * nhid, nclass)
        self.dropout = dropout

    def forward(self, g, x):
        x = F.relu(self.gc1(g, x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(g, x)
        # x = self.fc(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc3(x, adj)
        # return F.log_softmax(x, dim=1)
        return x


class GCN_check(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_check, self).__init__()
        print(nfeat, nhid, nclass, dropout)
        self.gc1 = GraphConv(nfeat, nhid)
        self.fc = nn.Linear(nhid, nclass)

    def forward(self, g, x):
        x = F.relu(self.gc1(g, x))
        x = self.fc(x)
        return x

class GCN2_check(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN2_check, self).__init__()
        print(nfeat, nhid, nclass, dropout)
        self.gc1 = GraphConv(nfeat, nhid, norm='both')
        self.gc2 = GraphConv(nhid, nhid, norm='both')
        self.fc = nn.Linear(nhid, nclass)

    def forward(self, g, x):
        x = F.relu(self.gc1(g, x))
        x = F.relu(self.gc2(g, x))
        x = self.fc(x)
        return x


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(nfeat, nhid)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(nhid, nclass)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_heads):
        super(GAT, self).__init__()
        self.num_heads = num_heads
        self.conv1 = GATConv(in_feats, hidden_feats, num_heads)
        self.conv2 = GATConv(hidden_feats*num_heads, hidden_feats, num_heads)
        self.fc = nn.Linear(hidden_feats, 1)
        self.dropout = nn.Dropout(0.5)
    def forward(self, g, x):
        h = self.conv1(g, x).flatten(1)
        # print()
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(g, h).mean(1)
        h = self.fc(h)
        return h

class GCN_binary(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_binary, self).__init__()
        self.body = GCN_Body(nfeat, nhid, dropout)
        self.fc = nn.Linear(nhid, nclass)

    def forward(self, g, x):
        x, _ = self.body(g, x)
        x = self.fc(x)
        return x


# def GCN(nn.Module):
class GCN_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN_Body, self).__init__()

        self.gc1 = GraphConv(nfeat, nhid)
        self.gc2 = GraphConv(nhid, nhid)
        self.dropout = nn.Dropout(dropout)
        self.retain_feats = []

    def forward(self, g, x):
        out = F.relu(self.gc1(g, x))
        x = out
        x = self.dropout(x)
        x = self.gc2(g, x)
        # x = self.dropout(x)
        return x, out


class BeMap(nn.Module):
    def __init__(self, graph, attrs, num_layers, lam=0.5, beta=0.25, save_num=4):
        """
        the model to generate the fair subgraph for each epoch
        :param graph: the original graph
        :param attrs: the attributes of the nodes
        :param num_layers: the balance score is calculated within the num_layers hop neighbors
        :param beta: the ratio of the saved nodes when neighbors are from same group
        :param save_num: the minimum number of the saved nodes when neighbors are from same groups
        """
        super(BeMap, self).__init__()
        self.graph = dgl.from_scipy(graph)
        self.beta = beta
        self.lam = lam
        self.save_num = save_num
        self.init_neighbor(attrs)
        self.init_edge()
        self.neighbor_attrs = attrs
        self.sp_val = np.zeros((2, self.graph.num_nodes())).astype(dtype=np.int_)
        for layer in range(num_layers):
            self.sp_value_update()
        self.sp_value_inverse()


    def init_neighbor(self, attrs):
        """
        initialize the neighbors of each node, i.e., self.neighbor
        :param attrs: the attributes of the nodes
        """
        # the neighbors of each node, self.neighbor[i][0] (a list) is the neighbors of i from the group of 0
        self.neighbor = [[[], []] for i in range(self.graph.num_nodes())]

        src_nodes, dst_nodes = self.graph.edges()
        self.tril_neigh = [[] for i in range(self.graph.num_nodes())]  # the lower triangle neighbors
        self.triu_neigh = [[] for i in range(self.graph.num_nodes())]  # the upper triangle neighbors
        for i in range(int(src_nodes.shape[0])):
            if src_nodes[i] > dst_nodes[i]:
                self.tril_neigh[dst_nodes[i]].append(src_nodes[i])
                self.neighbor[int(dst_nodes[i])][int(attrs[src_nodes[i]])].append(int(src_nodes[i]))
            elif src_nodes[i] < dst_nodes[i]:
                self.triu_neigh[dst_nodes[i]].append(src_nodes[i])
                self.neighbor[int(dst_nodes[i])][int(attrs[src_nodes[i]])].append(int(src_nodes[i]))
        print('=' * 5 + 'finish neighbor init' + '=' * 5)

    def init_edge(self):
        """
        initialize the edge dictionary, i.e., self.edges_dict
        the format of the key is 'src/dst', the value is the index of the edge
        """
        edges = self.graph.edges()
        self.edges_dict = {str(int(edges[1][i])) + '/' + str(int(edges[0][i])): i
                           for i in range(int(edges[0].shape[0]))}
        print('=' * 5 + 'finish edge init' + '=' * 5)

    def sp_value_update(self):
        """
        update the sp_value of each node,
        and sp_value is the difference on the numbers of the neighbors within L hops from different groups
        """
        sample_val = np.copy(self.sp_val)  # the sp_value of the neighbors within L-1 hops
        for dst in range(self.graph.num_nodes()):
            ret = 2 * int(self.neighbor_attrs[dst]) - 1  # [0, 1]->[-1, 1], init the bias of the current node
            for src in self.tril_neigh[dst]:
                ret += (2 * int(self.neighbor_attrs[src]) - 1 + int(self.sp_val[0, src]))  # add the bias in the L hops
            sample_val[0, dst] = ret  # bias of the neighbors with sensitive attribute in tril

            ret = 2 * int(self.neighbor_attrs[dst]) - 1  # [0, 1]->[-1, 1], init the bias of the current node
            for src in self.triu_neigh[dst]:
                ret += (2 * int(self.neighbor_attrs[src]) - 1 + int(self.sp_val[1, src]))  # add the bias in the L hops
            sample_val[1, dst] = ret  # bias of the neighbors with sensitive attribute in triu

        self.sp_val = sample_val
        print('=' * 5 + 'sample value update' + '=' * 5)

    def sp_value_inverse(self, alpha=1):
        self.sp_val = 1 / (np.abs(self.sp_val[0, :] + self.sp_val[1, :]) + 1) ** alpha  # the inverse of the sp_value
        del self.tril_neigh, self.triu_neigh  # release the memory
        self.neigh_sp_val = [[[], []] for i in range(self.graph.num_nodes())]  # the sampling prob of the neighbors
        for node in range(self.graph.num_nodes()):
            for attr in [0, 1]:
                if len(self.neighbor[node][attr]) > 0:
                    # normalize the sp_value of the neighbors to get the sampling prob
                    self.neigh_sp_val[node][attr] = self.sp_val[self.neighbor[node][attr]]
                    self.neigh_sp_val[node][attr] /= np.sum(self.neigh_sp_val[node][attr])
        print('=' * 5 + 'finish sample value init' + '=' * 5)

    def get_subgraph(self, epoch):
        """
        get a fair subgraph for the current epoch
        """
        subedge = self.init_rand_subgraph(epoch)
        return dgl.edge_subgraph(self.graph, subedge)

    def init_rand_subgraph(self, epoch):
        """
        create a fair balanced subgraph for the current epoch
        """
        random.seed(2 * epoch + 1)
        edge_idx_subgraph = []
        for i in range(int(self.graph.nodes().shape[0])):

            s = int(self.neighbor_attrs[i])
            num_0, num_1 = len(self.neighbor[i][0]), len(self.neighbor[i][1])
            num_0_slf = num_0 if s else num_0 + 1
            num_1_slf = num_1 + 1 if s else num_1
            if num_0 == 0 and num_1 == 0:  # no neighbors, then add self loop
                edge_idx_subgraph.append(self.edges_dict[str(i) + '/' + str(i)])
            elif num_0_slf == 0:  # all the neighbors are from the group of 1
                k = min(max(self.save_num, int((num_1 + 3) * self.beta)), num_1)
                neigh_set = np.random.choice(np.arange(num_1), size=k, replace=False, p=self.neigh_sp_val[i][1])
                qualify_neighbor = np.array(self.neighbor[i][1])[neigh_set].tolist()
                qualify_neighbor.append(i)
                assert len(qualify_neighbor) > 0
                edge_idx_subgraph.extend([self.edges_dict[str(i) + '/' + str(v)] for v in qualify_neighbor])

            elif num_1_slf == 0:  # all the neighbors are from the group of 0
                k = min(max(self.save_num, int((num_0 + 3) * self.beta)), num_0)
                neigh_set = np.random.choice(np.arange(num_0), size=k, replace=False, p=self.neigh_sp_val[i][0])
                qualify_neighbor = np.array(self.neighbor[i][0])[neigh_set].tolist()
                qualify_neighbor.append(i)
                assert len(qualify_neighbor) > 0
                edge_idx_subgraph.extend([self.edges_dict[str(i) + '/' + str(v)] for v in qualify_neighbor])

            else:  # the neighbors are from both groups
                # todo: min_num = min(num_0, num_1)
                max_num = min(num_0 / (self.lam + 1e-4), num_1 / (1.0001-self.lam))
                min0 = round(max_num * self.lam)
                min1 = round(max_num * (1-self.lam))
                min0 = min0 if s else min0 - 1
                min1 = min1 - 1 if s else min1

                if min0 == num_0:
                    qual0 = self.neighbor[i][0]
                elif min0 <= 0:
                    qual0 = []
                else:
                    neigh_set = np.random.choice(np.arange(num_0), size=min0, replace=False, p=self.neigh_sp_val[i][0])
                    assert len(neigh_set) > 0, 'error: empty set of neighbors with s=0, set is ' + str(neigh_set)
                    qual0 = np.array(self.neighbor[i][0])[neigh_set].tolist()

                if min1 == num_1:
                    qual1 = self.neighbor[i][1]
                elif min1 <= 0:
                    qual1 = []
                else:
                    neigh_set = np.random.choice(np.arange(num_1), size=min1, replace=False, p=self.neigh_sp_val[i][1])
                    assert len(neigh_set) > 0, 'error: empty set of neighbors with s=1, set is ' + str(neigh_set)
                    qual1 = np.array(self.neighbor[i][1])[neigh_set].tolist()

                qualify_neighbor = qual0 + qual1 + [i]
                assert len(qualify_neighbor) > 0
                edge_idx_subgraph.extend([self.edges_dict[str(i) + '/' + str(v)] for v in qualify_neighbor])

        return edge_idx_subgraph


class BeMap_GCN(BeMap):
    def __init__(self, nfeat, nhid, nclass, dropout, graph, attrs, beta=0.25, lam=0.5, save_num=4,
                use_dropout=False, layer_num=2, use_cuda=False, norm='both'):
        super(BeMap_GCN, self).__init__(graph, attrs, lam=lam, num_layers=layer_num, beta=beta, save_num=save_num)
        self.use_cuda = use_cuda
        self.use_dropout = use_dropout
        self.gc1 = GraphConv(nfeat, nhid, norm=norm)
        self.gc2 = GraphConv(nhid, nhid, norm=norm)
        self.fc = nn.Linear(nhid, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, epoch):
        if epoch == -1:
            sg = self.graph
        else:
            sg = self.get_subgraph(epoch)
        if self.use_cuda:
            sg = sg.to(torch.device('cuda'))
        x = F.relu(self.gc1(sg, x))
        if self.use_dropout:
            x = self.dropout(x)
        x = self.gc2(sg, x)
        x = self.fc(x)
        return x


class BeMap_GAT(BeMap):
    def __init__(self, nfeat, nhid, nclass, graph, attrs, dropout, beta=1 / 4, use_cuda=False, save_num=4,
                 rand_num=10, num_layers=2, lam=0.5):
        super(BeMap_GAT, self).__init__(graph, attrs, beta=beta, save_num=save_num, num_layers=num_layers, lam=lam)
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(nfeat, nhid, num_heads=4, activation=F.elu))
        for i in range(1, 2):
            self.layers.append(GATConv(nhid * 4, nhid, num_heads=4, activation=F.elu))
        self.fc = nn.Linear(nhid * 4, nclass)
        self.dropout = dropout
        self.use_cuda = use_cuda

    def forward(self, h, epoch):
        if epoch == -1:
            sg = self.graph
        else:
            sg = self.get_subgraph(epoch)
        if self.use_cuda:
            sg = sg.to(torch.device('cuda'))
        # list of hidden representation at each layer (including the input layer)

        for i, conv in enumerate(self.layers):
            # print('i', i, h.shape)
            h = conv(sg, h).flatten(1)
            if i != len(self.layers) - 1:
                h = F.dropout(h, self.dropout)

        h = self.fc(h)
        return h  # score_over_layer
