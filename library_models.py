from dgl.nn import SAGEConv, GATConv, GraphConv
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class DynGraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, lookback, feat_drop=0):
        super(DynGraphSAGE, self).__init__()

        self.gnn = GraphSAGE(in_feats, h_feats, feat_drop)

    def forward(self, g_list, in_feat_list):
        h_list = []
        for idx, g in enumerate(g_list):

            h = self.gnn(g, in_feat_list)

            h_list.append(h)

        return h_list


class DynGATModel(nn.Module):
    def __init__(self, in_feats, h_feats, lookback, feat_drop=0):
        super(DynGATModel, self).__init__()


        self.gnn = GATModel(in_feats, h_feats, feat_drop)

    def forward(self, g_list, in_feat_list):
        h_list = []
        for idx, g in enumerate(g_list):

            h = self.gnn(g, in_feat_list)

            h_list.append(h)

        return h_list


class DynGCNModel(nn.Module):
    def __init__(self, in_feats, h_feats, lookback, feat_drop=0):
        super(DynGCNModel, self).__init__()


        self.gnn = GCNModel(in_feats, h_feats, feat_drop)

    def forward(self, g_list, in_feat_list):
        h_list = []
        for idx, g in enumerate(g_list):

            h = self.gnn(g, in_feat_list)

            h_list.append(h)

        return h_list



class GCNModel(nn.Module):
    def __init__(self, in_feats, h_feats, feat_drop=0):
        super(GCNModel, self).__init__()
        norm = nn.BatchNorm1d(h_feats, affine=True)
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, feat_drop=0, agg_type='pool'):
        super(GraphSAGE, self).__init__()
        norm = nn.BatchNorm1d(h_feats, affine=True)
        self.conv1 = SAGEConv(in_feats, h_feats, agg_type, norm=norm)
        self.conv2 = SAGEConv(h_feats, h_feats, agg_type, feat_drop=feat_drop)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


class GATModel(nn.Module):
    def __init__(self, in_feats, h_feats, feat_drop=0):
        super(GATModel, self).__init__()

        self.conv1 = GATConv(in_feats, h_feats, num_heads=3)
        self.conv2 = GATConv(h_feats * 3, h_feats, num_heads=1, feat_drop=feat_drop)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = h.view(-1, h.size(1) * h.size(2))
        h = F.relu(h)
        h = self.conv2(g, h)
        h = h.squeeze()
        return h



class LSTMModel(nn.Module):
    def __init__(self, gnn_dim, hidden_dim, fcn_dim, num_layers, lookback, dropout=0.2, use_gru=False):
        super(LSTMModel, self).__init__()
        self.input_dim = gnn_dim * 2
        self.hidden_dim = hidden_dim
        self.lookback = lookback
        if use_gru:
            self.rnn = nn.GRU(self.input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        else:
            self.rnn = nn.LSTM(self.input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.tanh = nn.Tanh()
        self.leakyReLU = nn.LeakyReLU(0.1)
        self.W1 = nn.Linear(self.hidden_dim, fcn_dim)

        self.W2 = nn.Linear(fcn_dim, 1)
        self.dropout_layer = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(self.input_dim)
        self.norm2 = nn.LayerNorm(fcn_dim)
        self.bn1 = nn.BatchNorm1d(fcn_dim)

    def apply_edges(self, edges):
        embed = torch.cat([edges.src['h_0'], edges.dst['h_0']], 1)
        for i in range(1, self.lookback):
            h = torch.cat([edges.src[f'h_{i}'], edges.dst[f'h_{i}']], 1)
            embed = torch.cat([embed, h], dim=1)

        return {'embed': embed}

    def forward(self, g, h_list):
        with g.local_scope():
            for idx, h in enumerate(h_list):
                g.ndata[f'h_{idx}'] = h
            g.apply_edges(self.apply_edges)

            h = g.edata['embed']
            h = h[torch.randperm(h.size()[0])].view(-1, self.lookback, self.input_dim)

            h, _ = self.rnn(h)

            h = self.W1(h[:, -1])

            h = self.dropout_layer(h)
            h = F.relu(h)

            h = self.W2(h)
            return h.squeeze(1)


class FCN_MODEL(nn.Module):
    def __init__(self, gnn_dim, hidden_dim, lookback, dropout=0.2):
        super(FCN_MODEL, self).__init__()
        self.input_dim = gnn_dim * 2
        self.hidden_dim = hidden_dim
        self.lookback = lookback

        self.tanh = nn.Tanh()
        self.leakyReLU = nn.LeakyReLU(0.1)
        self.W1 = nn.Linear(self.lookback * self.input_dim, hidden_dim)

        self.W3 = nn.Linear(hidden_dim, 1)
        self.dropout_layer = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(self.input_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def apply_edges(self, edges):

        importance = np.arange(1, self.lookback + 1)
        importance = importance / sum(importance)
        embed = importance[0] * torch.cat([edges.src['h_0'], edges.dst['h_0']], 1)
        for i in range(1, self.lookback):
            h = importance[i] * torch.cat([edges.src[f'h_{i}'], edges.dst[f'h_{i}']], 1)
            embed = torch.cat([embed, h], dim=1)

        return {'embed': embed}

    def forward(self, g, h_list):
        with g.local_scope():
            for idx, h in enumerate(h_list):
                g.ndata[f'h_{idx}'] = h
            g.apply_edges(self.apply_edges)

            h = g.edata['embed']

            h = self.W1(self.tanh(h))
            h = self.norm2(h)
            h = self.dropout_layer(h)
            h = F.relu(h)

            h = self.W3(h)
            return h.squeeze(1)