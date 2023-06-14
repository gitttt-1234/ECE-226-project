import argparse

import dgl.nn as dglnn
import dgl
from dgl import AddSelfLoop

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch.conv import GraphConv
from model.GIN.readout import SumPooling, AvgPooling, MaxPooling

class GAT_3(nn.Module):

    def __init__(self, in_size, hid_size, out_size, heads,final_dropout):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        print("head size: ",len(heads))
        self.gat_layers.append(
            dgl.nn.GATConv(
                in_size,
                hid_size,
                heads[0],
                feat_drop=0.5,
                attn_drop=0.6,
                activation=F.elu,
            )
        )
        self.gat_layers.append(
            dgl.nn.GATConv(
                hid_size * heads[0],
                hid_size,
                heads[1],
                feat_drop=0.5,
                attn_drop=0.6,
                activation=F.elu,
            )
        )
        
        self.pool=SumPooling()
        self.fc1 = nn.Linear(hid_size, out_size)
        self.drop = nn.Dropout(final_dropout)
        

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            if i == 1:  
                h = h.mean(1)
            else:  
                h = h.flatten(1)
        pooled_h = self.pool(g, h)
        res = self.fc1(pooled_h)
        return self.drop(res)


class GAT_4(nn.Module):

    def __init__(self, in_size, hid_size, out_size, heads,final_dropout):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        print("head size: ",len(heads))
        # three-layer GAT
        self.gat_layers.append(
            dgl.nn.GATConv(
                in_size,
                hid_size,
                heads[0],
                feat_drop=0.5,
                attn_drop=0.6,
                activation=F.elu,
            )
        )
        self.gat_layers.append(
            dgl.nn.GATConv(
                hid_size * heads[0],
                hid_size,
                heads[1],
                feat_drop=0.5,
                attn_drop=0.6,
                activation=F.elu,
            )
        )
        
        self.gat_layers.append(
            dgl.nn.GATConv(
                hid_size * heads[1],
                hid_size,
                heads[2],
                feat_drop=0.5,
                attn_drop=0.6,
                activation=F.elu,
            )
        )
        self.pool=SumPooling()
        self.fc1 = nn.Linear(hid_size, out_size)
        self.drop = nn.Dropout(final_dropout)
        

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            if i == 2:  
                h = h.mean(1)
            else:  
                h = h.flatten(1)
        pooled_h = self.pool(g, h)
        res = self.fc1(pooled_h)
        return self.drop(res)


def GAT_3_32(dataset):
    return GAT_3(in_size=dataset.dim_nfeats, hid_size=32, out_size=dataset.gclasses, heads=[6,4],final_dropout=0.5)
    
def GAT_4_32(dataset):
    return GAT_4(in_size=dataset.dim_nfeats, hid_size=32, out_size=dataset.gclasses, heads=[6,4,3],final_dropout=0.5)