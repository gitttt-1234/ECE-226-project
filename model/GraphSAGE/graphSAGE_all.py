# # adopted from 'GraphNorm: A Principled Approach to Accelerating Graph Neural Network Training, cai et al.'
# # import os
# # import sys
# # sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import SAGEConv

# from .readouts.basic_readout import readout_function

# """
# Base paper: https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf
# """

# class GraphSAGE(nn.Module):

#     def __init__(self, n_feat, n_class, n_layer, agg_hidden, fc_hidden, dropout, readout, device,
#     aggregation='mean'):
        
#         super(GraphSAGE, self).__init__()
        
#         self.n_layer = n_layer
#         self.dropout = dropout
#         self.readout = readout
#         self.aggregation = aggregation
#         self.device = device
#         self.readout_dim = agg_hidden * n_layer
        
#         # Graph sage layer
#         self.graph_sage_layers = []
#         for i in range(n_layer):
#             if i == 0:
#                 sage = SAGEConv(n_feat, agg_hidden).to(device)
#             else:
#                 sage = SAGEConv(agg_hidden, agg_hidden).to(device)
#             sage.aggr = self.aggregation
#             self.graph_sage_layers.append(sage)
        
#         if self.aggregation == 'max':
#             self.fc_max = nn.Linear(agg_hidden, agg_hidden)
        
#         # Fully-connected layer
#         self.fc1 = nn.Linear(self.readout_dim, fc_hidden)
#         self.fc2 = nn.Linear(fc_hidden, n_class)
    
#     def preprocessing(self, edge_matrix_list, x_list, node_count_list, edge_matrix_count_list):
#         total_edge_matrix_list = []
#         start_edge_matrix_list = []
#         end_edge_matrix_list = []
#         batch_list = []
#         total_x_list = []
        
#         max_value = torch.tensor(0).to(self.device)
#         for i, edge_matrix in enumerate(edge_matrix_list):
#             for a in range(edge_matrix_count_list[i]):
#                 start_edge_matrix_list.append(max_value + edge_matrix[a][0])
#                 end_edge_matrix_list.append(max_value + edge_matrix[a][1])
#             if max_value < max_value + edge_matrix[edge_matrix_count_list[i] - 1][0]:
#                 max_value = max_value + edge_matrix[edge_matrix_count_list[i] - 1][0]
#         total_edge_matrix_list.append(start_edge_matrix_list)
#         total_edge_matrix_list.append(end_edge_matrix_list)
        
#         for i in range(len(x_list)):
#             for a in range(node_count_list[i]):
#                 batch_list.append(i)
#                 total_x_list.append(x_list[i][a].cpu().numpy())
        
#         return torch.tensor(total_edge_matrix_list).long().to(self.device), torch.tensor(batch_list).float().to(self.device), torch.tensor(total_x_list).float().to(self.device)
              
#     def forward(self, data):
#         x, adj = data[:2]
#         edge_matrix_list = data[6]
#         node_count_list = data[7]
#         edge_matrix_count_list = data[8]
#         total_edge_matrix_list, batch_list, total_x_list = self.preprocessing(edge_matrix_list, x, node_count_list, edge_matrix_count_list)
        
#         x_list = []
#         x = total_x_list
        
#         for i in range(self.n_layer):
           
#            # Graph sage layer
#            x = F.relu(self.graph_sage_layers[i](x, total_edge_matrix_list))
#            if self.aggregation == 'max':
#                x = torch.relu(self.fc_max(x))
           
#            # Dropout
#            if i != self.n_layer - 1:
#                x = F.dropout(x, p=self.dropout, training=self.training)
             
#            x_list.append(x)
        
#         x = torch.cat(x_list, dim=1)
           
#         # Readout
#         x = readout_function(x, self.readout, batch=batch_list, device=self.device)
#         x = x.reshape(adj.size()[0], self.readout_dim)
        
#         # Fully-connected layer
#         x = F.relu(self.fc1(x))
#         x = F.softmax(self.fc2(x))

#         return x

#     def __repr__(self):
#         layers = ''
        
#         for i in range(self.n_layer):
#             layers += str(self.graph_sage_layers[i]) + '\n'
#         layers += str(self.fc1) + '\n'
#         layers += str(self.fc2) + '\n'
#         return layers

# def GS(dataset):
#     return GraphSAGE(n_feat=dataset.dim_nfeats, n_class=dataset.gclasses, n_layer=5, agg_hidden=64,fc_hidden=100, dropout=0.5, readout='avg', device='cuda')
    

import dgl
import dgl.nn as dglnn
import sklearn.linear_model as lm
import sklearn.metrics as skm
import torch as th
import torch.functional as F
import torch.nn as nn
import tqdm
from model.GIN.readout import AvgPooling


class GraphSAGE(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, dropout):
        super().__init__()
        self.init(in_feats, n_hidden, n_classes, n_layers, dropout)

    def init(
        self, in_feats, n_hidden, n_classes, n_layers, dropout
    ):
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        if n_layers > 1:
            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
            for i in range(1, n_layers - 1):
                self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
            self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        else:
            self.layers.append(dglnn.SAGEConv(in_feats, n_classes, "mean"))
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.linears_prediction = nn.Linear(n_hidden, n_classes)
        self.pool = AvgPooling()

    def forward(self, g, h):

        hidden_rep = []
        for i in range(self.n_layers-1):
            
            x = h
            g = g.to("cuda")

            h = self.layers[i](g,h)
            
            h = self.activation(h)

            hidden_rep.append(h)
          
        score_over_layer = 0

        pooled_h = self.pool(g, hidden_rep[-1])
        score_over_layer += self.dropout(self.linears_prediction(pooled_h))

        return score_over_layer


    # def forward(self, blocks, x):

    #     h = x
    #     for l, (layer, block) in enumerate(zip(self.layers, blocks)):
    #         h = layer(block, h)
    #         if l != len(self.layers) - 1:
    #             h = self.activation(h)
    #             h = self.dropout(h)
    #     return h

    def inference(self, g, x, device, batch_size, num_workers):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.

        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        for l, layer in enumerate(self.layers):
            y = th.zeros(
                g.num_nodes(),
                self.n_hidden if l != len(self.layers) - 1 else self.n_classes,
            )

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DataLoader(
                g,
                th.arange(g.num_nodes()).to(g.device),
                sampler,
                device=device if num_workers == 0 else None,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers,
            )

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
        return y


def compute_acc_unsupervised(emb, labels, train_nids, val_nids, test_nids):
    """
    Compute the accuracy of prediction given the labels.
    """
    emb = emb.cpu().numpy()
    labels = labels.cpu().numpy()
    train_nids = train_nids.cpu().numpy()
    train_labels = labels[train_nids]
    val_nids = val_nids.cpu().numpy()
    val_labels = labels[val_nids]
    test_nids = test_nids.cpu().numpy()
    test_labels = labels[test_nids]

    emb = (emb - emb.mean(0, keepdims=True)) / emb.std(0, keepdims=True)

    lr = lm.LogisticRegression(multi_class="multinomial", max_iter=10000)
    lr.fit(emb[train_nids], train_labels)

    pred = lr.predict(emb)
    f1_micro_eval = skm.f1_score(val_labels, pred[val_nids], average="micro")
    f1_micro_test = skm.f1_score(test_labels, pred[test_nids], average="micro")
    return f1_micro_eval, f1_micro_test
   

def GS(dataset):
  return GraphSAGE(in_feats=dataset.dim_nfeats, n_hidden=32, n_classes=dataset.gclasses, n_layers=4, dropout=0.3)
  #return GraphSAGE(n_feat=dataset.dim_nfeats, n_class=dataset.gclasses, n_layer=5, agg_hidden=64,fc_hidden=100, dropout=0.5, readout='avg', device='cuda')
  
