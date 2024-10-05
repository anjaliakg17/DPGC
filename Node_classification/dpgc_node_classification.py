import os
import torch
from locale import currency
import math
from pickle import FALSE
from re import L
from unicodedata import name
import numpy as np
import random
import torch
import torch.nn.functional as F
import networkx as nx
import torch_geometric
import math
from sklearn.manifold import TSNE
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse, get_laplacian
from torch_geometric.data import Data
from torch_geometric.datasets import CitationFull
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import Planetoid


import json
import scipy as sp
from scipy.sparse import csr_matrix
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
import time
import warnings
warnings.filterwarnings("ignore")

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

##########################################################################
#                           Utility
########################################################################
def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask


def random_splits_citation(data, num_classes):
    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)
    rest_index = torch.cat([i[20:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]
    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[500:1500], size=data.num_nodes)

    return data
def split(data, num_classes, split_percent):
    indices = []
    num_test = (int)(data.num_nodes * split_percent / num_classes)
    for i in range(num_classes):
        index = (data.y == i).nonzero().reshape(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)
    
    test_index = torch.cat([i[:num_test] for i in indices], dim=0)
    val_index = torch.cat([i[num_test:int(num_test*1.5)] for i in indices], dim=0)
    train_index = torch.cat([i[int(num_test*1.5):] for i in indices], dim=0)
    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(test_index, size=data.num_nodes)
    return data

########################################################################################################################
def hashed_values(data, no_of_hash, feature_size):
    # WL Kernel
    g_adj = to_dense_adj(data.edge_index, edge_attr= data.edge_attr)[0]
    wlf = (1/2)*torch.matmul(g_adj,data.x) + data.x
    wlx = torch.cat((data.x, wlf), dim = 1)
    # 
    Wl = torch.FloatTensor(no_of_hash, 2*feature_size).uniform_(0,1) #.normal_(0,1
    Bin_values = torch.matmul(wlx, Wl.T)

    # Wl = torch.FloatTensor(no_of_hash, feature_size).uniform_(0,1) #.normal_(0,1
    # Bin_values = torch.matmul(wlf, Wl.T)
    return Bin_values
#######################################################################################################################
def partition(list_bin_width, Bin_values, sigma, no_of_hash):
    summary_dict = {}
    for bin_width in list_bin_width:
        bias = torch.tensor([np.random.normal(loc=0.0, scale=sigma) for i in range(no_of_hash)])#.to(device)
        temp = torch.floor((1/bin_width)*(Bin_values + bias))#.to(device)
        cluster, _ = torch.mode(temp, dim = 1)
        dict_hash_indices = {}

        no_nodes = Bin_values.shape[0]

        for i in range(no_nodes):
            dict_hash_indices[i] = int(cluster[i]) #.to('cpu')

        summary_dict[bin_width] = dict_hash_indices

    return summary_dict
#########################################################################################################################
def get_key(val, g_coarsened):
  KEYS = []
  for key, value in g_coarsened.items():
    if val == value:
      KEYS.append(key)
  return len(KEYS),KEYS

def one_hot(x, class_count):
    return torch.eye(class_count)[x, :]

#################################################################################################################
#                                                DPGC                                                          #
#################################################################################################################
def DPGC(data, dataset, no_of_hash, binwidth):
    feature_size = dataset.num_features
    num_classes = dataset.num_classes 
    Bin_values = hashed_values(data, no_of_hash, feature_size)
    sigma = 1 

    summary_dict = {}
    list_bin_width = [binwidth]
    summary_dict = partition(list_bin_width, Bin_values, sigma, no_of_hash)

    for bin_width in list_bin_width:
      time4 = time.time()
      print(f'  Checking ratio for bin width: {bin_width}')
      current_bin_width_summary = summary_dict[bin_width]

      values = current_bin_width_summary.values()

      unique_values = set(values)
      rr = 1 - len(unique_values)/len(values)
      print("   Reduction Ratio:"+"{:.2%}".format(rr))
      
      dict_blabla ={}

      C_diag = torch.zeros(len(unique_values), device= device)
      help_count = 0

      for v in unique_values:
          C_diag[help_count],dict_blabla[help_count] = get_key(v, current_bin_width_summary)
          help_count += 1


      #print(f"total unique values :{len(unique_values)}")
      P_hat = torch.zeros((data.num_nodes, len(unique_values)), device= device)

      for x in dict_blabla:
          if len(dict_blabla[x]) == 0:
            print("zero element in this supernode",x)
          for y in dict_blabla[x]:
              P_hat[y,x] = 1

      P_hat = P_hat.to_sparse()
      P = torch.sparse.mm(P_hat,(torch.diag(torch.pow(C_diag, -1/2))))

      features =  dataset[0].x.to(device = device).to_sparse()

      cor_feat = (torch.sparse.mm((torch.t(P)), features.to_dense())).to_sparse()

      i = data.edge_index
      v = torch.ones(data.edge_index.shape[1])
      shape = torch.Size([data.x.shape[0],data.x.shape[0]])

      g_adj_tens = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(device = device)
      g_coarse_adj = torch.sparse.mm(torch.t(P_hat) , torch.sparse.mm( g_adj_tens , P_hat))

    C_diag_matrix = np.diag(np.array(C_diag.to('cpu'), dtype = np.float32))


    g_coarse_dense = g_coarse_adj.to_dense().to('cpu').numpy() + C_diag_matrix - np.identity(C_diag_matrix.shape[0], dtype = np.float32)
    edge_weight = g_coarse_dense[np.nonzero(g_coarse_dense)]
    edges_src = torch.from_numpy((np.nonzero(g_coarse_dense))[0])
    edges_dst = torch.from_numpy((np.nonzero(g_coarse_dense))[1])
    edge_index_corsen = torch.stack((edges_src, edges_dst))
    edge_features = torch.from_numpy(edge_weight)
    del C_diag_matrix
    del g_coarse_adj
    del g_coarse_dense
    del edge_weight
    del edges_dst
    del i
    del v

    Y = np.array(data.y.cpu())
    Y = one_hot(Y,dataset.num_classes).to(device)
    Y[~data.train_mask] = torch.Tensor([0 for _ in range(num_classes)]).to(device)
    labels_coarse = torch.argmax(torch.sparse.mm(torch.t(P).double() , Y.double()).double() , 1).to(device)

    data_coarsen = Data(x=cor_feat, edge_index = edge_index_corsen, y = labels_coarse)
    data_coarsen.edge_attr = edge_features

    return data_coarsen.to(device)
#--------------------------------------------------------------------------------------------------------------|
###############################################################################################################|
from torch_geometric.datasets import Planetoid
dataset = Planetoid(root='data/Cora', name='Cora')
#dataset = Planetoid(root='data/CiteSeer', name='CiteSeer')
#dataset = Planetoid(root = 'data/PubMed', name = 'PubMed')

# The Coauthor datasers do not have marks
#dataset = Coauthor(root = 'data/Physics', name = 'Physics')
# dataset = Coauthor(root = 'data/CS', name = 'CS')
#####################################################################################

data = dataset[0]
test_split_percent = 0.2
data = split(data, dataset.num_classes,test_split_percent) # cora & citeseer
#data = random_splits_citation(data, dataset.num_classes)
# For [Cora 30% binwidth = 0.0124] [Cora 50% : binwidth = 0.028][ Cora 70% binwidth = 0.067]
# Without WLK [Cora 70% binwidth = 0.0113]
# With WLK  [CiteSeer 30% : binwidth = 0.0124][CiteSeer 50% : binwidth = 0.030][CiteSeer 70% : binwidth = 0.074]
# Without WLK PubMed 50% : binwidth = 0.000039
# delta = 0.002
# epcilon_list = [0.1]#, 0.2, 0.5, 1, 2, 4, 6, 8]
no_of_hash = 1000
#binwidth = 0.039
#binwidth =  0.000039 #pubmed
#binwidth = 0.0170
binwidth = 0.028
###################################################################################
#                            GCN
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """ GCNConv layers """
        self.conv1 = GCNConv(data.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
####################
########### GAT 
class GAT(torch.nn.Module):
    def __init__(self, hidden_channels=16):
        super(GAT, self).__init__()
        torch.manual_seed(42)

        # Initialize the layers
        self.conv1 = GATConv(dataset.num_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.out = Linear(hidden_channels, dataset.num_classes)

    def forward(self, data):
        # First Message Passing Layer (Transformation)
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        # Second Message Passing Layer
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        # Output layer 
        x = F.softmax(self.out(x), dim=1)
        return x
##############################################################



for i in range(1):
    for j in range(1):
        coarsened_data = DPGC(data, dataset, no_of_hash, binwidth)
        #coarsened_data = split(coarsened_data, dataset.num_classes,test_split_percent)
        coarsened_data = random_splits_citation(coarsened_data, dataset.num_classes)
        # print("----------------------------------")
        # print(coarsened_data)
        # print("train mask:", coarsened_data.train_mask.sum())
        # print("val mask  :", coarsened_data.val_mask.sum())
        # print("test mask :", coarsened_data.test_mask.sum())
        # print("----------------------------------")

        #################################################
        #Training GCN on Coarsened data
        #################################################
        model = GCN().to(device)
        #model = GAT().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        def compute_accuracy(pred_y, y):
            return (pred_y == y).sum()
        # train the model
        model.train()
        losses = []
        accuracies = []
        for epoch in range(500):
            optimizer.zero_grad()
            out = model(coarsened_data)
            loss = F.nll_loss(out[coarsened_data.train_mask], coarsened_data.y[coarsened_data.train_mask])
            correct = compute_accuracy(out.argmax(dim=1)[coarsened_data.train_mask], coarsened_data.y[coarsened_data.train_mask])
            acc = int(correct) / int(coarsened_data.train_mask.sum())
            loss.backward()
            optimizer.step()
            # if (epoch+1) % 100 == 0:
            #     print('Epoch: {}, Loss: {:.4f}, Training Acc: {:.4f}'.format(epoch+1, loss.item(), acc))

        model.eval()
        pred = model(data).argmax(dim=1)
        correct = compute_accuracy(pred[data.test_mask], data.y[data.test_mask])
        acc = int(correct) / int(data.test_mask.sum())
        print(f' Accuracy Orignal Test Mask : {acc:.4f}')

        # model.eval()
        # pred = model(data).argmax(dim=1)
        # correct = compute_accuracy(pred[data.train_mask], data.y[data.train_mask])
        # acc = int(correct) / int(data.train_mask.sum())
        # print(f'    Accuracy Train Mask: {acc:.4f}')
    # ####################################################################################


# counts =10
# acclist = []
# for round in range(counts):
#   print('Round', round+1)
#   #coarsened_data = DPGC(data, dataset, no_of_hash, binwidth)
#   coarsened_data = DPGC(data, dataset, no_of_hash, binwidth, delta, epcilon)
#   coarsened_data = random_splits_citation(coarsened_data, dataset.num_classes)
#   model = GCN().to(device)
#   optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

#   model.train()
#   losses = []
#   accuracies = []
#   for epoch in range(500):
#       optimizer.zero_grad()
#       out = model(coarsened_data)
#       loss = F.nll_loss(out[coarsened_data.train_mask], coarsened_data.y[coarsened_data.train_mask])
#       correct = compute_accuracy(out.argmax(dim=1)[coarsened_data.train_mask], coarsened_data.y[coarsened_data.train_mask])
#       acc = int(correct) / int(coarsened_data.train_mask.sum())

#       loss.backward()
#       optimizer.step()
#     #   if (epoch+1) % 100 == 0:
#     #       print('Epoch: {}, Loss: {:.4f}, Training Acc: {:.4f}'.format(epoch+1, loss.item(), acc))

#   model.eval()
#   pred = model(data).argmax(dim=1)
#   correct = compute_accuracy(pred[data.train_mask], data.y[data.train_mask])
#   acc = int(correct) / int(data.train_mask.sum())
#   acclist.append(acc)

# print ('ACC mean:', '{0:0.4f}'.format(np.mean(acclist)))
# print ('ACC std:', '{0:0.4f}'.format(np.std(acclist)))