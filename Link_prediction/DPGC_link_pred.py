import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import torch_geometric.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np
import math
from collections import Counter

from locale import currency
from sklearn.manifold import TSNE
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
########################################################################################
#                         Utility
########################################################################################
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

def hashed_values(data, no_of_hash, feature_size, function = 'dot'):
    g_adj = to_dense_adj(data.edge_index, edge_attr= data.edge_attr)[0]
    wlf = (1/2)*torch.matmul(g_adj,data.x) + data.x
    augdata = torch.cat((data.x, wlf), dim = 1)  
    Wl = torch.FloatTensor(no_of_hash, 2*feature_size).uniform_(0,1)
    Bin_values = torch.matmul(augdata, Wl.T)
    return Bin_values

def partition(list_bin_width,Bin_values,no_of_hash):
    print(Bin_values.shape)
    summary_dict = {}
    for bin_width in list_bin_width:

        bias = torch.tensor([random.uniform(-bin_width, bin_width) for i in range(no_of_hash)])#.to(device)
        #bias = torch.tensor([np.random.normal(loc=0.0, scale=1.0) for i in range(no_of_hash)])#.to(device)

        temp = torch.floor((1/bin_width)*(Bin_values + bias))#.to(device)
        cluster, _ = torch.mode(temp, dim = 1)
        dict_hash_indices = {}

        no_nodes = Bin_values.shape[0]

        for i in range(no_nodes):
            dict_hash_indices[i] = int(cluster[i]) #.to('cpu')

        summary_dict[bin_width] = dict_hash_indices

    return summary_dict

def get_key(val, g_coarsened):
  KEYS = []
  for key, value in g_coarsened.items():
    if val == value:
      KEYS.append(key)
  return len(KEYS),KEYS

def one_hot(x, class_count):
    return torch.eye(class_count)[x, :]    
####################################################################################################################
#                                                     DPGC                                                         #
####################################################################################################################
def DPGC(data, dataset, binwidth):
    no_of_hash = 1000
    feature_size = dataset.num_features
    num_classes = dataset.num_classes
    Bin_values = hashed_values(data, no_of_hash, feature_size, function = 'dot')
    time3 = time.time()

    summary_dict = {}
    list_bin_width = [binwidth]
    summary_dict = partition(list_bin_width,Bin_values,no_of_hash)

    for bin_width in list_bin_width:
      time4 = time.time()
      print(f'Checking Ratio for bin width: {bin_width}')
      current_bin_width_summary = summary_dict[bin_width]

      values = current_bin_width_summary.values()
      #current_bin_width_summary = get_data_for_of_sample(values)

      unique_values = set(values)
      #print(f'unique values: {len(unique_values)}')
      #print(f'total values: {len(values)}')
      rr = 1 - len(unique_values)/len(values)
      print(f'Reduction Ratio: {rr}')

      # key ->    unique_values(super node identity)
      # value ->  all nodes in this super_node
      dict_blabla ={}

      C_diag = torch.zeros(len(unique_values), device= device)
      help_count = 0

      # i thinnk this can be improved
      # does this have a time complexity if O(N*v) ? i.e for each unique value searching each node hash value
      #counter = 0
      for v in unique_values:
          C_diag[help_count],dict_blabla[help_count] = get_key(v, current_bin_width_summary)
          help_count += 1


      print(f"Total Unique Values :{len(unique_values)}")
      # P_hat is bool 2D array which represent nodes contained in supernodes
      P_hat = torch.zeros((data.num_nodes, len(unique_values)), device= device)

      for x in dict_blabla:
          if len(dict_blabla[x]) == 0:
            print("zero element in this supernode",x)
          for y in dict_blabla[x]:
              P_hat[y,x] = 1

      #print(f"safe checkpoint for memory crash P_hat size is: {P_hat.shape}")
      P_hat = P_hat.to_sparse()

      #dividing by number of elements in each supernode to get average value
      P = torch.sparse.mm(P_hat,(torch.diag(torch.pow(C_diag, -1/2))))

      features =  data.x.to(device = device).to_sparse()
      cor_feat = (torch.sparse.mm((torch.t(P)), features.to_dense())).to_sparse()
      print(f"Orignal features size for all of nodes: {np.shape(features)}")
      print(f"Coarsen features size for all of nodes: {np.shape(cor_feat)}")


      i = data.edge_index
      v = torch.ones(data.edge_index.shape[1])
      shape = torch.Size([data.x.shape[0],data.x.shape[0]])
      #-------------------

      g_adj_tens = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(device = device)
      print(f"Graph Adjacency Shape: ",g_adj_tens.shape)
      #print(torch.t(P_hat).device)
      #print(g_adj_tens.device)
      #print(P_hat.device)
      g_coarse_adj = torch.sparse.mm(torch.t(P_hat) , torch.sparse.mm( g_adj_tens , P_hat))

    C_diag_matrix = np.diag(np.array(C_diag.to('cpu'), dtype = np.float32))
    #print(np.count_nonzero(g_coarse_adj.to_dense().to('cpu').numpy())/2)


    # for GCN training
    g_coarse_dense = g_coarse_adj.to_dense().to('cpu').numpy() + C_diag_matrix - np.identity(C_diag_matrix.shape[0], dtype = np.float32)

    print("Corsened Graph Adjacency Shape ",np.shape(g_coarse_dense))
    edge_weight = g_coarse_dense[np.nonzero(g_coarse_dense)]
    edges_src = torch.from_numpy((np.nonzero(g_coarse_dense))[0])
    edges_dst = torch.from_numpy((np.nonzero(g_coarse_dense))[1])
    edge_index_corsen = torch.stack((edges_src, edges_dst))
    #print(edge_index_corsen)
    #print("check       --",edge_index_corsen.shape)
    edge_features = torch.from_numpy(edge_weight)
    #num_classes = dataset.num_classes

    #---------------------
    #del P_hat
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
#-----------------------------------------------End of DPGC-----------------------------------------------------------------------
#----------------------------------------------Link Prediction Utility---------------------------------------------------------------------------
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling
import torch
from torch_geometric.nn import GCNConv
class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)  # product of a pair of nodes on each edge

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


def train_link_predictor(model, train_data, val_data, optimizer, criterion, n_epochs):
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)

        # sampling training negatives for every training epoch
        neg_edge_index = negative_sampling(
            edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

        edge_label_index = torch.cat(
            [train_data.edge_label_index, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([
            train_data.edge_label,
            train_data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)

        out = model.decode(z, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()

        val_auc = eval_link_predictor(model, val_data)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val AUC: {val_auc:.3f}")

    return model

@torch.no_grad()
def eval_link_predictor(model, data):

    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()

    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())

def show_graph_stats(graph):
    print(f"Number of nodes: {graph.x.shape[0]}")
    print(f"Number of node features: {graph.x.shape[1]}")
    print(f"Number of edges: {graph.edge_index.shape[1]}")

#--------------------------------------- Link prediction function end---------------------------------------------------------
##############################################################################################################################
from torch_geometric.datasets import Planetoid
dataset = Planetoid(root='data/Cora', name='Cora')
#dataset = Planetoid(root='data/Citeseer', name='Citeseer')
#dataset = Planetoid(root='data/Pubmed', name='Pubmed')
data = dataset[0]
#print(data)

# The Coauthor datasers do not have marks
# dataset = Coauthor(root = 'data/Physics', name = 'Physics')
# dataset = Coauthor(root = 'data/CS', name = 'CS')
# data = dataset[0]
# test_split_percent = 0.2
# data = split(data, dataset.num_classes,test_split_percent) 


# With WLK [Cora 30% binwidth = 0.0124] [Cora 50% : binwidth = 0.028][ Cora 70% binwidth = 0.067]
# Without WLK [Cora 70% binwidth = 0.0113]
# Without WLK [PubMed 95% binwidth = 0.00069][PubMed 97% binwidth = 0.001213][PubMed 98% binwidth = 0.002013] [PubMed 99% binwidth = 0.003313]
# Without WLK [PubMed 95% binwidth = 0.00069]
#binwidth = 0.0113

###########################################
# g_adj = to_dense_adj(data.edge_index, edge_attr= data.edge_attr)[0]
# wlf = (1/2)*torch.matmul(g_adj,data.x) + data.x
# data.x = torch.cat((data.x, wlf), dim = 1)
#print(data.x[0])
# print(data)
####################################
binwidth = 0.020213
coarsened_data = DPGC(data, dataset, binwidth)
# #prepraing data for training
# Random split
coarsened_data = random_splits_citation(coarsened_data, dataset.num_classes)
print(coarsened_data)
# 80-20 split 
test_split_percent = 0.2
coarsened_data = split(coarsened_data, dataset.num_classes, test_split_percent) 

# ###########################################################################################################################
# # test_split_percent = 0.2
# # #binwidth = 0.020213
# # binwidth = 0.0113
# # data = split(data, dataset.num_classes,test_split_percent) 

# # coarsened_data = DPGC(data, dataset, binwidth)
# # #prepraing data for training
# # #coarsened_data = random_splits_citation(coarsened_data, dataset.num_classes)
# # coarsened_data = split(coarsened_data, dataset.num_classes, test_split_percent)
# # print(coarsened_data)
# ############################################################
# #graph = dataset[0]
# # show_graph_stats(graph)
import torch_geometric.transforms as T
split = T.RandomLinkSplit(num_val=0.05, num_test=0.1,is_undirected=True, add_negative_train_samples=False, neg_sampling_ratio=1.0,)
#graph = dataset[0]
graph = coarsened_data
train_data, val_data, test_data = split(graph)
model = Net(dataset.num_features, 128, 64)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()
n_epochs=100
model = train_link_predictor(model, train_data, val_data, optimizer, criterion, n_epochs)
test_auc = eval_link_predictor(model, test_data)
print(f"Test: {test_auc:.3f}")

# Testing on Orignal Dataset
graph = dataset[0]
#print(graph)
train_data, val_data, test_data = split(graph)
test_auc = eval_link_predictor(model, test_data)
print(f"Test Accuaracy : {test_auc:.3f}")
# #-------------------------------------------------------------------------------------------------------------