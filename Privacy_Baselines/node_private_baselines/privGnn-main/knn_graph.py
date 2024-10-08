import aggr_and_network_functions
import preprocess_datasets
import active_learning
import os
from autodp import rdp_bank, dp_acct, rdp_acct, privacy_calibrator
import numpy as np
import utils
import metrics
import statistics
import torch
import itertools
import networkx as nx
from scipy.spatial import distance
import seaborn as sns
from torch_geometric.datasets import Planetoid

import matplotlib.pyplot as plt
import sbm
import pate
import sys
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import all_config as config
config = config.config

sns.set(rc={'figure.figsize':(20,10)}) # set figure size width x height
sns.set_theme(style="ticks")
sns.set_style(style="white")

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.utils import subgraph, to_undirected
from torch_sparse import SparseTensor
from torch_geometric.nn import Node2Vec
from torch_geometric.datasets import Reddit, Amazon
from torch_geometric.data import Data
from torch_cluster import grid_cluster
from kmeans_pytorch import kmeans
import random
import time

from syft.frameworks.torch.dp import pate as pate_analysis

# No more condor so assign manually
if torch.cuda.is_available():
    torch.cuda.set_device(1)  # change this cos sometimes port 0 is full

try:
    import torch_cluster  # noqa
    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None

if config.extract_features == "feature":
    isFeature = True
else:
    isFeature = False


# For us, hog = using normal feature from the data while feature = updating the teacher features with model trained on student data (student model)!


# Steps:
# 1. create 1 model but 2 different parameters for each of the train {No just use the model reset params}
# 2. Call the train() model from the aggr_and_network_functions
# 3. Get the feature


# To run tkt
# set tkt to true and reddit to True in the config file
# Run as normal

if torch.cuda.is_available():
    home_root = "/dstore/home/xxxx/private-knn-for-graphs/code/"
else:
    home_root = "./"


result_file = open(home_root + "resultfile_privateGNN_" + config.data_name+".txt", "a")

num_answered_query_per_run = []
ac_ag_labels_per_run = []
baseline1_test_acc_per_run = []
baseline1_star_test_acc_per_run = []
baseline2_test_acc_per_run = []
final_test_acc_per_run = []

noisy_screening_comp_per_run = []
e_dp_epsilon_comp_per_run = []

rand_state_per_run = []

global_start_time = time.time()

# for which_run in range(1, config.num_runs):
# Fix the rand_state. So you don't have to recreate indexes everytime
# random_data = [2763934991,87952126,461858464,2251922041,2203565404,2569991973,569824674,2721098863,836273002,2935227127]
random_data = [2763934991]
for rand_state in random_data:

    if torch.cuda.is_available():
        config.save_model = "/dstore/home/xxxx/private-knn-for-graphs/code/save_model/graph"+config.data_name
        data_root = "/dstore/home/xxxx/private-knn-for-graphs/code/dataset"

        cudnn.benchmark = False #it was false befpre
        cudnn.deterministic = True
        torch.cuda.manual_seed_all(rand_state)
        torch.manual_seed(rand_state)
    else:
        config.save_model = "save_model/graph"+config.data_name
        data_root = "./dataset"
        torch.manual_seed(rand_state)

    np.random.seed(rand_state)
    random.seed(rand_state)

    acct = rdp_acct.anaRDPacct()
    dependent_acct = rdp_acct.anaRDPacct()
    delta = config.delta
    b = 1 / config.epsilon
    laplacian = lambda x: rdp_bank.RDP_laplace({'b': b}, x)  # 1 is the sensitivity
    
   
    
    dir_path = config.save_model
    torch.manual_seed(rand_state)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config.use_sage:
        # SAGE architecture
        model = aggr_and_network_functions.SAGE(config.nb_features, config.hidden_channels, config.nb_labels, config.num_layers,
                                                  config.dropout).to(device)  # hardcoded
        modelBaseline_1 = aggr_and_network_functions.SAGE(config.nb_features, config.hidden_channels, config.nb_labels,
                                                            config.num_layers, config.dropout).to(device)  # hardcoded
        modelBaseline_1_star = aggr_and_network_functions.SAGE(config.nb_features, config.hidden_channels, config.nb_labels,
                                                            config.num_layers, config.dropout).to(device)  # hardcoded
        modelBaseline_2 = aggr_and_network_functions.SAGE(config.nb_features, config.hidden_channels, config.nb_labels,
                                                            config.num_layers, config.dropout).to(device)  # hardcoded

    else:
        # GCN architecture
        model = aggr_and_network_functions.GCN(config.nb_features, config.hidden_channels, config.nb_labels, config.num_layers,
                                                 config.dropout).to(device)  # hardcoded
        modelBaseline_1 = aggr_and_network_functions.GCN(config.nb_features, config.hidden_channels, config.nb_labels, config.num_layers,
                                                           config.dropout).to(device)  # hardcoded
        modelBaseline_1_star = aggr_and_network_functions.GCN(config.nb_features, config.hidden_channels, config.nb_labels, config.num_layers,
                                                           config.dropout).to(device)  # hardcoded
        modelBaseline_2 = aggr_and_network_functions.GCN(config.nb_features, config.hidden_channels, config.nb_labels, config.num_layers,
                                                           config.dropout).to(device)  # hardcoded

    print("Model", model)
    model = model.to(device)
    modelBaseline_1 = modelBaseline_1.to(device)
    modelBaseline_1_star = modelBaseline_1_star.to(device)
    modelBaseline_2 = modelBaseline_2.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    optimizer_baseline_1 = torch.optim.Adam(modelBaseline_1.parameters(), lr=config.lr)
    optimizer_baseline_1_star = torch.optim.Adam(modelBaseline_1_star.parameters(), lr=config.lr)
    optimizer_baseline_2 = torch.optim.Adam(modelBaseline_2.parameters(), lr=config.lr)

    evaluator = Evaluator(name='ogbn-arxiv') #The name is fine. It's for any multiclass classificiation

    def save_embedding(model):
        torch.save(model.embedding.weight.data.cpu(), config.save_model+'/embedding.pt')


    # override pytorch info
    class Node2VecEdit(Node2Vec):
        def pos_sample(self, batch):
            batch = batch.repeat(self.walks_per_node)
            rowptr, col, _ = self.adj.csr()
            rw = random_walk(rowptr, col, batch, self.walk_length, self.p, self.q)
            if not isinstance(rw, torch.Tensor):
                rw = rw[0]

            walks = []
            num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
            for j in range(num_walks_per_rw):
                walks.append(rw[:, j:j + self.context_size])
            return torch.cat(walks, dim=0)

        def neg_sample(self, batch):
            batch = batch.repeat(self.walks_per_node * self.num_negative_samples)

            rw = torch.randint(self.adj.sparse_size(0),
                               (batch.size(0), self.walk_length))
            rw = torch.cat([batch.view(-1, 1), rw], dim=-1)

            walks = []
            num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
            for j in range(num_walks_per_rw):
                walks.append(rw[:, j:j + self.context_size])
            return torch.cat(walks, dim=0)




    def extract_feature(path=None, use_sparse=False):
        '''
        train_data = private data
        test_data = public data
        '''

        if config.is_reddit_dataset:
            # Reddit dataset
            dataset =  Reddit(data_root+"/Reddit")
            data = dataset[0]
            data = data.to(device)

            print("Reddit dataaaaaaaa", data)
            private_data_x, private_data_y, private_data_edge_index, public_data_x, public_data_y, public_data_edge_index, idx_train_public, idx_test_public = preprocess_datasets.get_inductive_spilt(data, config.nb_labels, 300, 15000, 15000, rand_state, config.data_name) #hardcoded # 500, 20500, 20500

        elif config.is_amazon_dataset:
            # for amazon computer
            dataset =  Amazon(data_root+"/Amazon", "Computers")
            data = dataset[0]
            data = data.to(device)

            print("Amazon dataaaaaaaa", data)
            private_data_x, private_data_y, private_data_edge_index, public_data_x, public_data_y, public_data_edge_index, idx_train_public, idx_test_public = preprocess_datasets.get_inductive_spilt(data, config.nb_labels, 250, 3000, 3000, rand_state, config.data_name) #hardcoded # 250, 3000, 3000

        elif config.is_cora_dataset:
            # Cora
            dataset = Planetoid(data_root + "/Cora", name="cora")
            data = dataset[0]
            data = data.to(device)

            print("Cora dataaaaaaaa", data)
            private_data_x, private_data_y, private_data_edge_index, public_data_x, public_data_y, public_data_edge_index, idx_train_public, idx_test_public = preprocess_datasets.get_inductive_spilt(
                data, config.nb_labels, 100, 1000, 1000, rand_state, config.data_name)  # hardcoded # 250, 3000, 3000
        elif config.is_arxiv_random_split:
            # Arxiv dataset
            dataset = PygNodePropPredDataset(root=data_root, name='ogbn-arxiv')
            data = dataset[0]
            data = data.to(device)
            print("data", data)
            print("data.y", data.y)
            print("len(data.y)", len(data.y))

            private_data_x, private_data_y, private_data_edge_index, public_data_x, public_data_y, public_data_edge_index, idx_train_public, idx_test_public = preprocess_datasets.get_inductive_spilt(
                data, config.nb_labels, 100, 4000, 4000, rand_state, config.data_name)  # hardcoded # 250, 3000, 3000

        else:
            # Arxiv dataset
            dataset = PygNodePropPredDataset(root=data_root, name='ogbn-arxiv')
            data = dataset[0]
            data = data.to(device)

            print("Arxiv dataaaaaaaa", data)
            private_data_x, private_data_y, private_data_edge_index, public_data_x, public_data_y, public_data_edge_index, idx_train_public, idx_test_public = preprocess_datasets.get_preprocessed_arxiv_dataset(dataset, use_sparse, device)



        # =============== For PATE graphs===========
        if config.use_pate_graph:  # Multiple graphs are created here
            if config.split_graph_randomly:
                # Randomly split the idx and create multiple graph
                private_data_idx = np.array([x for x in range(len(private_data_x))])
                print("private_data_idx", private_data_idx)
                # randomly shuffle
                p = np.random.permutation(len(private_data_x))
                private_data_idx = private_data_idx[p]
                private_data_x = private_data_x[p]

                private_data_y = private_data_y[p]
                all_split_private_idx = np.array_split(private_data_idx, config.num_teacher_graphs)
                all_split_private_data_x = np.array_split(private_data_x.cpu(), config.num_teacher_graphs)
                all_split_private_data_y = np.array_split(private_data_y.cpu(), config.num_teacher_graphs)

                print(all_split_private_idx)
                # change to long tensor
                for s in range(len(all_split_private_idx)):
                    all_split_private_idx[s] = torch.LongTensor(all_split_private_idx[s])
                print("all_split_idx", all_split_private_idx)

                # create multiple graphs from this, then push into the private edge index, private x and private y
                all_split_private_data_edge_index = []

                print("dddd", private_data_x.shape[0])

                for k in range(len(all_split_private_idx)):
                    print(k)
                    each_subgraph, _ = subgraph(all_split_private_idx[k], private_data_edge_index, relabel_nodes=True,
                                                num_nodes=private_data_x.shape[0])
                    all_split_private_data_edge_index.append(each_subgraph)

                private_data_edge_index, private_data_x, private_data_y = all_split_private_data_edge_index, all_split_private_data_x, all_split_private_data_y

            else:
                # Use SBM to create multiple graphs
                # This private data is a list of tensor. So you need slice it i.e all_gen_graph, all_corr_nodes_features, all_corr_nodes_labels
                private_data_edge_index, private_data_x, private_data_y = sbm.create_new_private_graph(
                    len(private_data_x), private_data_edge_index, private_data_y, private_data_x, rand_state,
                    create_multiple_graphs=True, num_graphs_to_create=config.num_teacher_graphs, plot_graph=False,
                    device=device)



        if config.use_sbm:
            # Normal SBM approach. 1 giant graph and
            private_data_x, private_data_edge_index = sbm.create_new_private_graph(len(private_data_x),
                                                                                   private_data_edge_index,
                                                                                   private_data_y,
                                                                                   #private_data_y_flat,
                                                                                   private_data_x, rand_state,
                                                                                   plot_graph=False, device=device)
            print("SBM used!")

        # Second iteration i.e using the feature of the public data to update the features of the private
        if config.extract_features == "feature":
            if torch.cuda.is_available():
                filename = '/dstore/home/xxxx/private-knn-for-graphs/code/save_model/graph'+config.data_name+'/knn_num_neighbor_'+str(config.nb_teachers)+'/'+str(config.nb_teachers)+'_stdnt_.checkpoint.pth.tar'
            else:
                filename = './save_model/graph'+config.data_name+'/knn_num_neighbor_'+str(config.nb_teachers)+'/'+str(config.nb_teachers)+'_stdnt_.checkpoint.pth.tar'


            # input the private_data_y to view performance of using the updated feature
            private_data_x, _ = aggr_and_network_functions.pred(model, private_data_x, private_data_y,
                                                                  private_data_edge_index[:, :0], evaluator, filename, True)

            # changed to using full edge index for public data {03.03}
            # To input edge index for publuc, pass in the entire public_data_x. Then slice {No more slicing. See below comment}. This solves the indexerror index out of range in self error
            # For feature extraction, we input all the public_data and idx_train_public is not used in slicing.
            public_data_x, _ = aggr_and_network_functions.pred(model, public_data_x, public_data_y, public_data_edge_index, evaluator,
                                                                       filename, return_feature=True)


        return private_data_x, private_data_y, private_data_edge_index, public_data_x, public_data_y, public_data_edge_index, idx_train_public, idx_test_public

    def prepare_student_data(save=False):
        """
        prepares training data for the student model
        :param save: if set to True, will dump student training labels predicted by
                     the ensemble of teachers (with Laplacian noise) as npy files.
                     It also dumps the clean votes for each class (without noise) and
                     the labels assigned by teachers
        :return: pairs of (data, labels) to be used for student training and testing

        """
        # Load the dataset

        private_data_x, private_data_y, private_data_edge_index, public_data_x, public_data_y, public_data_edge_index, public_train_idx, public_test_idx = extract_feature()

        public_data_train_x = public_data_x[public_train_idx]
        public_data_train_y = public_data_y[public_train_idx]
        public_data_test_x = public_data_x[public_test_idx]
        public_data_test_y = public_data_y[public_test_idx]



        public_test_labels = np.array(public_data_test_y.cpu())
        public_test_labels = np.array(list(itertools.chain.from_iterable(public_test_labels)))

        # Plot distribution of the public_test labels
        preprocess_datasets.plot_labels(public_test_labels, is_all_test_labels=True)


        # Real train n test labels. Here test = public train, train = private train
        train_labels = np.array(private_data_y.cpu())
        test_labels = np.array(public_data_train_y.cpu())  # this is for calculating accuracy btw the predicted by teachers n original groundtruth


        train_labels = np.array(list(itertools.chain.from_iterable(train_labels)))
        test_labels = np.array(list(itertools.chain.from_iterable(test_labels)))

        if config.use_pagerank:
            public_train_al_selection_idx = active_learning.page_rank(public_data_edge_index, public_train_idx, public_test_idx, config.nb_nodes_to_select)


        if config.use_clustering:
            public_train_al_selection_idx = active_learning.clustering(public_data_train_x, config.num_clusters, config.stdnt_share, device)


        if config.is_tkt:
            if config.use_al:
                stdnt_data = public_data_train_x[public_train_al_selection_idx]
            else:
                # Normal i.e default without doing any AL but TKT
                stdnt_data = public_data_train_x[:config.stdnt_share]
        else:
            # Normal i.e default without doing any AL or TKT
            stdnt_data = public_data_train_x[:config.stdnt_share]

        num_train = private_data_x.shape[0]

        if config.is_tkt:
            teachers_preds = np.zeros([stdnt_data.shape[0]])
        else:
            # original
            teachers_preds = np.zeros([stdnt_data.shape[0], config.nb_teachers])

        for idx in range(len(stdnt_data)):
            query_data = stdnt_data[idx]
            select_teacher = np.random.choice(private_data_x.shape[0], int(config.prob * num_train), replace=False)
            print("select_teacher", select_teacher.shape)

            # Euclidean distance
            dis = np.linalg.norm(private_data_x[select_teacher].cpu() - query_data.cpu(),
                                 axis=1)  # ==========> Real KNN distance metric

            k_index = select_teacher[np.argsort(dis)[:config.nb_teachers]] #sorting the distance and select only top config.nb_teachers
            print("Query no: ", idx, "out of: ", len(stdnt_data))



            if config.is_tkt:
                k_index_for_graph = torch.LongTensor(k_index)

                new_private_data_edge_index, _ = subgraph(k_index_for_graph, private_data_edge_index, relabel_nodes=True, num_nodes=private_data_x.shape[0])

                # Train the model on the train data and edge index and then use the data as the test?
                # It will produce a single output as result

                for i in range(0, config.student_epoch):
                    private_data_x_trained, _ = aggr_and_network_functions.train_baseline(model, optimizer, k_index_for_graph, private_data_x[k_index_for_graph], private_data_y[k_index_for_graph], new_private_data_edge_index, evaluator, True)

                    if config.use_al:
                        # if config.use_pagerank:
                        current_stdnt_idx = torch.LongTensor([public_train_idx[public_train_al_selection_idx][idx]]) # TODO slice the public_train_idx[] with some real tensor
                    else:
                        # Default i.e no AL is used
                        current_stdnt_idx = torch.LongTensor([public_train_idx[:config.stdnt_share][idx]])

                    # test and reset params after last epoch
                    if i == config.student_epoch - 1:
                        # Put in all your public data but only select one (the query). The slicing is the key here i.e using current_stdnt_idx
                        # Laplacian noise is added here by setting the last parameter of aggr_and_network_functions.test_baseline() to True
                        final_test_acc, teacher_after_train_pred = aggr_and_network_functions.test_baseline(model, current_stdnt_idx, public_data_x, public_data_y,
                                                                   public_data_edge_index, evaluator, False, True)
                        print("final Test Acc after training for ", config.student_epoch ,"epochs: ",final_test_acc)
                        model.reset_parameters()


                # use the predictions to train the student model
                teachers_preds[idx] = np.array(teacher_after_train_pred.cpu(), dtype=np.int32)
            else:
                # Original from the KNN paper. Just by using the direct label of the KNN and taking the max
                # sum over the number of teachers, which make it easy to compute their votings
                teachers_preds[idx] = np.array(train_labels[k_index], dtype=np.int32)
                # print("teachers_preds[idx]", teachers_preds[idx]) #This is the label of each of the teachers for a particular student. Shape is num_teachers


        if config.is_tkt:
            teachers_preds = np.asarray(teachers_preds, dtype=np.int32)
            print("teachers_preds tkt", teachers_preds.shape) # num_queries
        else:
            teachers_preds = np.asarray(teachers_preds, dtype=np.int32)
            print("teachers_preds", teachers_preds.shape)  # num_queries x num_teachers

        dir_path = os.path.join(config.save_model, 'knn_num_neighbor_' + str(config.nb_teachers))

        noisy_screening_comp = 0#acct.get_eps(delta)

        if config.is_tkt:
            # For TKT. This is noisy now. Noise added by setting the last parameter of aggr_and_network_functions.test_baseline() to True
            # idx in this case is nb_nodes_to_select
            if config.use_al:
                # if config.use_pagerank:
                # idx for
                new_idx = [x for x in range(0, config.nb_nodes_to_select)]
            else:
                # default without AL
                new_idx = [x for x in range(0, config.stdnt_share)] # TODO because we are using it to slice the test_labels of the seletected student labels (not currently used cos we returning all). So it's like returning all labels. If we are not returning all the query, you need to edit this to only return idx of the selected queries that the teacher answered as in the aggregation_knn step

            # Changed idx, stdnt_labels to the new format of TKT
            idx, stdnt_labels = np.array(new_idx), teachers_preds #aggr_and_network_functions.aggregation_knn(teachers_preds, config.gau_scale). No need for agrregation since TKT returns a single prediction

            print("idx", idx.shape) #total_number_of_queries cos we are answering all for now

            total_num_queries = len(teachers_preds)
            num_answered_query = len(stdnt_labels)
            print('answer {} queries over {} TKT'.format(num_answered_query, total_num_queries))

        else:
            # original
            idx, stdnt_labels, remain_idx = aggr_and_network_functions.aggregation_knn(teachers_preds, config.gau_scale)
            print("idx", idx[0].shape) #num_answered_queries {using [0] cos its a tuple of arrays}

            total_num_queries = len(teachers_preds)
            num_answered_query = len(stdnt_labels)
            print('answer {} queries over {}'.format(num_answered_query, total_num_queries))


        print("stdnt_labels",stdnt_labels)

        # plot stat of labels answered
        preprocess_datasets.plot_labels(stdnt_labels, num_answered_query=num_answered_query, total_num_queries=total_num_queries)

        # Moment accountant for Laplacian
        acct.compose_poisson_subsampled_mechanisms(laplacian, config.prob,
                                                   num_answered_query)  # 1K is the same as coeff=len(teachers_preds)i.e num_queries len(teachers_preds) is the total amount of queries

        e_dp_epsilon_comp = acct.get_eps(delta)  # ===>This is the epsilon delta dp
        print("Composition of \epsilon, delta Laplacian mechanisms gives {} epsilon".format(e_dp_epsilon_comp))
        

        if config.is_tkt:
            if config.use_al:
                # if config.use_pagerank:
                correct_ans_label_list, ac_ag_labels = metrics.accuracy(stdnt_labels,
                                                                            test_labels[public_train_al_selection_idx])
            else:
                # default without using AL
                correct_ans_label_list, ac_ag_labels = metrics.accuracy(stdnt_labels, test_labels[:config.stdnt_share]) #TODO [idx] deleted. We can only include it if you wanna select some subset. For now, we are returning all
        else:
            # original. I'm not doing any AL for this original KNN version since the accuracy is bad from the beginning anyways!
            correct_ans_label_list, ac_ag_labels = metrics.accuracy(stdnt_labels, test_labels[:config.stdnt_share][idx])

        # print("Accuracy of the aggregated labels Original: " + ac_ag_labels_original)
        print("Accuracy of the aggregated labels: " + str(ac_ag_labels))

        # Plot stat of correctly answered labels
        preprocess_datasets.plot_labels(correct_ans_label_list, num_answered_query=num_answered_query, is_correct_label=True)

        # get original test index of the remaining. This will be used along side original public_data_y and public_data_x
        stdnt_data_test_idx = public_test_idx

        if save:
            # Prepare filepath for numpy dump of labels produced by noisy aggregation
            dir_path = os.path.join(config.save_model, 'knn_num_neighbor_' + str(config.nb_teachers))
            utils.mkdir_if_missing(dir_path)
            filepath = dir_path + 'answer_' + str(len(stdnt_labels)) + '_knn_voting.npy'  # NOLINT(long-line)
            label_file = dir_path + 'answer_' + str(len(stdnt_labels)) + '.pkl'

            # Dump student noisy labels array
            with open(filepath, 'wb') as file_obj:
                np.save(file_obj, teachers_preds)

        if config.is_tkt:
            # Modified for tkt. We remove the [0] cos idx is a list
            if config.use_al:
                # if config.use_pagerank:
                confident_data = [public_data_train_x[public_train_al_selection_idx][i] for i in idx]
                stdnt_train_idx = [public_train_idx[public_train_al_selection_idx][i].item() for i in
                                   idx]  # item converts it to list rather than list of tensor
            else:
                # default without using AL
                confident_data = [public_data_train_x[:config.stdnt_share][i] for i in idx]
                stdnt_train_idx = [public_train_idx[:config.stdnt_share][i].item() for i in
                                   idx]  # item converts it to list rather than list of tensor
        else:
            # original. Here idx is a tuple of array. So we need select the 1st. Thus [0] is included
            # confident data are those which pass the noisy screening
            confident_data = [public_data_train_x[:config.stdnt_share][i] for i in idx[0]]
            stdnt_train_idx = [public_train_idx[:config.stdnt_share][i].item() for i in
                               idx[0]]  # item converts it to list rather than list of tensor

        # convert to long tensor
        stdnt_train_idx = torch.LongTensor(np.array(stdnt_train_idx))
        # Solution: We need not return the confident data but the stdnt_train_idx of the confident data. Then we will use stdnt_train_idx to slice the public data

        stdnt_test_idx = stdnt_data_test_idx

        return confident_data, private_data_x, private_data_y, private_data_edge_index, public_data_x, public_data_y, public_data_edge_index, public_train_idx, public_test_idx, stdnt_labels, stdnt_train_idx, stdnt_test_idx, num_answered_query, ac_ag_labels, noisy_screening_comp, e_dp_epsilon_comp


    def train_student(model):
        """
        This function trains a student using predictions made by an ensemble of
        teachers. The student and teacher models are trained using the same
        neural network architecture.
        :param model: student model
        """
        # Call helper function to prepare student data using teacher predictions
        dir_path = os.path.join(config.save_model, 'knn_num_neighbor_' + str(config.nb_teachers))

        # Main deal. no baselines
        if not config.compute_baselines and not config.use_pate_graph:
            _, private_data_x, private_data_y, private_data_edge_index, public_data_x, public_data_y, public_data_edge_index, public_train_idx, public_test_idx, stdnt_labels, stdnt_train_idx, stdnt_test_idx, num_answered_query, ac_ag_labels, noisy_screening_comp, e_dp_epsilon_comp = prepare_student_data(
                save=True)

            utils.mkdir_if_missing(dir_path)

            # for saving the model
            filename = os.path.join(dir_path, str(config.nb_teachers) + '_stdnt_.checkpoint.pth.tar')
            print('save_file', filename)
            print('stdnt_label used for train', stdnt_labels.shape)

            # updates teacher with features of student i.e using student to predict teachers
            # students share 1 edge index
            final_test_acc = aggr_and_network_functions.train_each_teacher(model, optimizer, config.student_epoch,
                                                                             public_data_x, public_data_y, stdnt_labels,
                                                                             public_data_edge_index,
                                                                             stdnt_train_idx, stdnt_test_idx, evaluator,
                                                                             filename, isFeature)

            baseline1_test_acc = baseline1_star_test_acc = baseline2_test_acc = 0 #"Not computed" # only our method. No baseline

        elif config.use_pate_graph:
            print("Use PATE")
            # all the private_data_x, private_data_y and private_data_edge_index are list of tensors. with the len of config.num_teacher_graphs
            private_data_x, private_data_y, private_data_edge_index, public_data_x, public_data_y, public_data_edge_index, public_train_idx, public_test_idx = extract_feature()

            public_data_train_x = public_data_x[public_train_idx]
            public_data_train_y = public_data_y[public_train_idx]
            public_data_test_x = public_data_x[public_test_idx]
            public_data_test_y = public_data_y[public_test_idx]

            stdnt_data_x = public_data_train_x[:config.stdnt_share] #e.g select 1K nodes
            stdnt_data_y_original = public_data_train_y[:config.stdnt_share] #e.g select 1K nodes

            new_idx = [x for x in range(0, config.stdnt_share)] # cos it's public, we are selecting the first say 1K nodes

            # Changed idx, stdnt_labels to the new format of TKT
            idx = np.array(new_idx)

            stdnt_train_idx = [public_train_idx[:config.stdnt_share][i].item() for i in
                               idx]  # item converts it to list rather than list of tensor

            # convert to long tensor
            stdnt_train_idx = torch.LongTensor(np.array(stdnt_train_idx))

            stdnt_test_idx = public_test_idx


            # all trained teacher models. Here, the private_data_x etc. are list of tensors for each graph
            all_teacher_models = pate.train_models(config.num_teacher_graphs, config.student_epoch, private_data_x, private_data_edge_index, private_data_y, evaluator, device)

            print("Teacher Models", len(all_teacher_models))

            # The noisy pred will be the new label for the student i.e stdnt_labels.
            non_noisy_preds, stdnt_labels = pate.aggregated_teachers(all_teacher_models, public_data_x, public_data_y, public_data_edge_index, stdnt_train_idx,
                                config.nb_labels, config.epsilon, evaluator, device)

            #
            print("non_noisy_preds", non_noisy_preds)
            print("noisy_pred", stdnt_labels)

            test_labels = np.array(
                public_data_train_y.cpu())  # this is for calculating accuracy btw the predicted by teachers n original groundtruth

            #     # Flatten list
            test_labels = np.array(list(itertools.chain.from_iterable(test_labels)))

            # aggregated accuracy
            correct_ans_label_list, ac_ag_labels = metrics.accuracy(stdnt_labels,
                                                                    test_labels[:config.stdnt_share])
            num_answered_query = len(stdnt_labels)

            # We just reuse baseline model here. It's just same as redefining a new model anyways
            stdnt_model = modelBaseline_1
            stdnt_optimizer = optimizer_baseline_1

            # train student model
            pate.train_student_pate(stdnt_model, stdnt_optimizer, stdnt_train_idx, public_data_x, stdnt_labels, public_data_edge_index,
                               evaluator, config.student_epoch, device)

            # test student model on on test data #Final
            _, final_test_acc = pate.predict_pate(stdnt_model, public_data_x, public_data_y, public_data_edge_index, stdnt_test_idx, evaluator, device, is_stdnt_train=False)

            xxx = pate.pate_graph(private_data_x, private_data_y, private_data_edge_index, public_data_x, public_data_y, public_data_edge_index, public_train_idx, public_test_idx)
            print(xxx)

            print("Privacy result")

            np.save(home_root+config.log_filename+str(config.epsilon)+"teacherpred", non_noisy_preds)
            np.save(home_root+config.log_filename+str(config.epsilon)+"stdntlabels", stdnt_labels)

            baseline1_test_acc = baseline1_star_test_acc = baseline2_test_acc = noisy_screening_comp = e_dp_epsilon_comp = 0

        else:
            # Do only baseline

            # Reextract features to avoid changes that has been made to the different values. This is only for the baselines
            private_data_x, private_data_y, private_data_edge_index, public_data_x, public_data_y, public_data_edge_index, public_train_idx, public_test_idx = extract_feature()

            # baseline training and testing
            baseline1_test_acc = aggr_and_network_functions.train_test_baselines(modelBaseline_1, optimizer_baseline_1,
                                                                                   config.student_epoch, public_train_idx,
                                                                                   public_test_idx, public_data_x,
                                                                                   public_data_y, public_data_edge_index,
                                                                                   private_data_x, private_data_y,
                                                                                   private_data_edge_index, evaluator,
                                                                                   isBaseline1=True)

            # subsample
            select_teacher = np.random.choice(private_data_x.shape[0], int(config.prob * private_data_x.shape[0]), replace=False)
            select_teacher = list(select_teacher)
            print("len(select_teacher) Baseline", len(select_teacher))
            baseline1_star_private_data_x = private_data_x[select_teacher]
            baseline1_star_private_data_y = private_data_y[select_teacher]

            # get the same number of data / nodes in the private data e.g 1T
            baseline1_star_private_data_x = baseline1_star_private_data_x[:config.nb_teachers]
            baseline1_star_private_data_y = baseline1_star_private_data_y[:config.nb_teachers]

            # select exactly 1T or 3T "teachers" (private nodes) in this case dataset to for creating the subgraph
            private_data_subset = select_teacher[:config.nb_teachers]

            # We need to create a graph from this 1K nodes as well to use the sampling version
            # changed from select_teacher to private_data_subset. This is the real and fair comparison!
            # if you wanna use all_private_data but subsampled, then change private_data_subset to select_teacher for creating the subgraph
            baseline1_star_private_data_edge_index, _ = subgraph(torch.LongTensor(private_data_subset), private_data_edge_index, relabel_nodes=True, num_nodes=private_data_x.shape[0])

            # Lastly, select the same amount of students to label. This is the train_idx
            baseline1_star_public_train_idx = public_train_idx[:config.stdnt_share]

            # baseline1_star with subsampling for the private data. Simply change the private_data with the baseline1_star_private_data
            baseline1_star_test_acc = aggr_and_network_functions.train_test_baselines(modelBaseline_1_star, optimizer_baseline_1_star,
                                                                                   config.student_epoch, baseline1_star_public_train_idx,
                                                                                   public_test_idx, public_data_x,
                                                                                   public_data_y, public_data_edge_index,
                                                                                   baseline1_star_private_data_x, baseline1_star_private_data_y,
                                                                                   baseline1_star_private_data_edge_index, evaluator,
                                                                                   isBaseline1=True, isBaseline1_star=True)


            print("Got here 2")

            baseline2_test_acc = aggr_and_network_functions.train_test_baselines(modelBaseline_2, optimizer_baseline_2,
                                                                                   config.student_epoch, public_train_idx,
                                                                                   public_test_idx, public_data_x,
                                                                                   public_data_y, public_data_edge_index,
                                                                                   private_data_x, private_data_y,
                                                                                   private_data_edge_index, evaluator,
                                                                                   isBaseline1=False)


            num_answered_query = ac_ag_labels = final_test_acc= noisy_screening_comp = e_dp_epsilon_comp = 0 #"Not computed" #only baselines

        return num_answered_query, ac_ag_labels, baseline1_test_acc, baseline1_star_test_acc, baseline2_test_acc, final_test_acc, noisy_screening_comp, e_dp_epsilon_comp


    def main(argv=None):
        # result_file = open(home_root + "resultfile_privateGNN.txt", "a")
        # writing all print to file
        old_stdout = sys.stdout
        log_file = open(home_root+config.log_filename+str(config.epsilon), "w")

        sys.stdout = log_file

        start_time = time.time()
        num_answered_query, ac_ag_labels, baseline1_test_acc, baseline1_star_test_acc, baseline2_test_acc, final_test_acc, noisy_screening_comp, e_dp_epsilon_comp = train_student(
            model)

        print("Num Answered Queries", num_answered_query)
        print("Aggregated label Acc", ac_ag_labels)
        print("Baseline1", baseline1_test_acc)
        print("Baseline1_star", baseline1_star_test_acc)
        print("Baseline2", baseline2_test_acc)
        print("Test Accuracy", final_test_acc)
        print("Comp1 Noisy screening epsilon", noisy_screening_comp)
        print("Comp2 Noisy Aggr epsilon", e_dp_epsilon_comp)
        # print("max_norm_logits", max_norm_logits)
        print("config.epsilon", config.epsilon)

        end_time = time.time()
        total_run_time = end_time - start_time

        print("Final time: ", total_run_time)

        num_answered_query_per_run.append(num_answered_query)
        ac_ag_labels_per_run.append(ac_ag_labels)
        baseline1_test_acc_per_run.append(baseline1_test_acc)
        baseline1_star_test_acc_per_run.append(baseline1_star_test_acc)
        baseline2_test_acc_per_run.append(baseline2_test_acc)
        final_test_acc_per_run.append(final_test_acc)

        noisy_screening_comp_per_run.append(noisy_screening_comp)
        e_dp_epsilon_comp_per_run.append(e_dp_epsilon_comp)
        rand_state_per_run.append(rand_state)

        sys.stdout = old_stdout

        log_file.close()

        print("rand_state", rand_state)





    if __name__ == '__main__':
        main()
