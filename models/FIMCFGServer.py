import numpy as np
import torch.optim
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

from utils.datasets import *
from models.baseModels import *
from torch.nn.functional import normalize
import torch.nn as nn
from utils.loss import *
from utils.util import *
from utils.evaluation import *
import matplotlib.pyplot as plt



class FIMCFGServer(nn.Module):
    def __init__(self, config):
        super(FIMCFGServer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._config = config
        self._view_num = config['view_num']
        self.init_centroid = None
        self._n_clusters = config['n_clustering']

    def pre_aggregate(self, h_v_list, mask):
        sim_mat_v_list=[]
        for idx in range(len(h_v_list)):
            sim_mat_v=get_similarity_matrix(h_v_list[idx], method=self._config['method'], norm=False)
            sim_mat_v *= mask[:, idx].reshape(-1, 1)
            sim_mat_v_list.append(sim_mat_v)
        sim_mat_glo = sum(sim_mat_v_list) / len(sim_mat_v_list)
        adj_glo = get_masked_adjacency_matrix(sim_mat_glo, k=self._config['topk'])

        return adj_glo


    def forward(self, h_list,  w_view_list, w_sample_list, logger):
        sim_mat_hat_list=[]
        for h_v in h_list:
            sim_mat_hat = get_similarity_matrix(h_v, method=self._config['method'])
            sim_mat_hat_list.append(sim_mat_hat)
        with torch.no_grad():
            instance_weight_list = []
            for w_sample, sim_mat_hat in zip(w_sample_list, sim_mat_hat_list):
                w_sample = w_sample.reshape(len(w_sample), 1)
                w_sample = np.tile(w_sample, sim_mat_hat.shape[0])
                w_sample = torch.tensor(w_sample)
                instance_weight_list.append(w_sample)
            stacked_instance_weight = torch.stack(instance_weight_list)
            stacked_instance_weight = F.softmax(stacked_instance_weight, dim=0)
            stacked_adj_hat = torch.stack(sim_mat_hat_list)
            stacked_instance_weight = stacked_instance_weight.to(self.device)
            stacked_adj_hat = stacked_adj_hat.to(self.device)
            stacked_adj_glo_hat = stacked_instance_weight * stacked_adj_hat
            adj_glo = torch.sum(stacked_adj_glo_hat, dim=0)/len(sim_mat_hat_list)

            view_weight = torch.tensor(w_view_list)
            view_weight = 1.0 + torch.log(1.0 + view_weight / torch.abs(view_weight).sum())
            logger.info(f'view_weight:{view_weight}')
            for i in range(len(h_list)):
                h_list[i] = h_list[i] * view_weight[i]
            h_glo = torch.cat(h_list, dim=1)

            self.cluster = ClusterLayer(self._n_clusters)
            h_glo = h_glo.to(self.device)

            self.init_centroid, _, p_glo, y_pred = self.cluster(h_glo,self.init_centroid)

            begin = 0
            end = 0
            u_glo_list = []
            for i in range(len(h_list)):
                end += h_list[i].shape[1]
                u_glo_list.append(self.init_centroid[:, begin:end] / view_weight[i])
                begin = end

            adj_glo=get_masked_adjacency_matrix(adj_glo,k=self._config['topk'])
            return adj_glo, u_glo_list, p_glo, y_pred, h_glo


