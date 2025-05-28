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


class FIMCFGClient(nn.Module):
    def __init__(self, config, view_idx):
        super(FIMCFGClient, self).__init__()
        self._config = config
        self._view_idx = view_idx
        self._n_clusters = config['n_clustering']

        self.gnnEncoder = GraphEncoder(config['Autoencoder']['gatEncoder1'][view_idx],
                                       config['Autoencoder']['gatEncoder2'][view_idx])
        self.gnnDecoder = GraphDecoder(method=self._config['method'])
        self.deepClusterLayer = DeepClusterLayer(config['n_clustering'],config['o_dim'])

        self.fusionNet = FusionNet(config['Autoencoder']['gatEncoder1'][view_idx][-1]+config['Autoencoder']['gatEncoder2'][view_idx][-1], config['o_dim'])
        self.X_hat=None
        self.adj_v_hat=None

    def predict(self, h, centroid=None):
        self.cluster = ClusterLayer(self._n_clusters, init_centers=centroid)
        centers, q, p, _ = self.cluster(h)
        return q, p, centers


    def pretrain(self,X,mask_vector,optimizer,device):
        criterion_rec = nn.MSELoss(reduction='sum')
        sim_mat_v=get_similarity_matrix(X,method=self._config['method'],norm=False)
        adj_v=get_masked_adjacency_matrix(sim_mat_v, k=self._config['topk'])
        adj_v=get_masked_similarity_matrix(adj_v,k=self._config['topk'])
        for k in range(200):
            h_v=self.gnnEncoder.pretrain_model(X,normalize_adj(adj=adj_v))
            adj_v_hat=get_similarity_matrix(h_v,method=self._config['method'],norm=False)
            adj_v_hat=get_masked_similarity_matrix(adj_v_hat,k=self._config['topk'])
            loss=criterion_rec(adj_v_hat,adj_v)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        h_v = self.gnnEncoder.pretrain_model(X, normalize_adj(adj_v))

        return h_v.detach()



    def client_train(self, X, adj_glo, u_glo, p_glo, mask_vector, optimizer, logger,
                     accumulated_metrics,
                     device):
        torch.autograd.set_detect_anomaly(True)
        LOSS = []
        lamb_re_g = self._config['training']['lamb_re']
        lamb_re_c=self._config['training']['lamb_re_f']
        lamb_kl = self._config['training']['lamb_kl']

        epochs = self._config['training']['epoch']
        batch_size = self._config['training']['batch_size']
        batch_size = batch_size if X.shape[0] > batch_size else X.shape[0]

        criterion_rec_g = ReconLoss(temperature=1.0, device=device,method=self._config['method']).to(device)
        criterion_kl = KLLoss().to(device)
        criterion_rec_c=nn.MSELoss(reduction='mean')

        if u_glo is not None:
            self.deepClusterLayer.init_parameters(u_glo)

        if self.X_hat is not None:
            with torch.no_grad():
                X[~mask_vector,:]=self.X_hat[~mask_vector,:]
        sim_mat_v = get_similarity_matrix(X, method=self._config['method'])
        adj_v = get_masked_adjacency_matrix(sim_mat_v, k=self._config['topk'])
        adj_v[~mask_vector, :] = adj_glo[~mask_vector, :]

        for k in range(epochs):
            h_v, h_v_glo, X_hat = self.gnnEncoder(X, normalize_adj(adj_v),
                                                  normalize_adj(adj_glo))

            h=self.fusionNet(torch.cat([h_v,h_v_glo],dim=1))


            centers, q_v, p_v, labels = self.deepClusterLayer(h)

            loss_rec_g = lamb_re_g * criterion_rec_g(adj_v=adj_v,
                                             adj_glo=adj_glo,
                                             h_v=h_v,
                                             h_v_glo=h_v_glo,
                                                h=h)
            loss=loss_rec_g

            X_complete=X[mask_vector,:]
            X_hat_complete=X_hat[mask_vector,:]
            X_complete,X_hat_complete=shuffle(X_complete,X_hat_complete)
            loss_rec_c=0.0
            for batch_X,batch_X_hat,batch_No in next_batch(X_complete,X_hat_complete,batch_size=batch_size):
                loss_rec_c += lamb_re_c*criterion_rec_c(batch_X, batch_X_hat)
            loss += loss_rec_c

            if p_glo is not None:
                loss_kl = lamb_kl *(criterion_kl(q_v, p_glo, reduction='mean'))
            else:
                loss_kl = criterion_kl(q_v, p_v, reduction='mean')

            loss += loss_kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            h_v, h_v_glo, X_hat = self.gnnEncoder(X, normalize_adj(get_masked_adjacency_matrix(adj_v,k=self._config['topk'])),
                                                  normalize_adj(get_masked_adjacency_matrix(adj_glo,k=self._config['topk'])))
            h = self.fusionNet(torch.cat([h_v, h_v_glo], dim=1))
            self.X_hat=X_hat
            u, _, _, pred_y = self.deepClusterLayer(h)
            q, p_v, _ = self.predict(h=h, centroid=u)
            w_view, w_sample = self.cluster.get_weight(h)
        return h.detach(), w_view, w_sample, LOSS,  pred_y
