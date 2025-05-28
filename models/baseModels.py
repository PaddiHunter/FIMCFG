import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from utils.graph_adjacency import *
from utils.util import *


class GNNLayer(Module):
    def __init__(self, in_features_dim, out_features_dim, activation='relu', use_bias=True):
        super(GNNLayer, self).__init__()
        self._in_features_dim = in_features_dim
        self._out_features_dim = out_features_dim
        self._use_bias = use_bias
        # state trainable parameter
        self.weight = Parameter(torch.FloatTensor(self._in_features_dim, self._out_features_dim))
        if (self._use_bias):
            self.bias = Parameter(torch.FloatTensor(self._out_features_dim))
        self.init_parameters()

        self._bn1d = nn.BatchNorm1d(self._out_features_dim)
        if activation == 'sigmoid':
            self._activation = nn.Sigmoid()
        elif activation == 'leakyrelu':
            self._activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self._activation = nn.Tanh()
        elif activation == 'relu':
            self._activation = nn.ReLU()
        else:
            raise ValueError('Unknown activation type is %s', self._activation)

    def forward(self, features, adj, active=True, batchnorm=True):
        support = torch.mm(features, self.weight)
        output = torch.mm(adj, support)
        if self._use_bias:
            output = output + self.bias
        if batchnorm:
            output = self._bn1d(output)
        if active:
            output = self._activation(output)
        return output

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        if self._use_bias:
            torch.nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)



class DecoderLayer(Module):
    def __init__(self, in_features_dim, out_features_dim, activation='relu', use_bias=True):
        super(DecoderLayer, self).__init__()
        self._in_features_dim = in_features_dim
        self._out_features_dim = out_features_dim
        self._activation = activation
        self._use_bias = use_bias
        self._weight = Parameter(torch.FloatTensor(in_features_dim, out_features_dim))
        if use_bias:
            self._bias=Parameter(torch.FloatTensor(out_features_dim))

        self.init_parameters()

        self._bn1d = nn.BatchNorm1d(out_features_dim)

        if activation == 'sigmoid':
            self._activation = nn.Sigmoid()
        elif activation=='relu':
            self._activation=nn.ReLU()
        elif activation == 'leakyrelu':
            self._activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self._activation = nn.Tanh()
        else:
            raise ValueError('Unknown activation type is %s', self._activation)

    def forward(self, features, active=True, batchnorm=True):
        output = torch.mm(features, self._weight)
        if self._use_bias:
            output = output + self._bias
        if batchnorm:
            output = self._bn1d(output)
        if active:
            output = self._activation(output)
        return output


    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self._weight)
        if self._use_bias:
            torch.nn.init.zeros_(self._bias)
        else:
            self.register_parameter('bias', None)

class FusionNet(nn.Module):
    def __init__(self, in_dim, out_dim, activation='relu', batchnorm=True, fusion_wide=1024):
        super(FusionNet, self).__init__()
        self._activation = activation
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._batchnorm = batchnorm
        self._layers = nn.ModuleList()
        self._layers.append(nn.Linear(self._in_dim,fusion_wide,bias=True))
        if self._batchnorm:
            self._layers.append(nn.BatchNorm1d(fusion_wide))
        self._layers.append(nn.ReLU())
        self._layers.append(nn.Linear(fusion_wide,self._out_dim,bias=True))
        self._layers.append(nn.ReLU())
        self.init_net()

    def init_net(self):
        for layer in self._layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, features):
        for layer in self._layers:
            features=layer(features)
        return features




class GraphEncoder(nn.Module):
    def __init__(self, encoder_dim1, encoder_dim2, activation='relu', batchnorm=True):
        super(GraphEncoder, self).__init__()
        self._dim1 = len(encoder_dim1) - 1
        self._dim2 = len(encoder_dim2) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        encoder_layers1 = []
        encoder_layers2 = []
        for i in range(self._dim1):
            encoder_layers1.append(GNNLayer(encoder_dim1[i], encoder_dim1[i + 1], activation=self._activation))
        self._encoder1 = nn.Sequential(*encoder_layers1)
        for i in range(self._dim2):
            encoder_layers2.append(GNNLayer(encoder_dim2[i], encoder_dim2[i + 1], activation=self._activation))
        self._encoder2 = nn.Sequential(*encoder_layers2)
        decoder_layers = []
        decoder_dim=[]
        for dim1,dim2 in zip(list(reversed(encoder_dim1)),list(reversed(encoder_dim2))):
            decoder_dim.append(dim1+dim2)
        decoder_dim[-1]=int(decoder_dim[-1]/2)
        print(decoder_dim)
        for i in range(len(decoder_dim) - 1):
            decoder_layers.append(DecoderLayer(decoder_dim[i],decoder_dim[i+1]))

        self._decoder = nn.Sequential(*decoder_layers)

    def forward(self, x, adj_v, adj_glo, skip_connect_v=False, skip_connect_glo=False):
        adj_v = adj_v.clone()
        adj_glo = adj_glo.clone()
        h_v = self._encoder1[0](x, adj_v)
        for layer in self._encoder1[1:-1]:
            if skip_connect_v:
                h_v = layer(h_v, adj_v) + h_v
            else:
                h_v = layer(h_v, adj_v) + h_v
        h_v = self._encoder1[-1](h_v, adj_v, True, False)

        h_v_glo = self._encoder2[0](x, adj_glo)
        for layer in self._encoder2[1:-1]:
            if skip_connect_glo:
                h_v_glo = layer(h_v_glo, adj_glo) + h_v_glo
            else:
                h_v_glo = layer(h_v_glo, adj_glo) + h_v_glo

        h_v_glo = self._encoder2[-1](h_v_glo, adj_glo, True, False)
        h = torch.cat([h_v, h_v_glo], dim=1)

        h_hat=self._decoder[0](h)
        for layer in self._decoder[1:-1]:
            h_hat=layer(h_hat)
        h_hat=self._decoder[-1](h_hat,True,False)
        return h_v, h_v_glo, h_hat

    def pretrain_model(self,x,adj_v,skip_connect_v=False, skip_connect_glo=False):
        adj_v = adj_v.clone()
        h_v = self._encoder1[0](x, adj_v)
        for layer in self._encoder1[1:-1]:
            if skip_connect_v:
                h_v = layer(h_v, adj_v) + h_v
            else:
                h_v = layer(h_v, adj_v) + h_v
        h_v = self._encoder1[-1](h_v, adj_v, True, False)
        return h_v

    def decode_c(self, h):
        X_hat = self._decoder[0](h)
        for layer in self._decoder[1:-1]:
            X_hat = layer(X_hat)
        X_hat = self._decoder[-1](X_hat, True, False)
        return X_hat



class GraphDecoder(nn.Module):
    """reconstruct the adj from features"""

    def __init__(self, activation=None, batchnorm=False,method='heat'):
        super(GraphDecoder, self).__init__()
        self._activation = activation
        self._batchnorm = batchnorm
        self._method=method

    def forward(self, z):
        adj = get_similarity_matrix(z, method=self._method)
        if self._activation:
            adj = self._activation(adj)
        if (self._batchnorm):
            adj = F.normalize(adj, dim=1)

        return adj


class InnerProductDecoderW(nn.Module):
    """reconstruct the adj from features with a trainable parameter W"""

    def __init__(self, z_dim, activation=None, batchnorm=True):
        super(InnerProductDecoderW, self).__init__()
        self._activation = activation
        self._batchnorm = batchnorm
        self.W = Parameter(torch.Tensor(z_dim, z_dim))
        torch.nn.init.xavier_uniform_(self.W)

    def forward(self, z):
        adj = z @ self.W @ z.t()
        if self._activation:
            adj = self._activation(adj)
        if self._batchnorm:
            adj = torch.softmax(adj, dim=1)
        return adj


class DeepClusterLayer(nn.Module):
    def __init__(self, n_clusters, dim, init_centroids=None):
        super(DeepClusterLayer, self).__init__()
        self._n_clusters = n_clusters
        self._dim = dim
        self._centroids = Parameter(torch.FloatTensor(self._n_clusters, self._dim))
        if init_centroids is not None:
            self.init_parameters(init_centroids)
        else:
            torch.nn.init.xavier_uniform_(self._centroids)

    def forward(self, h):
        distance = torch.cdist(h, self._centroids)
        soft_labels = 1.0 / (distance ** 2 + 1)
        soft_labels = soft_labels / torch.sum(soft_labels, dim=1, keepdim=True)
        target_labels = (soft_labels ** 2) / torch.sum(soft_labels ** 2, dim=1, keepdim=True)
        labels = torch.argmax(target_labels, dim=1)
        labels = labels.cpu().numpy()
        return self._centroids.detach(), soft_labels, target_labels, labels

    def init_parameters(self, init_centroids):
        self._centroids = nn.Parameter(init_centroids)


class ClusterLayer(nn.Module):

    def __init__(self, n_clustering, init_centers=None, max_iter=300):
        super(ClusterLayer, self).__init__()
        self._n_clustering = n_clustering
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        self._init_centers = init_centers
        self.centroids = None

    def forward(self, X, init_centroids=None):
        X_cpu = X.to('cpu')
        with torch.no_grad():
            if init_centroids is not None:
                if init_centroids.device != 'cpu':
                    init_centroids = init_centroids.to('cpu')
                self.cluster = KMeans(n_clusters=self._n_clustering, random_state=0, init=init_centroids,
                                      max_iter=1000).fit(X_cpu)
            else:
                self.cluster = KMeans(n_clusters=self._n_clustering, random_state=0, max_iter=1000).fit(X_cpu)
            self.centroids = self.cluster.cluster_centers_
            self.centroids = torch.tensor(self.centroids).float()

        if self.centroids.is_cpu:
            self.centroids = self.centroids.to(self.device)
        distances = torch.cdist(X, self.centroids)
        soft_labels = 1.0 / (distances ** 2 + 1)
        soft_labels = soft_labels / torch.sum(soft_labels, dim=1, keepdim=True)
        target_labels = (soft_labels ** 2) / (torch.sum(soft_labels ** 2, dim=1, keepdim=True))
        labels = torch.argmax(target_labels, dim=1)
        labels = labels.cpu().numpy()
        return self.centroids, soft_labels, target_labels, labels

    def get_weight(self, X):
        _, _, _, result = self.forward(X)
        X_cpu = X.cpu().numpy()
        score = silhouette_score(X_cpu, result)
        sample_score = silhouette_samples(X_cpu, result)

        return score, sample_score
