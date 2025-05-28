import math
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.util import *
from utils.graph_adjacency import *

EPS = sys.float_info.epsilon


class ContrastiveLoss(nn.Module):


    def __init__(self, batchsize, device, method='heat'):
        super(ContrastiveLoss, self).__init__()
        self._batchsize = batchsize
        self.device = device
        self.mask = self.get_contrastive_mask(batchsize)

        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self._method = method

    def get_contrastive_mask(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones(N, N).to(self.device)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask


    def forward(self, h_i, h_j):
        N = 2 * self._batchsize
        h_ = torch.cat((h_i, h_j), dim=0)
        sim=get_similarity_matrix(h_,method=self._method)

        sim_i_j = torch.diag(sim, self._batchsize)
        sim_j_i = torch.diag(sim, -self._batchsize)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(self.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss = loss / N

        return loss


class ReconLoss(nn.Module):
    def __init__(self, temperature, device, method='heat'):
        super(ReconLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self._method=method

    def forward(self, adj_v, adj_glo, h_v,h_v_glo,h):
        recon_loss = 0
        adj_v_hat1 = get_similarity_matrix(h_v, method=self._method)
        adj_v_hat2 = get_similarity_matrix(h_v_glo, method=self._method)
        adj_hat=get_similarity_matrix(h, method=self._method)
        MSELoss = torch.nn.MSELoss(reduction='mean')
        recon_loss+=MSELoss(adj_v,adj_v_hat1)
        recon_loss+=MSELoss(adj_glo,adj_v_hat2)
        recon_loss+=MSELoss(adj_hat,adj_glo)

        return recon_loss


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    def forward(self, q, p,reduction='sum'):
        log_p = torch.log(p)
        log_q = torch.log(q)
        kl_loss = torch.sum(p * (log_p - log_q),dim=-1)
        if reduction == 'mean':
            kl_loss = torch.mean(kl_loss)
        else:
            kl_loss = torch.sum(kl_loss)
        return kl_loss

