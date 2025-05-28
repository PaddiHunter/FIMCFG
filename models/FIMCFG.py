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
from models.FIMCFGServer import *
from models.FIMCFGClient import *

class FIMCFG(nn.Module):
    def __init__(self, config):
        super(FIMCFG, self).__init__()
        self._config = config
        self._n_clusters = config['n_clustering']
        self._view_num = config['view_num']
        self.client_list = []
        self.optimizer_list = []
        self.u_glo_list = []
        self.adj_glo = None
        self.p_glo = None
        self.y_pred = None
        self.h_glo=None

        self.h_list = []
        self.w_view_list = []
        self.w_sample_list = []
        for i in range(self._view_num):
            exec(f'self.client{i} = FIMCFGClient(self._config, {i})')
            exec(f'self.client_list.append(self.client{i})')
            exec(
                f'self.optimizer{i} = torch.optim.Adam(self.client{i}.parameters(), lr=self._config[\'training\'][\'lr\'])')
            exec(f'self.optimizer_list.append(self.optimizer{i})')
            self.u_glo_list.append(None)
        self.server = FIMCFGServer(self._config)

    def run_train(self, X_train_list, Y_list, mask, logger, accumulated_metrics, device):
        h_v_list=[]
        # 预训练
        for idx in range(len(X_train_list)):
            h_v = self.client_list[idx].pretrain(X=X_train_list[idx].clone(),
                                                       mask_vector=mask[:, idx].bool(),
                                                       optimizer=self.optimizer_list[idx],
                                                       device=device)
            h_v_list.append(h_v)
        # 预聚合
        with torch.no_grad():
            self.adj_glo = self.server.pre_aggregate(h_v_list, mask)

        for i in range(self._config['training']['communication']):
            self.h_list = []
            self.sim_mat_hat_list = []
            self.w_view_list = []
            self.w_sample_list = []
            # train client
            for idx in range(self._view_num):
                h_v,  w_view, w_sample, loss_v,pred_y= self.client_list[idx].client_train(X=X_train_list[idx].clone(),
                                                                                             adj_glo=self.adj_glo,
                                                                                             u_glo=self.u_glo_list[idx],
                                                                                             p_glo=self.p_glo,
                                                                                             mask_vector=mask[:,
                                                                                                         idx].bool(),
                                                                                             optimizer=
                                                                                             self.optimizer_list[idx],
                                                                                             logger=logger,
                                                                                             accumulated_metrics=accumulated_metrics,
                                                                                             device=device)


                self.h_list.append(h_v)
                self.w_view_list.append(w_view)
                self.w_sample_list.append(w_sample)
                logger.info(f'y_pred of client {idx}' + str(
                    evaluation(y_pred=pred_y, y_true=Y_list[0], accumulated_metrics=accumulated_metrics)))

            self.adj_glo, self.u_glo_list, self.p_glo, self.y_pred, self.h_glo = self.server(h_list=self.h_list,
                                                                                       w_view_list=self.w_view_list,
                                                                                       w_sample_list=self.w_sample_list,logger=logger)

            for idx,sim_mat_hat in enumerate(self.sim_mat_hat_list):
                logger.info(f'client {idx} adj_hat:'+
                      str(get_KGC(get_masked_similarity_matrix(sim_mat_hat, k=self._config['topk']), Y_list=Y_list[0],
                              k=self._config['topk'])))



            scores = evaluation(y_pred=self.y_pred, y_true=Y_list[0], accumulated_metrics=accumulated_metrics)
            logger.info(f'communication {i} scores: {scores}')



        scores = evaluation(y_pred=self.y_pred, y_true=Y_list[0], accumulated_metrics=accumulated_metrics)
        logger.info(f'communication {i} scores: {scores}')
        print(str(scores))
        return accumulated_metrics['acc'][-1], accumulated_metrics['nmi'][-1], accumulated_metrics['ARI'][-1]


