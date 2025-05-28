import numpy as np
import torch
from torch.autograd import Variable
import logging
import datetime
import os
import random
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder
import math

def get_mask( data_len, missing_rate,view_num):
    """
    :param view_num: number of views
    :param data_size: size of data
    :param missing_ratio: missing ratio
    :return: mask matrix
    """
    assert view_num >= 2
    miss_sample_num = math.floor(data_len*missing_rate)
    data_ind = [i for i in range(data_len)]
    random.shuffle(data_ind)
    miss_ind = data_ind[:miss_sample_num]
    mask = np.ones([data_len, view_num])
    for j in range(miss_sample_num):
        while True:
            rand_v = np.random.rand(view_num)
            v_threshold = np.random.rand(1)
            observed_ind = (rand_v >= v_threshold)
            ind_ = ~observed_ind
            rand_v[observed_ind] = 1
            rand_v[ind_] = 0
            if np.sum(rand_v) > 0 and np.sum(rand_v) < view_num:
                break
        mask[miss_ind[j]] = rand_v

    return mask

def get_logger(config, main_dir='../logs/'):
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    plt_name = str(config['dataset']) + ' ' + str(config['missing_rate']).replace('.', ' ') + ' ' + str(
        datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H-%M-%S'))

    fh = logging.FileHandler(
        main_dir + str(config['dataset']) + ' ' + str(config['missing_rate']).replace('.', '') + ' ' + str(
            datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H-%M-%S')) + '.logs')

    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, plt_name


def get_device():
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = str('0')  # set device
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    return device


def setup_seed(seed):
    """set up random seed"""
    np.random.seed(seed)
    random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 3)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_np(x):
    """Normalize"""
    x = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))
    return x


def get_KGC(adj, Y_list, k):
    adj_cpu=adj.clone().cpu()
    idx = torch.where(adj_cpu > 0)
    count = np.where(Y_list[idx[0]] == Y_list[idx[1]])
    count = np.sum(len(count[0]))
    return 1.0 * count / (k * adj.shape[0])


def test_missing_rate(mm):
    N=mm.shape[0]
    M=mm.shape[1]
    sum_v=torch.sum(mm,dim=1)
    sum_v=torch.where(sum_v<M,1,0)
    sum_v=torch.sum(sum_v,dim=-1)/N
    print('missing rate is '+str(sum_v))
    for i in range(M):
        mv=mm[:,i]
        mv=torch.sum(mv,dim=-1)
        mv=1.0-(mv/N)
        print(f'missing rate of {i} th view is {mv}')