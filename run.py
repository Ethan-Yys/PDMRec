import argparse
import os
import sys

import pandas as pd
import torch
import numpy as np
import logging
import time
import datetime

from logging import getLogger
from tensorboardX import SummaryWriter
import config
from logger import *
from utils import *
from solver import Solver
from model import *
from dataset import Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def get_args(args, defaults):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', '-m', type=str, default='LightSANs', help='name of models')
    parser.add_argument('--phase', type=str, default=defaults.phase)
    parser.add_argument('--prefix', type=str, default=defaults.prefix)
    parser.add_argument('--seed', type=int, default=defaults.seed)
    parser.add_argument('--is_finetune', type=int, default=defaults.is_finetune)
    parser.add_argument('--is_orthogonal', type=int, default=defaults.is_orthogonal)
    parser.add_argument('--dataset', '-d', type=str, default=defaults.Dataset, help='name of datasets')
    parser.add_argument('--out_log_dir', type=str, default=defaults.out_log_dir)
    parser.add_argument('--best_ckpt_path', type=str, default='runs/')

    parser.add_argument('--batch_size', type=int, default=defaults.BATCH_SIZE)
    parser.add_argument('--learner', type=str, default=defaults.learner)
    parser.add_argument('--learning_rate', type=float, default=defaults.learning_rate)
    parser.add_argument('--weight_decay', type=float, default=defaults.weight_decay)
    parser.add_argument('--max_seq_len', type=int, default=defaults.max_seq_len)

    parser.add_argument('--user_inter_num_interval', type=str, default=defaults.user_inter_num_interval)
    parser.add_argument('--item_inter_num_interval', type=str, default=defaults.item_inter_num_interval)

    parser.add_argument('--test_step', type=int, default=defaults.test_step)
    parser.add_argument('--ks', type=int, default=defaults.ks)
    parser.add_argument('--topk', type=int, default=defaults.topk)
    parser.add_argument('--max_epoch', type=int, default=defaults.max_epoch)
    parser.add_argument('--patience', type=int, default=defaults.patience)

    # params of model ETHAN
    parser.add_argument('--n_layers', type=int, default=defaults.n_layers)
    parser.add_argument('--n_heads', type=int, default=defaults.n_heads)
    parser.add_argument('--k_interests', type=int, default=defaults.k_interests)
    parser.add_argument('--hidden_size', type=int, default=defaults.hidden_size)
    parser.add_argument('--inner_size', type=int, default=defaults.inner_size)
    parser.add_argument('--hidden_dropout_prob', type=float, default=defaults.hidden_dropout_prob)
    parser.add_argument('--attn_dropout_prob', type=float, default=defaults.attn_dropout_prob)
    parser.add_argument('--hidden_act', type=str, default=defaults.hidden_act)
    parser.add_argument('--layer_norm_eps', type=float, default=defaults.layer_norm_eps)
    parser.add_argument('--initializer_range', type=float, default=defaults.initializer_range)
    parser.add_argument('--loss_type', type=str, default=defaults.loss_type)

    parser.add_argument('--lmd', type=float, default=defaults.lmd)
    parser.add_argument('--tau', type=float, default=defaults.tau)
    parser.add_argument('--sim', type=str, default=defaults.sim)

    parameters = parser.parse_args(args)
    return parameters


def _init_logging(params):
    init_logger(params)
    logger = getLogger()
    for k in params:
        logger.info("{:15}=      {}".format(k, params[k]))
    return logger


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main(args, defaults):
    params = get_args(args, defaults)
    params = vars(params)

    setup_seed(params['seed'])

    logger = _init_logging(params)
    logger.info(params['prefix'])
    dataset = Dataset(params)
    n_item = dataset.item_num
    n_user = dataset.user_num

    model = trans_to_cuda(MyModel(params, n_user, n_item))
    print(model)

    solver = Solver(params, dataset, model)
    writer = SummaryWriter('runs_tensorboard/{}'.format('{}-{}'.format('model', get_local_time())))

    params['best_ckpt_path'] += '{}-{}.pth'.format('ETHAN', get_local_time())

    # # 测试用
    # params['best_ckpt_path'] += 'ETHAN-Mar-17-2022_15-52-24.pth'

    if 'train' in params['phase']:
        solver.train(writer)
    if 'test' in params['phase']:
        solver.test()


if __name__ == '__main__':
    main(sys.argv[1:], config.Config)
