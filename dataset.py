import numpy as np
import random, logging, os
from logging import getLogger

import pandas as pd

from collections import Counter, defaultdict
from logger import *


class Dataset:
    def __init__(self, params):
        self.params = params
        self.dataset_name = params['dataset']
        self.data_path = './data/' + self.dataset_name + '/data.inter'
        self.df_inter = pd.read_csv(self.data_path, sep='\t', dtype={'uesr_id': str,
                                                                     'item_id': str,
                                                                     'timestamp': float})

        self.logger = getLogger()

        self.max_seq_len = params['max_seq_len']
        self.user_inter_num_interval = params['user_inter_num_interval']
        self.item_inter_num_interval = params['item_inter_num_interval']

        if params['item_inter_num_interval']:
            self._filter_by_inter_num('item_id')
        if params['user_inter_num_interval']:
            self._filter_by_inter_num('user_id')

        self.user_counter = Counter(self.df_inter['user_id'].values)
        self.item_counter = Counter(self.df_inter['item_id'].values)
        self.field2id = {}
        self._remap(['user_id', 'item_id'])
        self.user_num = len(self.field2id['user_id'])
        self.item_num = len(self.field2id['item_id'])
        self.logger.info('user_num is {}'.format(self.user_num))
        self.logger.info('item_num is {}'.format(self.item_num))
        self.logger.info('inter_num is {}'.format(len(self.df_inter)))
        self.sss = 0

    def _filter_by_inter_num(self, field):
        if field == 'user_id':
            endpoint_pair = self.user_inter_num_interval[1:-1].split(',')
        elif field == 'item_id':
            endpoint_pair = self.item_inter_num_interval[1:-1].split(',')
        left_point, right_point = float(endpoint_pair[0]), float(endpoint_pair[1])

        field_inter_num = Counter(self.df_inter[field].values) if self.user_inter_num_interval else Counter()

        ids_illegal = {id_ for id_ in field_inter_num if left_point <= field_inter_num[id_] <= right_point}

        if len(ids_illegal) == 0:
            return 0
        self.df_inter = self.df_inter[self.df_inter[field].isin(ids_illegal)]
        self.df_inter.reset_index(drop=True, inplace=True)

    def _remap(self, field_list):
        for field in field_list:
            field_value_list = self.df_inter[field].values.tolist()
            new_ids_list, token_raw = pd.factorize(field_value_list)
            new_ids_list += 1
            token_raw = np.array(['[PAD]'] + list(token_raw))
            token2id = {t: i for i, t in enumerate(token_raw)}
            self.field2id[field] = token2id
            self.df_inter[field] = new_ids_list



