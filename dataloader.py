import copy
from utils import *
import numpy as np
import random, logging, os


class AbstractDataLoader:
    def __init__(self, dataset):
        self.dataset_dict = self._read_data(dataset.df_inter)
        self.n_item = dataset.item_num
        self.n_user = dataset.user_num

    def _read_data(self, df_inter):
        inter_dict = {}
        for user_id, df_sample in df_inter.groupby('user_id'):
            inter_dict[user_id] = df_sample['item_id'].tolist()
        return inter_dict


class TrainDataLoader(AbstractDataLoader):
    def __init__(self, params, dataset):
        super(TrainDataLoader, self).__init__(dataset)
        self.params = params
        self.batch_size = params['batch_size']
        self.max_seq_len = params['max_seq_len']
        self.train_seq_dict = self.get_train_seq()
        self.train_data = self.get_train_data()
        self.train_seq_num = len(self.train_data)
        self.n_batch = self.train_seq_num // self.batch_size
        if self.train_seq_num % self.batch_size:
            self.n_batch += 1
        self.cur_batch = 0
        self.sss = 0

    def get_train_seq(self):
        train_seq_dict = {}
        for k, v in self.dataset_dict.items():
            train_seq_dict[k] = self.dataset_dict[k][:-2]
        return train_seq_dict

    def merge_user_seq(self, a, b):
        ans = []
        i = len(a) - 1
        j = len(b) - 1
        b = b[::-1]

        while i >= 0 and j >= 0:
            ans.insert(0, a[i])
            ans.insert(0, b[j])
            i -= 1
            j -= 1
        if i == -1:
            return b[:j + 1] + ans
        if j == -1:
            return a[:i + 1] + ans

    # def get_train_data(self):
    #     train_data = []
    #     for uid, seq in self.train_seq_dict.items():
    #         train_data.append([uid, seq[:-1], seq[-1]])
    #     return train_data

    # 传统的序列增强方法
    def get_train_data(self):
        max_seq_len = self.max_seq_len
        if self.params['is_finetune'] == 0:
            train_data = []
            for uid, seq in self.train_seq_dict.items():
                for idx, target in enumerate(seq):
                    if idx == 0:
                        continue
                    if idx < max_seq_len:
                        train_data.append([uid, seq[:idx], target])
                    else:
                        train_data.append([uid, seq[idx - max_seq_len:idx], target])
            return train_data

        else:
            train_data = []
            for uid, seq in self.train_seq_dict.items():
                if len(seq) <= 10:
                    train_data.append([uid, seq[:len(seq) - 1], seq[len(seq) - 1]])
                else:
                    idx = int(len(seq) * 0.85)
                    while idx < len(seq):
                        if idx < max_seq_len:
                            train_data.append([uid, seq[:idx], seq[idx]])
                        else:
                            train_data.append([uid, seq[idx - max_seq_len:idx], seq[idx]])
                        idx += 1
            return train_data

    # # ETHAN用于视频的构建序列方式
    # def get_train_data(self):
    #     train_data = []
    #     for uid, seq in self.train_seq_dict.items():
    #         for idx, target in enumerate(seq):
    #             if len(seq) <= self.max_seq_len:
    #                 # train_data.append([uid, self.merge_user_seq(seq[:idx], seq[idx+1:]), target])
    #                 train_data.append([uid, seq[idx + 1:] + seq[:idx], target])
    #             else:
    #                 if idx < self.max_seq_len // 2:
    #                     if idx == 0:
    #                         train_data.append([uid, [seq[idx+1]], target])
    #                         continue
    #                     # train_data.append([uid, self.merge_user_seq(seq[:idx], seq[idx+1:2*idx+1]), target])
    #                     train_data.append([uid, seq[idx + 1:2 * idx + 1] + seq[:idx], target])
    #                 elif len(seq) - idx < self.max_seq_len // 2:
    #                     a = idx-(len(seq)-idx-1)-1
    #                     train_data.append([uid, seq[idx + 1:] + seq[a:idx], target])
    #                     # train_data.append([uid, self.merge_user_seq(seq[a:idx], seq[idx+1:]), target])
    #                 else:
    #                     train_data.append([uid, seq[:idx], target])
    #     return train_data
    def __len__(self):
        return self.train_seq_num // self.batch_size

    def __iter__(self):
        random.shuffle(self.train_data)
        return self

    def __next__(self):
        aug_seqs_1 = []
        aug_lens_1 = []
        aug_seqs_2 = []
        aug_lens_2 = []
        idxs = np.arange(self.cur_batch * self.batch_size,
                         min(self.train_seq_num, (self.cur_batch + 1) * self.batch_size))
        # idxs = np.random.choice(np.arange(0, self.train_seq_num), size=self.batch_size, replace=False)
        if len(idxs) == 0:
            return
        uids = [self.train_data[idx][0] for idx in idxs]
        seqs = [self.train_data[idx][1] for idx in idxs]
        # for i in range(len(seqs)):
        #     for j in range(len(seqs[i])):
        #         if random.random() > 0.9:
        #             seqs[i][j] = np.random.randint(1, self.n_item)

        tars = [self.train_data[idx][2] for idx in idxs]
        lens = [min(self.max_seq_len, len(self.train_data[idx][1])) for idx in idxs]

        for seq, _len in zip(seqs, lens):
            # if _len > 1:
            #     switch = random.sample(range(3), k=2)
            # else:
            #     switch = [3, 3]
            #     aug_seq = seq
            #     aug_len = _len
            # if switch[0] == 0:
            #     aug_seq, aug_len = item_crop(seq, _len)
            # if switch[0] == 1:
            #     aug_seq, aug_len = item_mask(seq, _len, self.n_item)
            # if switch[0] == 2:
            #     aug_seq, aug_len = item_reorder(seq, _len, 0.2)
            # aug_seqs_1.append(aug_seq + ((self.max_seq_len - len(aug_seq)) * [0]))
            # aug_lens_1.append(aug_len)
            #
            # if switch[1] == 0:
            #     aug_seq, aug_len = item_crop(seq, _len)
            # if switch[1] == 1:
            #     aug_seq, aug_len = item_mask(seq, _len, self.n_item)
            # if switch[1] == 2:
            #     aug_seq, aug_len = item_reorder(seq, _len, 0.2)
            # aug_seqs_2.append(aug_seq + ((self.max_seq_len - len(aug_seq)) * [0]))
            # aug_lens_2.append(aug_len)
            if _len == 1:
                aug_seqs_1.append(seq + ((self.max_seq_len - len(seq)) * [0]))
                aug_lens_1.append(_len)
                aug_seqs_2.append(seq + ((self.max_seq_len - len(seq)) * [0]))
                aug_lens_2.append(_len)
            else:
                aug_seq, aug_len = item_reorder(seq, _len, 0.2)
                aug_seqs_1.append(aug_seq + ((self.max_seq_len - len(aug_seq)) * [0]))
                aug_lens_1.append(aug_len)
                aug_seq, aug_len = item_reorder(seq, _len, 0.2)
                aug_seqs_2.append(aug_seq + ((self.max_seq_len - len(aug_seq)) * [0]))
                aug_lens_2.append(aug_len)

        for i in range(len(seqs)):
            seqs[i] = seqs[i] + (self.max_seq_len - len(seqs[i])) * [0]

        # 增强构建序列方式
        # for i in range(len(seqs)):
        #     seqs[i] += [0] * (self.max_seq_len - len(seqs[i]))

        # 采样为最简单的随机负采样一个
        negs = list(np.random.choice(np.arange(1, self.n_item), size=len(tars), replace=False))
        for i in range(len(negs)):
            while negs[i] == tars[i]:
                negs[i] = np.random.randint(1, self.n_item)

        self.cur_batch += 1
        if self.cur_batch >= self.n_batch:
            self.cur_batch = 0
            raise StopIteration

        return uids, seqs, tars, negs, lens, aug_seqs_1, aug_lens_1, aug_seqs_2, aug_lens_2


class ValidDataLoader(AbstractDataLoader):
    def __init__(self, params, dataset):
        super(ValidDataLoader, self).__init__(dataset)
        self.batch_size = params['batch_size']
        self.max_seq_len = params['max_seq_len']
        self.valid_seq_dict = self.get_valid_seq()
        self.valid_data = self.get_valid_data()
        self.valid_seq_num = len(self.valid_data)
        self.n_batch = self.valid_seq_num // self.batch_size
        if self.valid_seq_num % self.batch_size:
            self.n_batch += 1
        self.cur_batch = 0
        self.sss = 0

    def get_valid_seq(self):
        valid_seq_dict = {}
        for k, v in self.dataset_dict.items():
            valid_seq_dict[k] = self.dataset_dict[k][:-1]
        return valid_seq_dict

    def get_valid_data(self):
        valid_data = []
        for uid, seq in self.valid_seq_dict.items():
            valid_data.append([uid, seq[:-1], seq[-1]])
        return valid_data

    def __iter__(self):
        return self

    def __len__(self):
        return self.valid_seq_num // self.batch_size

    def __next__(self):
        idxs = np.arange(self.cur_batch * self.batch_size,
                         min(self.valid_seq_num, (self.cur_batch + 1) * self.batch_size))

        if len(idxs) == 0:
            return

        uids = [self.valid_data[idx][0] for idx in idxs]
        seqs = [self.valid_data[idx][1] for idx in idxs]
        tars = [self.valid_data[idx][2] for idx in idxs]
        lens = [min(self.max_seq_len, len(self.valid_data[idx][1])) for idx in idxs]
        his_index = copy.deepcopy(seqs)
        for i in range(len(seqs)):
            seqs[i] = seqs[i][-self.max_seq_len:]
            seqs[i] = seqs[i] + (self.max_seq_len - len(seqs[i])) * [0]

        self.cur_batch += 1
        if self.cur_batch >= self.n_batch:
            self.cur_batch = 0
            raise StopIteration

        return uids, seqs, tars, lens, his_index


class TestDataLoader(AbstractDataLoader):
    def __init__(self, params, dataset):
        super(TestDataLoader, self).__init__(dataset)
        self.batch_size = params['batch_size']
        self.max_seq_len = params['max_seq_len']
        self.test_seq_dict = self.dataset_dict
        self.test_data = self.get_test_data()
        self.test_seq_num = len(self.test_data)
        self.n_batch = self.test_seq_num // self.batch_size
        if self.test_seq_num % self.batch_size:
            self.n_batch += 1
        self.cur_batch = 0
        self.sss = 0

    def get_test_data(self):
        test_data = []
        for uid, seq in self.test_seq_dict.items():
            test_data.append([uid, seq[:-1], seq[-1]])
        return test_data

    def __iter__(self):
        return self

    def __len__(self):
        return self.test_seq_num // self.batch_size

    def __next__(self):
        idxs = np.arange(self.cur_batch * self.batch_size,
                         min(self.test_seq_num, (self.cur_batch + 1) * self.batch_size))

        if len(idxs) == 0:
            return

        uids = [self.test_data[idx][0] for idx in idxs]
        seqs = [self.test_data[idx][1] for idx in idxs]
        tars = [self.test_data[idx][2] for idx in idxs]
        lens = [min(self.max_seq_len, len(self.test_data[idx][1])) for idx in idxs]
        his_index = copy.deepcopy(seqs)
        for i in range(len(seqs)):
            seqs[i] = seqs[i][-self.max_seq_len:]
            seqs[i] = seqs[i] + (self.max_seq_len - len(seqs[i])) * [0]
        self.cur_batch += 1
        if self.cur_batch >= self.n_batch:
            self.curr_batch = 0
            raise StopIteration

        return uids, seqs, tars, lens, his_index
