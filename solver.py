from logging import getLogger
import os
import time

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from dataset import *
from dataloader import *
from utils import *


class Solver(object):
    def __init__(self, params, dataset, model):
        self.params = params
        self.logger = getLogger()
        self.model = model

        # 这是双向SASRec的最优模型
        # if params['is_finetune'] == 1:
        #     self.model.load_state_dict(torch.load('./runs/ETHAN-Mar-06-2022_14-39-43.pth'))

        self.test_step = params['test_step']
        self.ks = params['ks']
        self.topk = params['topk']
        self.batch_size = params['batch_size']

        self.phase = params['phase']
        self.learner = params['learner']
        self.learning_rate = params['learning_rate']
        self.weight_decay = params['weight_decay']
        self.optimizer = self._build_optimizer(self.model.parameters())

        self.dataset = dataset
        self.train_data = TrainDataLoader(params, self.dataset)
        self.logger.info('train data loaded')
        self.valid_data = ValidDataLoader(params, self.dataset)
        self.logger.info('valid data loaded')
        self.test_data = TestDataLoader(params, self.dataset)
        self.logger.info('test data loaded')

    def _build_optimizer(self, model_params):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(model_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(model_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(model_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(model_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(model_params, lr=self.learning_rate)
            if self.weight_decay > 0:
                self.logger.warning('Sparse Adam cannot argument received argument [{weight_decay}]')
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(model_params, lr=self.learning_rate)
        return optimizer

    def train(self, writer):
        step = 0
        loss_sum = 0
        best_metrics = 0
        trials = 0
        loss_func = self.model.calculate_loss

        if not os.path.exists('runs'):
            os.mkdir('runs')

        best_model_path = self.params['best_ckpt_path']
        patience = self.params['patience']
        topk = self.topk
        max_epoch = self.params['max_epoch']
        epoch_idx = 0
        for i in range(0, max_epoch):
            for uids, seqs, tars, negs, lens, \
                aug_seqs_1, aug_lens_1, aug_seqs_2, aug_lens_2 in tqdm(self.train_data,
                                                                       total=len(self.train_data),
                                                                       ncols=100,
                                                                       desc=set_color(f"Train {epoch_idx:>2}", 'pink')):
                # torch.cuda.empty_cache()
                uids = trans_to_cuda(torch.LongTensor(uids))
                seqs = trans_to_cuda(torch.LongTensor(seqs))
                tars = trans_to_cuda(torch.LongTensor(tars))
                negs = trans_to_cuda(torch.LongTensor(negs))
                lens = trans_to_cuda(torch.LongTensor(lens))
                if len(aug_seqs_1) != 0:
                    aug_seqs_1 = trans_to_cuda(torch.LongTensor(aug_seqs_1))
                    aug_lens_1 = trans_to_cuda(torch.LongTensor(aug_lens_1))
                    aug_seqs_2 = trans_to_cuda(torch.LongTensor(aug_seqs_2))
                    aug_lens_2 = trans_to_cuda(torch.LongTensor(aug_lens_2))
                self.optimizer.zero_grad()
                # step += 1
                # if step > 1000 * self.test_step:
                #     break
                losses = loss_func(uids, seqs, tars, negs, lens, aug_seqs_1, aug_lens_1, aug_seqs_2, aug_lens_2)
                # loss = sum(losses)
                loss = losses

                loss.backward()
                self.optimizer.step()
                loss_sum += loss.item()
                writer.add_scalar("loss", loss.item(), epoch_idx)
            epoch_idx += 1
            # record
            if epoch_idx % 1 == 0:
                self.logger.info('Epoch:{:d}\tloss:{:4f}'.format(epoch_idx, loss_sum))
                loss_sum = 0

                self.model.eval()

                metrics_dict = self.eval(self.valid_data)

                self.model.train()

                cur_metric = metrics_dict["recall@{}".format(topk)]
                writer.add_scalar("recall@{}".format(topk), cur_metric, step)
                self.logger.info(metrics_dict)

                if metrics_dict['recall@{}'.format(topk)] > best_metrics:
                    torch.save(self.model.state_dict(), best_model_path)
                    best_metrics = metrics_dict['recall@{}'.format(topk)]
                    trials = 0
                else:
                    trials += 1
                    if trials > patience:
                        self.logger.info('train finished')
                        break

    def test(self):
        self.logger.info('test started')
        self.model.load_state_dict(torch.load(self.params['best_ckpt_path']))

        metrics_dict = self.eval(self.test_data)
        self.logger.info('\nrecall@10:{},mrr@10:{},ndcg@10:{}\nrecall@20:{},mrr@20:{},ndcg@20:{},'
                         '\nrecall@30:{},mrr@30:{},ndcg@30:{}\nrecall@40:{},mrr@40:{},ndcg@40:{},'
                         '\nrecall@50:{},mrr@50:{},ndcg@50:{}\nrecall@60:{},mrr@60:{},ndcg@60:{},'
                         '\nrecall@70:{},mrr@70:{},ndcg@70:{}\nrecall@80:{},mrr@80:{},ndcg@80:{},'
                         '\nrecall@90:{},mrr@90:{},ndcg@90:{}\nrecall@100:{},mrr@100:{},ndcg@100:{} '
                         .format(metrics_dict['recall@10'],
                                 metrics_dict['mrr@10'],
                                 metrics_dict['ndcg@10'],
                                 metrics_dict['recall@20'],
                                 metrics_dict['mrr@20'],
                                 metrics_dict['ndcg@20'],
                                 metrics_dict['recall@30'],
                                 metrics_dict['mrr@30'],
                                 metrics_dict['ndcg@30'],
                                 metrics_dict['recall@40'],
                                 metrics_dict['mrr@40'],
                                 metrics_dict['ndcg@40'],
                                 metrics_dict['recall@50'],
                                 metrics_dict['mrr@50'],
                                 metrics_dict['ndcg@50'],
                                 metrics_dict['recall@60'],
                                 metrics_dict['mrr@60'],
                                 metrics_dict['ndcg@60'],
                                 metrics_dict['recall@70'],
                                 metrics_dict['mrr@70'],
                                 metrics_dict['ndcg@70'],
                                 metrics_dict['recall@80'],
                                 metrics_dict['mrr@80'],
                                 metrics_dict['ndcg@80'],
                                 metrics_dict['recall@90'],
                                 metrics_dict['mrr@90'],
                                 metrics_dict['ndcg@90'],
                                 metrics_dict['recall@100'],
                                 metrics_dict['mrr@100'],
                                 metrics_dict['ndcg@100']
                                 ))
        self.logger.info('test finished')

    def eval(self, eval_data):

        self.model.eval()

        metric_dict = {}

        pos_index, pos_len = self.calculate_eval_results(eval_data)
        recall_results = self.results_to_recall(pos_index, pos_len)
        mrr_results = self.results_to_mrr(pos_index, pos_len)
        ndcg_results = self.results_to_ndcg(pos_index, pos_len)
        for k in self.ks:
            metric_dict['recall@{}'.format(k)] = round(recall_results[k - 1], 4)
        for k in self.ks:
            metric_dict['mrr@{}'.format(k)] = round(mrr_results[k - 1], 4)
        for k in self.ks:
            metric_dict['ndcg@{}'.format(k)] = round(ndcg_results[k - 1], 4)

        return metric_dict

    def results_to_recall(self, pos_index, pos_len):
        result_mat = np.cumsum(pos_index, axis=1) / pos_len.reshape(-1, 1)
        avg_result = result_mat.mean(axis=0)
        return avg_result

    def results_to_mrr(self, pos_index, pos_len):
        idxs = pos_index.argmax(axis=1)
        result = np.zeros_like(pos_index, dtype=np.float)
        for row, idx in enumerate(idxs):
            if pos_index[row, idx] > 0:
                result[row, idx:] = 1 / (idx + 1)
            else:
                result[row, idx:] = 0
        avg_result = result.mean(axis=0)
        return avg_result

    def results_to_ndcg(self, pos_index, pos_len):
        len_rank = np.full_like(pos_len, pos_index.shape[1])
        idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)

        iranks = np.zeros_like(pos_index, dtype=np.float)
        iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
        idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
        for row, idx in enumerate(idcg_len):
            idcg[row, idx:] = idcg[row, idx - 1]

        ranks = np.zeros_like(pos_index, dtype=np.float)
        ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
        dcg = 1.0 / np.log2(ranks + 1)
        dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)

        result = dcg / idcg
        avg_result = result.mean(axis=0)
        return avg_result

    def calculate_eval_results(self, eval_data):
        flag = 0

        iter_data = tqdm(eval_data, total=len(eval_data), ncols=100, desc=f"Evaluate")
        for uids, seqs, tars, lens, his_index in iter_data:
            uids = trans_to_cuda(torch.LongTensor(uids))
            seqs = trans_to_cuda(torch.LongTensor(seqs))
            tars = trans_to_cuda(torch.LongTensor(tars))
            lens = trans_to_cuda(torch.LongTensor(lens))
            in_batch_idx = torch.arange(self.batch_size)

            scores = self.model.full_sort_predict(uids, seqs, lens)
            # scores_1 = self.model.full_sort_predict(uids, seqs, lens)
            # if (scores == scores_1).all():
            #     print('==============')
            # else:
            #     print('xxxxxxxxxxxxxxxxxx')
            scores[:, 0] = -np.inf
            # for i, index_list in enumerate(his_index):
            #     for index in index_list:
            #         scores[i][index] = -np.inf
            _, topk_idx = torch.topk(scores, max(self.ks), dim=-1)
            pos_matrix = torch.zeros_like(scores, dtype=torch.int)
            pos_matrix[in_batch_idx, tars] = 1
            pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
            pos_idx = torch.gather(pos_matrix, dim=1, index=topk_idx)
            result = torch.cat((pos_idx, pos_len_list), dim=1)
            if flag == 0:
                results = result.cpu().clone().detach()
                flag = 1
            else:
                results = torch.cat((results, result.cpu().clone().detach()), dim=0)

        topk_idx_final, pos_len_list_final = torch.split(results, [max(self.ks), 1], dim=1)
        return results.to(torch.bool).numpy(), pos_len_list_final.squeeze(-1).numpy()
