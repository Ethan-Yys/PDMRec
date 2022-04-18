import torch
from torch import nn
import torch.nn.functional as F
from loss import BPRLoss
from layers import *
from utils import *
import random
import numpy as np


class MyModel(nn.Module):
    def __init__(self, params, n_user, n_item):
        super(MyModel, self).__init__()
        self.n_items = n_item
        self.n_users = n_user
        self.params = params
        self.max_seq_length = params['max_seq_len']

        # load parameters info
        self.n_layers = params['n_layers']
        self.n_heads = params['n_heads']
        self.hidden_size = params['hidden_size']  # same as embedding_size
        self.inner_size = params['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = params['hidden_dropout_prob']
        self.attn_dropout_prob = params['attn_dropout_prob']
        self.hidden_act = params['hidden_act']
        self.layer_norm_eps = params['layer_norm_eps']

        self.batch_size = params['batch_size']
        self.lmd = params['lmd']
        self.tau = params['tau']
        self.sim = params['sim']

        self.initializer_range = params['initializer_range']
        self.loss_type = params['loss_type']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = LightTransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")
        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)
        self.nce_fct = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        temp = []
        for p in model_parameters:
            a = p.size()
            b = np.prod(a)
            temp.append(b)
        params = sum(temp)
        # params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters' + f': {str(params)}'

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        # position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        # input_emb = item_emb + position_embedding
        input_emb = item_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, position_embedding,
                                      output_all_encoded_layers=True)
        # ETHSANs+对比loss
        output_list = trm_output[-1]
        # 全量hidden state拼接
        all_hidden_out = output_list[1].view(output_list[1].shape[0], -1)
        # 最后一个hidden state
        # all_hidden_out = gather_indexes(output_list[1], item_seq_len - 1)
        # 残差网络后的全量hidden拼接
        # all_hidden_out = output_list[0].view(output_list[0].shape[0], -1)
        output = gather_indexes(output_list[0], item_seq_len - 1)

        # # SASRec+全量hidden state的对比loss 即去掉position SANs的对比实验
        # output = trm_output[-1]
        # # 全量hidden state对比
        # all_hidden_out = output.view(output.size()[0], -1)
        # output = gather_indexes(output, item_seq_len - 1)

        return output, all_hidden_out  # [B H]

    def calculate_loss(self, uids, seqs, tars, negs, lens, aug_seqs_1, aug_lens_1, aug_seqs_2, aug_lens_2):
        item_seq = seqs
        item_seq_len = lens
        seq_output, _ = self.forward(item_seq, item_seq_len)
        pos_items = tars
        if self.loss_type == 'BPR':
            neg_items = negs
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight[:self.n_items]  # unpad the augmentation mask
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)

        # NCE
        # aug_item_seq1, aug_len1, aug_item_seq2, aug_len2 = self.augment(item_seq, item_seq_len)
        _, seq_output1 = self.forward(aug_seqs_1, aug_lens_1)
        _, seq_output2 = self.forward(aug_seqs_2, aug_lens_2)

        nce_logits, nce_labels = self.info_nce(seq_output1, seq_output2, temp=self.tau,
                                               batch_size=aug_lens_1.shape[0], sim=self.sim)

        # nce_logits = torch.mm(seq_output1, seq_output2.T)
        # nce_labels = torch.tensor(list(range(nce_logits.shape[0])), dtype=torch.long, device=item_seq.device)

        with torch.no_grad():
            alignment, uniformity = self.decompose(seq_output1, seq_output2, seq_output,
                                                   batch_size=item_seq_len.shape[0])

        nce_loss = self.nce_fct(nce_logits, nce_labels)

        # return loss + self.lmd * nce_loss, alignment, uniformity

        return loss + self.lmd * nce_loss

    def decompose(self, z_i, z_j, origin_z, batch_size):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        # pairwise l2 distace
        sim = torch.cdist(z, z, p=2)

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        alignment = positive_samples.mean()

        # pairwise l2 distace
        sim = torch.cdist(origin_z, origin_z, p=2)
        mask = torch.ones((batch_size, batch_size), dtype=bool)
        mask = mask.fill_diagonal_(0)
        negative_samples = sim[mask].reshape(batch_size, -1)
        uniformity = torch.log(torch.exp(-2 * negative_samples).mean())

        return alignment, uniformity

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot'):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, uids, seqs, lens):
        item_seq = seqs
        item_seq_len = lens
        seq_output, _ = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight[:self.n_items]  # unpad the augmentation mask
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
