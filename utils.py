# --coding=utf-8---
import torch
import math
import random
import numpy as np


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def item_crop(seq, length, eta=0.6):
    num_left = math.floor(length * eta)
    if num_left == 0:
        return seq, length
    crop_begin = random.randint(0, length - num_left)
    croped_item_seq = np.zeros(len(seq))
    if crop_begin + num_left < len(seq):
        croped_item_seq[:num_left] = seq[crop_begin:crop_begin + num_left]
    else:
        croped_item_seq[:num_left] = seq[crop_begin:]
    return croped_item_seq.tolist(), num_left


def item_mask(seq, length, item_num, gamma=0.3):
    num_mask = math.floor(length * gamma)
    if num_mask == 0:
        return seq, length
    mask_index = random.sample(range(length), k=num_mask)
    masked_item_seq = np.array(seq[:])
    masked_item_seq[np.array(mask_index)] = item_num  # token 0 has been used for semantic masking
    return masked_item_seq.tolist(), length


def item_reorder(seq, length, beta=0.6):
    num_reorder = math.floor(length * beta)
    if num_reorder == 0:
        return seq, length
    reorder_begin = random.randint(0, length - num_reorder)
    reordered_item_seq = seq[:]
    shuffle_index = list(range(reorder_begin, reorder_begin + num_reorder))
    random.shuffle(shuffle_index)
    reordered_item_seq[reorder_begin:reorder_begin + num_reorder] = np.array(reordered_item_seq)[
        np.array(shuffle_index)]
    return reordered_item_seq, length


def gather_indexes(output, gather_index):
    """Gathers the vectors at the specific positions over a minibatch"""
    gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
    output_tensor = output.gather(dim=1, index=gather_index)
    return output_tensor.squeeze(1)
