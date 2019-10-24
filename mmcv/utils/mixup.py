import random as rd
import torch
import numpy as np
# Mixup for 5D Tensors


def spatial_mixup(x, y, alpha=1.0, use_cuda=True):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    # use original x as main data, lam is for aux data
    if alpha > 0.:
        lam = np.random.random() / 2.5
    else:
        lam = 0.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = (1 - lam) * x + lam * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# zhaoyue advice: get diversity, ruin temporal connectivity
def spatial_temporal_mixup_3d(x, y, temporal_mixup_style, alpha=1.0, use_cuda=True):
    temporal_length = x.size()[3]
    if alpha > 0:
        lam_t = np.random.randint(0, temporal_length // 2 + 1)
    else:
        lam_t = 0

    lam_s = np.random.random() / 2.5

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    replaced_ind = rd.sample(range(temporal_length), lam_t)
    mixed_x = x.clone()

    lam_aux = lam_t / temporal_length * (1 - lam_s) + (1 - lam_t) / temporal_length * lam_s

    if temporal_mixup_style == 'concat':
        aux_first = np.random.random() < 0.5
        if aux_first:
            replaced_ind = list(range(lam_t))
        else:
            replaced_ind = list(range(temporal_length - lam_t, temporal_length))

    for idx in range(temporal_length):
        if idx in replaced_ind:
            mixed_x[:, :, :, idx] = (1 - lam_s) * x[index, :, :, idx] + lam_s * x[:, :, :, idx]
        else:
            mixed_x[:, :, :, idx] = (1 - lam_s) * x[:, :, :, idx] + lam_s * x[index, :, :, idx]

    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam_aux


def spatial_temporal_mixup_2d(x, y, alpha=1.0, use_cuda=True):
    temporal_length = x.size()[1]
    if alpha > 0:
        lam_t = np.random.randint(0, temporal_length // 2 + 1)
    else:
        lam_t = 0

    lam_s = np.random.random() / 2.5

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    replaced_ind = rd.sample(range(temporal_length), lam_t)
    mixed_x = x.clone()

    lam_aux = lam_t / temporal_length * (1 - lam_s) + (1 - lam_t) / temporal_length * lam_s
    for idx in range(temporal_length):
        if idx in replaced_ind:
            mixed_x[:, idx] = (1 - lam_s) * x[index, idx] + lam_s * x[:, idx]
        else:
            mixed_x[:, idx] = (1 - lam_s) * x[:, idx] + lam_s * x[index, idx]

    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam_aux


def temporal_mixup_3d(x, y, temporal_mixup_style, alpha=1.0, use_cuda=True):
    temporal_length = x.size()[3]
    # print('temporal_length: ', temporal_length)
    if alpha > 0:
        lam = np.random.randint(0, temporal_length // 2 + 1)
    else:
        lam = 0
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)



    mixed_x = x.clone()
    replaced_ind = rd.sample(range(temporal_length), lam)

    if temporal_mixup_style == 'concat':
        aux_first = np.random.random() < 0.5
        if aux_first:
            replaced_ind = list(range(lam))
        else:
            replaced_ind = list(range(temporal_length - lam, temporal_length))
    for idx in replaced_ind:
        mixed_x[:, :, :, idx] = x[index, :, :, idx]




    y_a, y_b = y, y[index]
    lam = lam / temporal_length
    return mixed_x, y_a, y_b, lam


def temporal_mixup_2d(x, y, alpha=1.0, use_cuda=True):
    temporal_length = x.size()[1]
    if alpha > 0:
        if allow_half:
            lam = np.random.randint(0, temporal_length // 2 + 1)
        else:
            lam = np.random.randint(0, temporal_length // 2)
    else:
        lam = 0
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    replaced_ind = rd.sample(range(temporal_length), lam)
    mixed_x = x.clone()

    for idx in replaced_ind:
        mixed_x[:, idx] = x[index, idx]

    y_a, y_b = y, y[index]
    lam = lam / temporal_length
    return mixed_x, y_a, y_b, lam
