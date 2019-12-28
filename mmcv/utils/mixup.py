import random as rd
try:
    import torch
except ImportError:
    pass
import numpy as np
# Mixup for 5D Tensors


def spatial_mixup(x, y, alpha=0.2, use_cuda=True):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    # use original x as main data, lam is for aux data
    lam = np.random.beta(alpha, alpha)
    lam = 1 - lam if lam > 0.5 else lam
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = (1 - lam) * x + lam * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# zhaoyue advice: get diversity, ruin temporal connectivity
def spatial_temporal_mixup_3d(x, y, temporal_mixup_style, alpha=0.2, use_cuda=True):
    temporal_length = x.size()[3]
    lam_t = np.random.beta(alpha, alpha)
    lam_t = 1 - lam_t if lam_t > 0.5 else lam_t
    lam_t = int(lam_t * temporal_length + 0.5)

    lam_s = np.random.beta(alpha, alpha)
    lam_s = 1 - lam_s if lam_s > 0.5 else lam_s

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


def spatial_temporal_mixup_2d(x, y, alpha=0.2, use_cuda=True):
    temporal_length = x.size()[1]

    lam_t = np.random.beta(alpha, alpha)
    lam_t = 1 - lam_t if lam_t > 0.5 else lam_t
    lam_t = int(lam_t * temporal_length + 0.5)

    lam_s = np.random.beta(alpha, alpha)
    lam_s = 1 - lam_s if lam_s > 0.5 else lam_s

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


def temporal_mixup_3d(x, y, temporal_mixup_style, alpha=0.2, use_cuda=True):
    temporal_length = x.size()[3]
    # print('temporal_length: ', temporal_length)
    lam = np.random.beta(alpha, alpha)
    lam = 1 - lam if lam > 0.5 else lam
    lam = int(lam * temporal_length + 0.5)

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


def temporal_mixup_2d(x, y, alpha=0.2, use_cuda=True):
    temporal_length = x.size()[1]
    lam = np.random.beta(alpha, alpha)
    lam = 1 - lam if lam > 0.5 else lam
    lam = int(lam * temporal_length + 0.5)

    
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
