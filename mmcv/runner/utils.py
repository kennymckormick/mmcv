import functools
import sys
import time
from getpass import getuser
from socket import gethostname

import mmcv
import torch
import torch.distributed as dist
from ..utils.mixup import spatial_mixup, temporal_mixup_3d, spatial_temporal_mixup_3d, temporal_mixup_2d, temporal_mixup_3d

def get_host_info():
    return '{}@{}'.format(getuser(), gethostname())


def get_dist_info():
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        initialized = dist.is_initialized()
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def master_only(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def obj_from_dict(info, parent=None, default_args=None):
    """Initialize an object from dict.

    The dict must contain the key "type", which indicates the object type, it
    can be either a string or type, such as "list" or ``list``. Remaining
    fields are treated as the arguments for constructing the object.

    Args:
        info (dict): Object types and arguments.
        parent (:class:`module`): Module which may containing expected object
            classes.
        default_args (dict, optional): Default arguments for initializing the
            object.

    Returns:
        any type: Object built from the dict.
    """
    assert isinstance(info, dict) and 'type' in info
    assert isinstance(default_args, dict) or default_args is None
    args = info.copy()
    obj_type = args.pop('type')
    if mmcv.is_str(obj_type):
        if parent is not None:
            obj_type = getattr(parent, obj_type)
        else:
            obj_type = sys.modules[obj_type]
    elif not isinstance(obj_type, type):
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_type(**args)


# The dimensions are N 1 C T H W
def mixup(data_dict, use_spatial_mixup, use_temporal_mixup):
    if use_spatial_mixup or use_temporal_mixup:
        data = []
        gt = []
        keys = list(data_dict.keys())
        len_source = list(map(lambda x: len(data_dict[x]), keys))
        start_point = {}
        end_point = {}
        st = 0
        for i, k in enumerate(keys):
            for idx in range(len_source[i]):
                sub_data = data_dict[k][idx]['img_group_0'].data[0]
                sub_gt = data_dict[k][idx]['gt_label'].data[0]
                start_point['{}_{}'.format(k, idx)] = st
                st += sub_data.shape[0]
                end_point['{}_{}'.format(k, idx)] = st
                data.append(sub_data)
                gt.append(sub_gt)

        data = torch.cat(data, dim=0)
        gt = torch.cat(gt, dim=0)

        if use_spatial_mixup and use_temporal_mixup:
            mixed_data, gt_label_a, gt_label_b , lam = spatial_temporal_mixup_3d(data, gt)
        elif use_spatial_mixup:
            mixed_data, gt_label_a, gt_label_b , lam = spatial_mixup(data, gt)
        elif use_temporal_mixup:
            mixed_data, gt_label_a, gt_label_b , lam = temporal_mixup_3d(data, gt)
        else:
            pass

        for i, k in enumerate(keys):
            for idx in range(len_source[i]):
                name = '{}_{}'.format(k, idx)
                st_point = start_point[name]
                ed_point = end_point[name]
                data_dict[k][idx]['img_group_0'].data[0] = mixed_data[st_point: ed_point]
                data_dict[k][idx]['gt_label'].data[0] = {'gt_label_a': gt_label_a[st_point: ed_point],
                                            'gt_label_b': gt_label_b[st_point: ed_point]}
                data_dict[k][idx]['lam'] = lam
        return data_dict
    else:
        return data_dict
