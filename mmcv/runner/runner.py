import logging
import os.path as osp
import time

import mmcv
import torch
import psutil, os, sys, gc
from . import hooks
from .log_buffer import LogBuffer
from torch.utils.data import DataLoader
from .hooks import (Hook, LrUpdaterHook, CheckpointHook, IterTimerHook,
                    OptimizerHook, lr_updater)
from mmcv.parallel import collate
from .checkpoint import load_checkpoint, save_checkpoint
from .priority import get_priority
from .utils import get_dist_info, get_host_info, get_time_str, obj_from_dict, mixup
import torch.distributed as dist
import random
import math
import itertools
import copy as cp
import pickle
import numpy as np
from torch.utils.data.sampler import Sampler



def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)

def memoStats():
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30
    print('memory GB:', memoryUse, flush=True)


# Actually 2 more variables added
# 1. val_acc: writed by logger_hook, at time after_val_epoch, default value 0
# 2. should_stop: writed by lr_updater, at time after_val_epoch(like scheduler.step()), default value False


class DistributedHandFreqSampler(Sampler):
    def __init__(self, dataset, handfreq, samples_per_gpu=1, num_replicas=None, rank=None):
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()
        self.rank = rank
        self.num_replicas = num_replicas
        self.dataset = dataset
        self.handfreq = handfreq

        self.samples_per_gpu = samples_per_gpu
        self.num_samples = math.ceil(len(dataset) / samples_per_gpu / num_replicas) * samples_per_gpu
        self.tot_samples = self.num_samples * num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.multinomial(torch.Tensor(self.handfreq), self.tot_samples, replacement=True, generator=g)
        indices = indices.data.numpy().tolist()
        # subsample
        indices = indices[self.rank:self.tot_samples:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def regenerate_dataloader_byfreq(old_dataloader, freq, epoch, base_distribution):
    rank, world_size = get_dist_info()
    dataset = old_dataloader.dataset
    batch_size = old_dataloader.batch_size
    num_workers = old_dataloader.num_workers
    assert base_distribution in ['uniform', 'original']
    if base_distribution == 'original':
        original_distribution = dataset.dict_lens
        freq = np.array(original_distribution) * np.array(freq)
        freq = freq.tolist()
    sampler = DistributedHandFreqSampler(dataset, freq, batch_size, world_size, rank)
    sampler.set_epoch(epoch)
    collate_fn = old_dataloader.collate_fn
    pin_memory = old_dataloader.pin_memory
    worker_init_fn = old_dataloader.worker_init_fn
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                            num_workers=num_workers, collate_fn=collate_fn,
                            pin_memory=pin_memory, worker_init_fn=worker_init_fn)
    return dataloader

class Runner(object):
    """A training helper for PyTorch.

    Args:
        model (:obj:`torch.nn.Module`): The model to be run.
        batch_processor (callable): A callable method that process a data
            batch. The interface of this method should be
            `batch_processor(model, data, train_mode) -> dict`
        optimizer (dict or :obj:`torch.optim.Optimizer`): If it is a dict,
            runner will construct an optimizer according to it.
        work_dir (str, optional): The working directory to save checkpoints
            and logs.
        log_level (int): Logging level.
        logger (:obj:`logging.Logger`): Custom logger. If `None`, use the
            default logger.
    """

    def __init__(self,
                 model,
                 batch_processor,
                 optimizer=None,
                 work_dir=None,
                 log_level=logging.INFO,
                 logger=None):
        assert callable(batch_processor)
        self.model = model
        if optimizer is not None:
            self.optimizer = self.init_optimizer(optimizer)
        else:
            self.optimizer = None
        self.batch_processor = batch_processor

        # create work_dir
        if mmcv.is_str(work_dir):
            self.work_dir = osp.abspath(work_dir)
            mmcv.mkdir_or_exist(self.work_dir)
        elif work_dir is None:
            self.work_dir = None
        else:
            raise TypeError('"work_dir" must be a str or None')

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        self._rank, self._world_size = get_dist_info()
        if logger is None:
            # print('my rank is ', self._rank, 'my logger is none')
            self.logger = self.init_logger(work_dir, log_level)
        else:
            self.logger = logger
        self.log_buffer = LogBuffer()

        self.mode = None
        self._hooks = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._max_epochs = 0
        self._max_iters = 0
        self.epoch_len = 0

        # variable added by haodong
        self.val_acc = 0.0
        self.train_acc = 0.0
        self.should_stop = False
        self.use_dynamic = False
        self.val_result = []



    @property
    def model_name(self):
        """str: Name of the model, usually the module class name."""
        return self._model_name

    # init variables for dynamic sampling
    def dynamic_init(self, num_label):
        self.num_label = num_label
        self.use_dynamic = True
        self.old_train_class_acc = torch.zeros([num_label]).type(torch.float).cuda()
        self.new_train_class_acc = torch.zeros([num_label]).type(torch.float).cuda()
        # calculate gain acc per epoch
        self.old_gain_acc = torch.zeros([num_label]).type(torch.float).cuda()
        self.new_gain_acc = torch.zeros([num_label]).type(torch.float).cuda()
        # current quota
        self.old_quota = np.array([10] * num_label)
        self.new_quota = np.array([10] * num_label)
        # marginal effect
        # self.web_marginal = torch.zeros([num_label]).type(torch.float).cuda()
        self.marginal = np.array([0.0] * num_label)
        # hit n tot
        self.hit = torch.zeros([num_label]).type(torch.float).cuda()
        self.tot = torch.zeros([num_label]).type(torch.float).cuda()

        self.all_quota = self.num_label * 10

    @property
    def rank(self):
        """int: Rank of current process. (distributed training)"""
        return self._rank

    @property
    def world_size(self):
        """int: Number of processes participating in the job.
        (distributed training)"""
        return self._world_size

    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    @property
    def inner_iter(self):
        """int: Iteration in an epoch."""
        return self._inner_iter

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_iters

    def init_optimizer(self, optimizer):
        """Init the optimizer.

        Args:
            optimizer (dict or :obj:`~torch.optim.Optimizer`): Either an
                optimizer object or a dict used for constructing the optimizer.

        Returns:
            :obj:`~torch.optim.Optimizer`: An optimizer object.

        Examples:
            >>> optimizer = dict(type='SGD', lr=0.01, momentum=0.9)
            >>> type(runner.init_optimizer(optimizer))
            <class 'torch.optim.sgd.SGD'>
        """
        if isinstance(optimizer, dict):
            # bn_no_weight_decay
            params = self.model.parameters()
            if 'bn_nowd' in optimizer and optimizer['bn_nowd']:
                optimizer.pop('bn_nowd')
                bn_params = []
                non_bn_params = []
                for name, p in self.model.named_parameters():
                    if 'bn' in name:
                        bn_params.append(p)
                    else:
                        non_bn_params.append(p)
                org_weight_decay = optimizer.pop('weight_decay')
                optim_params = [{'params': bn_params, 'weight_decay': 0},
                                {'params': non_bn_params, 'weight_decay': org_weight_decay}]
                optimizer = obj_from_dict(optimizer, torch.optim, dict(params=optim_params))
            else:
                optimizer = obj_from_dict(optimizer, torch.optim, dict(params=self.model.parameters()))
        elif not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                'optimizer must be either an Optimizer object or a dict, '
                'but got {}'.format(type(optimizer)))
        return optimizer

    def _add_file_handler(self,
                          logger,
                          filename=None,
                          mode='w',
                          level=logging.INFO):
        # TODO: move this method out of runner
        file_handler = logging.FileHandler(filename, mode)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
        return logger

    def init_logger(self, log_dir=None, level=logging.INFO):
        """Init the logger.

        Args:
            log_dir(str, optional): Log file directory. If not specified, no
                log file will be used.
            level (int or str): See the built-in python logging module.

        Returns:
            :obj:`~logging.Logger`: Python logger.
        """
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s', level=level)
        logger = logging.getLogger(__name__)
        if log_dir and self.rank == 0:
            filename = '{}.log'.format(get_time_str())
            log_file = osp.join(log_dir, filename)
            self._add_file_handler(logger, log_file, level=level)
        return logger

    def current_lr(self):
        """Get current learning rates.

        Returns:
            list: Current learning rate of all param groups.
        """
        if self.optimizer is None:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return [group['lr'] for group in self.optimizer.param_groups]

    def register_hook(self, hook, priority='NORMAL'):
        """Register a hook into the hook list.

        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def build_hook(self, args, hook_type=None):
        if isinstance(args, Hook):
            return args
        elif isinstance(args, dict):
            assert issubclass(hook_type, Hook)
            return hook_type(**args)
        else:
            raise TypeError('"args" must be either a Hook object'
                            ' or dict, not {}'.format(type(args)))

    def call_hook(self, fn_name):
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def load_checkpoint(self, filename, map_location='cpu', strict=False):
        self.logger.info('load checkpoint from %s', filename)
        return load_checkpoint(self.model, filename, map_location, strict,
                               self.logger)

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None):
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch + 1, iter=self.iter)

        if self.use_dynamic:
            meta['old_train_class_acc'] = self.old_train_class_acc
            meta['old_gain_acc'] = self.old_gain_acc
            meta['old_quota'] = self.old_quota
            meta['new_quota'] = self.new_quota
            meta['marginal'] = self.marginal

        filename = osp.join(out_dir, filename_tmpl.format(self.epoch + 1))
        linkname = osp.join(out_dir, 'latest.pth')
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filename, optimizer=optimizer, meta=meta)
        mmcv.symlink(filename, linkname)


    def train(self, data_loader, **kwargs):
        # The flag
        kwargs['validate'] = False
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(data_loader)
        self.log_buffer.clear()
        runner_info = {}
        runner_info['max_iters'] = self._max_iters


        self.call_hook('before_train_epoch')

        if 'train_ratio' not in kwargs:
            auxiliary_iter_times = [1] * (len(data_loader) - 1)
        else:
            auxiliary_iter_times = kwargs['train_ratio'][1:]

        if 'batch_flags' not in kwargs:
            batch_flags = [''] * len(data_loader)
        else:
            batch_flags = kwargs['batch_flags']

        if 'train_ratio' in kwargs:
            use_aux_per_niter = kwargs['train_ratio'][0]
        else:
            use_aux_per_niter = 1


        if len(data_loader) > 1:
            main_data_loader = data_loader[0]
            if 'dynamic' in kwargs and kwargs['dynamic']:
                base_distribution = kwargs['dynamic_base_distribution']
                main_data_loader = regenerate_dataloader_byfreq(main_data_loader, self.new_quota, self._epoch, base_distribution)
            auxiliary_data_loaders = data_loader[1: ]
            # 10/24/2019, 11:45:44 AM
            # auxiliary_data_iters = list(map(iter, auxiliary_data_loaders))
            auxiliary_data_iters = list(map(cycle, auxiliary_data_loaders))

            # support 2 style during iteration (mixup / no mixup)
            for i, main_data_batch in enumerate(main_data_loader):
                # Data Preparation Code, only when mixup
                # if i % 20 == 0:
                #     memoStats()
                runner_info['this_iter'] = self._iter

                use_cross_dataset_mixup = False
                if 'cross_dataset_mixup' in kwargs and kwargs['cross_dataset_mixup']:
                    use_cross_dataset_mixup = True

                if use_cross_dataset_mixup:
                    data_dict = {}
                    data_dict['main'] = [main_data_batch]
                    if self._iter % use_aux_per_niter == 0:
                        for idx, pair in enumerate(zip(auxiliary_data_iters, auxiliary_iter_times)):
                            it, nt = pair
                            aux_data = []
                            aux_name = 'aux{}'.format(idx)
                            for step in range(nt):
                                data_batch = next(it)
                                aux_data.append(data_batch)
                            data_dict[aux_name] = aux_data
                    # Data Mixup Code
                    # Only support 3D Mixup Here

                    spatial_mixup = 0
                    temporal_mixup = 0
                    if 'spatial_mixup' in kwargs and kwargs['spatial_mixup']:
                        spatial_mixup = 1
                    if 'temporal_mixup' in kwargs and kwargs['temporal_mixup']:
                        temporal_mixup = 1
                    mixup_beta = 0.2
                    if 'mixup_beta' in kwargs:
                        mixup_beta = kwargs['mixup_beta']
                    data_dict = mixup(data_dict, spatial_mixup, temporal_mixup, mixup_beta)


                    # Training Code
                    if 'lam' in data_dict['main'][0]:
                        lam = data_dict['main'][0]['lam']

                        keys = list(data_dict.keys())
                        for k in keys:
                            lt = len(data_dict[k])
                            for j in range(lt):
                                data_dict[k][j].pop('lam')

                        kwargs['lam'] = lam

                self._inner_iter = i
                self.call_hook('before_train_iter')
                kwargs['batch_flag'] = batch_flags[0]

                if use_cross_dataset_mixup:
                    outputs = self.batch_processor(
                        self.model, data_dict['main'][0], train_mode=True, source='', runner_info=runner_info, **kwargs)
                else:
                    outputs = self.batch_processor(
                        self.model, main_data_batch, train_mode=True, source='', runner_info=runner_info, **kwargs)


                if 'dynamic' in kwargs and kwargs['dynamic']:
                    # print('adding')
                    self.hit += outputs['hit']
                    self.tot += outputs['tot']

                if not isinstance(outputs, dict):
                    raise TypeError('batch_processor() must return a dict')
                if 'log_vars' in outputs:
                    self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
                self.outputs = outputs
                self.call_hook('after_train_iter')

                # No Aux Data for this iter, just go ahead
                if self._iter % use_aux_per_niter != 0:
                    self._iter += 1
                    continue

                for idx, nt in enumerate(auxiliary_iter_times):
                    kwargs['batch_flag'] = batch_flags[idx + 1]
                    for step in range(nt):
                        if use_cross_dataset_mixup:
                            data_batch = data_dict['aux{}'.format(idx)][step]
                        else:
                            data_batch = next(auxiliary_data_iters[idx])
                        self.call_hook('before_train_iter')
                        outputs = self.batch_processor(
                            self.model, data_batch, train_mode=True, source='/aux' + str(idx), runner_info=runner_info, **kwargs)
                        if not isinstance(outputs, dict):
                            raise TypeError('batch_processor() must return a dict')
                        if 'log_vars' in outputs:
                            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
                        self.outputs = outputs
                        self.call_hook('after_train_iter')
                self._iter += 1

        else:
            main_data_loader = data_loader[0]
            if 'dynamic' in kwargs and kwargs['dynamic']:
                base_distribution = kwargs['dynamic_base_distribution']
                main_data_loader = regenerate_dataloader_byfreq(main_data_loader, self.new_quota, self._epoch, base_distribution)
            for i, data_batch in enumerate(main_data_loader):
                if i % 100 == 0 and self.rank == 0:
                    memoStats()
                self._inner_iter = i
                self.call_hook('before_train_iter')
                outputs = self.batch_processor(
                    self.model, data_batch, train_mode=True, runner_info=runner_info, **kwargs)

                if 'dynamic' in kwargs and kwargs['dynamic']:
                    # print('adding')
                    self.hit += outputs['hit']
                    self.tot += outputs['tot']
                    # print('hit: ', self.hit)
                    # print('tot: ', self.tot)
                if not isinstance(outputs, dict):
                    raise TypeError('batch_processor() must return a dict')
                if 'log_vars' in outputs:
                    self.log_buffer.update(outputs['log_vars'],
                                           outputs['num_samples'])
                self.outputs = outputs
                # print(outputs)
                # print(sum(np.array(self.log_buffer.n_history['batch_acc'])))
                self.call_hook('after_train_iter')
                self._iter += 1

        if 'dynamic' in kwargs and kwargs['dynamic']:
            # if self._rank == 0:
            #     print('hit: ', self.hit)
            #     print('tot: ', self.tot)
            dist.all_reduce(self.hit)
            dist.all_reduce(self.tot)
            # if self._rank == 0:
            #     print('hit: ', self.hit)
            #     print('tot: ', self.tot)
            self.new_train_class_acc = self.hit / self.tot
            self.new_gain_acc = self.new_train_class_acc - self.old_train_class_acc

            # If we do resample?
            do_resample = True
            for step in [0] + kwargs['schedule']:
                if self._epoch >= step and self._epoch - step < 5:
                    do_resample = False

            if do_resample:
                assert kwargs['dynamic_policy'] in ['margin', 'naive']
                if kwargs['dynamic_policy'] == 'margin':
                    quota_diff = self.new_quota - self.old_quota
                    gain_acc_diff = self.new_gain_acc - self.old_gain_acc
                    num_label = kwargs['num_label']
                    if self._rank == 0:
                        print('quota_diff: ', quota_diff)
                        print('gain_acc_diff: ', gain_acc_diff)

                    marginal  = self.marginal
                    if self._rank == 0:
                        print('Old marginal: ', marginal)
                    # print(num_label)
                    for i in range(num_label):
                        # print(quota_diff[i])
                        if quota_diff[i] == 1:
                            # print('hit1')
                            marginal[i] = gain_acc_diff[i]
                        elif quota_diff[i] == -1:
                            # print('hit-1')
                            marginal[i] = -gain_acc_diff[i]

                    if self._rank == 0:
                        print('New marginal: ', marginal)


                    random_change = num_label // 40
                    to_change = random_change * 4
                    pairs = [(i, marginal[i]) for i in range(num_label)]
                    def key(item):
                        return item[1]
                    pairs.sort(key=key)
                    add_quota, remove_quota = [], []
                    for i in range(num_label):
                        idx = pairs[i][0]
                        if self.old_quota[idx] <= 6:
                            continue
                        else:
                            remove_quota.append(idx)
                        if len(remove_quota) >= to_change:
                            new_upper_bound = i
                            break
                    for i in range(num_label - 1, new_upper_bound, -1):
                        idx = pairs[i][0]
                        if self.old_quota[idx] >= 14:
                            continue
                        else:
                            add_quota.append(idx)
                        if len(add_quota) >= to_change:
                            break
                    if len(add_quota) != len(remove_quota):
                        minlen = min(len(add_quota), len(remove_quota))
                        add_quota = add_quota[:minlen]
                        remove_quota = remove_quota[:minlen]
                    quota_edit = set(add_quota + remove_quota)
                    random_add_quota, random_remove_quota = [], []
                    for i in range(random_change):
                        to_change = random.choice(range(num_label))
                        while to_change in quota_edit or self.old_quota[to_change] >= 14:
                            to_change = random.choice(range(num_label))
                        random_add_quota.append(to_change)
                        quota_edit.add(to_change)
                    for i in range(random_change):
                        to_change = random.choice(range(num_label))
                        while to_change in quota_edit or self.old_quota[to_change] <= 6:
                            to_change = random.choice(range(num_label))
                        random_remove_quota.append(to_change)
                        quota_edit.add(to_change)
                    self.old_quota = cp.deepcopy(self.new_quota)
                    for rm in (remove_quota + random_remove_quota):
                        self.new_quota[rm] -= 1
                    for ad in (add_quota + random_add_quota):
                        self.new_quota[ad] += 1
                    if self._rank == 0:
                        print('remove: ', remove_quota + random_remove_quota)
                        print('add: ', add_quota + random_add_quota)
                        print('old_quota: ', self.old_quota)
                        print('new_quota: ', self.new_quota)
                    self.marginal = marginal
                elif kwargs['dynamic_policy'] == 'naive':
                    gain_acc = self.new_gain_acc
                    num_label = kwargs['num_label']
                    gain_acc_tuples = []
                    for i in range(num_label):
                        gain_acc_tuples.append([i, gain_acc[i]])
                    def key(item):
                        return item[1]
                    gain_acc_tuples.sort(key=key)
                    self.old_quota = cp.deepcopy(self.new_quota)
                    to_change = num_label // 4
                    for i in range(to_change):
                        self.new_quota[gain_acc_tuples[i][0]] += 1


            if self._rank == 0:
                print('Training Acc Per Class:\n', self.new_train_class_acc, flush=True)
                print('Training Acc Gain Per Class:\n', self.new_gain_acc, flush=True)
                print('Marginal Effect of 1 quota:\n', self.marginal, flush=True)
                print('Current Quota:\n', self.new_quota, flush=True)


            self.old_train_class_acc = self.new_train_class_acc.clone()
            self.old_gain_acc = self.new_gain_acc.clone()
            self.hit -= self.hit
            self.tot -= self.tot



        self.call_hook('after_train_epoch')
        self._epoch += 1

    def val(self, data_loader, **kwargs):
        # the flag
        kwargs['validate'] = True
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.log_buffer.clear()
        self.val_result = []
        self.call_hook('before_val_epoch')

        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            with torch.no_grad():
                outputs = self.batch_processor(
                    self.model, data_batch, train_mode=False, **kwargs)
            if 'cls_score' in outputs:
                cls_score = outputs.pop('cls_score')
                self.val_result.append(cls_score.cpu().numpy())
            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])
            if 'batch_acc' in outputs:
                self.log_buffer.update({'batch_acc':  outputs['batch_acc']})
            self.outputs = outputs
            self.call_hook('after_val_iter')

        if self.val_result != []:
            val_result = np.concatenate(self.val_result, axis=0)
            len_val = val_result.shape[0]
            val_result = [val_result[i] for i in range(len_val)]
            with open(osp.join(self.work_dir, '{}.pkl'.format(self._rank)), 'wb') as fout:
                pickle.dump(val_result, fout)

        self.call_hook('after_val_epoch')
        if self.val_result != []:
            if self._rank == 0:
                results = []
                for i in range(self._world_size):
                    fin = open(osp.join(self.work_dir, '{}.pkl'.format(i)), 'rb')
                    results.append(pickle.load(fin))
                    fin.close()
                all_results = []
                for res in zip(*results):
                    all_results.extend(res)
                num_samples = len(data_loader.dataset)
                all_results = all_results[:num_samples]
                with open(osp.join(self.work_dir, 'val_{}.pkl'.format(self._epoch)), 'wb') as fout:
                    pickle.dump(all_results, fout)

                gts = [x.label for x in data_loader.dataset.video_infos]
                def intop(pred, label, n):
                    pred = list(map(lambda x: np.argsort(x)[-n:], pred))
                    hit = list(map(lambda l, p: l in p, label, pred))
                    return hit
                top1 = np.mean(intop(all_results, gts, 1))
                top5 = np.mean(intop(all_results, gts, 5))
                self.logger.info('VSummary: Epoch[{}] Top-1: {} Top-5: {}'.format(self._epoch, top1, top5))

    def resume(self, checkpoint, resume_optimizer=True,
               map_location='default'):
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        if self.use_dynamic:
            self.old_train_class_acc = checkpoint['meta']['old_train_class_acc']
            self.old_gain_acc = checkpoint['meta']['old_gain_acc']
            self.old_quota = checkpoint['meta']['old_quota']
            self.new_quota = checkpoint['meta']['new_quota']
            self.marginal = checkpoint['meta']['marginal']
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info('resumed epoch %d, iter %d', self.epoch, self.iter)

    def run(self, data_loaders, workflow, max_epochs, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
            max_epochs (int): Total training epochs.
        """
        # assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)

        self._max_epochs = max_epochs
        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow, max_epochs)
        self.epoch_len = len(data_loaders['train'][0])
        self.call_hook('before_run')

        while self.epoch < max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            'runner has no method named "{}" to run an epoch'.
                            format(mode))
                    epoch_runner = getattr(self, mode)
                elif callable(mode):  # custom train()
                    epoch_runner = mode
                else:
                    raise TypeError('mode in workflow must be a str or '
                                    'callable function, not {}'.format(
                                        type(mode)))
                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= max_epochs:
                        return
                    epoch_runner(data_loaders[mode], **kwargs)
            if self.should_stop:
                self.logger.info("nstep of decay exceeded, training terminates")
                break

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def register_lr_hooks(self, lr_config, priority):
        if isinstance(lr_config, LrUpdaterHook):
            self.register_hook(lr_config, priority)
        elif isinstance(lr_config, dict):
            assert 'policy' in lr_config
            # from .hooks import lr_updater
            hook_name = lr_config['policy'].title() + 'LrUpdaterHook'
            if not hasattr(lr_updater, hook_name):
                raise ValueError('"{}" does not exist'.format(hook_name))
            hook_cls = getattr(lr_updater, hook_name)
            self.register_hook(hook_cls(**lr_config), priority)
        else:
            raise TypeError('"lr_config" must be either a LrUpdaterHook object'
                            ' or dict, not {}'.format(type(lr_config)))

    def register_logger_hooks(self, log_config):
        log_interval = log_config['interval']
        for info in log_config['hooks']:
            logger_hook = obj_from_dict(
                info, hooks, default_args=dict(interval=log_interval))
            self.register_hook(logger_hook, priority='VERY_LOW')

    def register_training_hooks(self,
                                lr_config,
                                optimizer_config=None,
                                checkpoint_config=None,
                                log_config=None):
        """Register default hooks for training.

        Default hooks include:

        - LrUpdaterHook
        - OptimizerStepperHook
        - CheckpointSaverHook
        - IterTimerHook
        - LoggerHook(s)
        """
        if optimizer_config is None:
            optimizer_config = {}
        if checkpoint_config is None:
            checkpoint_config = {}
        self.register_lr_hooks(lr_config, 'LOWEST')
        self.register_hook(self.build_hook(optimizer_config, OptimizerHook))
        self.register_hook(self.build_hook(checkpoint_config, CheckpointHook))
        self.register_hook(IterTimerHook())
        if log_config is not None:
            self.register_logger_hooks(log_config)
