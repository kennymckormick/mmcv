# Copyright (c) Open-MMLab. All rights reserved.
import logging
import os.path as osp
import time

import torch
import psutil
import os
import sys
import gc
from . import hooks
from .log_buffer import LogBuffer
from torch.utils.data import DataLoader
from .hooks import (Hook, LrUpdaterHook, CheckpointHook, IterTimerHook,
                    OptimizerHook, lr_updater)
from mmcv.parallel import collate
import mmcv

from .checkpoint import load_checkpoint, save_checkpoint
from .dist_utils import get_dist_info
from .hooks import HOOKS, Hook, IterTimerHook
from .log_buffer import LogBuffer
from .priority import get_priority
from .utils import get_host_info, get_time_str, obj_from_dict, mixup
import torch.distributed as dist
import random
import math
import itertools
import copy as cp
import pickle
import numpy as np
from torch.utils.data.sampler import Sampler

fp16_enabled = True
try:
    from apex import amp
except ImportError:
    fp16_enabled = False
    pass


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
        self.num_samples = math.ceil(
            len(dataset) / samples_per_gpu / num_replicas) * samples_per_gpu
        self.tot_samples = self.num_samples * num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.multinomial(torch.Tensor(
            self.handfreq), self.tot_samples, replacement=True, generator=g)
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
    sampler = DistributedHandFreqSampler(
        dataset, freq, batch_size, world_size, rank)
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
        meta (dict | None): A dict records some import information such as
            environment info and seed, which will be logged in logger hook.
    """

    def __init__(self,
                 model,
                 batch_processor,
                 optimizer=None,
                 work_dir=None,
                 log_level=logging.INFO,
                 logger=None,
                 use_fp16=False,
                 meta=None):
        assert callable(batch_processor)
        global fp16_enabled

        self._rank, self._world_size = get_dist_info()

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
        self.val_result = []

        self.timestamp = get_time_str()
        if meta is not None:
            assert isinstance(meta, dict), '"meta" must be a dict or None'
        self.meta = meta

        # create work_dir
        if mmcv.is_str(work_dir):
            self.work_dir = osp.abspath(work_dir)
            mmcv.mkdir_or_exist(self.work_dir)
        elif work_dir is None:
            self.work_dir = None
        else:
            raise TypeError('"work_dir" must be a str or None')

        if logger is None:
            # print('my rank is ', self._rank, 'my logger is none')
            self.logger = self.init_logger(work_dir, log_level)
        else:
            self.logger = logger
        self.log_buffer = LogBuffer()

        self.model = model
        if optimizer is not None:
            self.optimizer = self.init_optimizer(optimizer)
        else:
            self.optimizer = None
        self.batch_processor = batch_processor
        self.use_fp16 = use_fp16
        if not fp16_enabled:
            self.logger.info('fp16 not enabled on this machine')
            self.use_fp16 = False

        if self.use_fp16:
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer)

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

    @property
    def model_name(self):
        """str: Name of the model, usually the module class name."""
        return self._model_name

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
            bn_params = []
            non_bn_params = []
            flow_layers_params = []
            for name, p in self.model.named_parameters():
                if 'bn' in name:
                    bn_params.append(p)
                elif 'flow_layer' in name:
                    flow_layers_params.append(p)
                else:
                    non_bn_params.append(p)
            weight_decay = optimizer['weight_decay']
            lr = optimizer['lr']

            if 'bn_nowd' in optimizer and optimizer['bn_nowd']:
                if flow_layers_params == []:
                    optim_params = [{'params': bn_params, 'weight_decay': 0},
                                    {'params': non_bn_params}]
                else:
                    optim_params = [{'params': bn_params, 'weight_decay': 0},
                                    {'params': non_bn_params},
                                    {'params': flow_layers_params, 'lr': lr * 0.01}]
            else:
                if flow_layers_params == []:
                    optim_params = [{'params': non_bn_params + bn_params}]
                else:
                    optim_params = [{'params': non_bn_params + bn_params},
                                    {'params': flow_layers_params, 'lr': lr * 0.01}]
            if 'bn_nowd' in optimizer:
                optimizer.pop('bn_nowd')
            if len(optim_params) == 1:
                optimizer = obj_from_dict(optimizer, torch.optim,
                                          dict(params=self.model.parameters()))
            else:
                optimizer = obj_from_dict(
                    optimizer, torch.optim, dict(params=optim_params))
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
            filename = '{}.log'.format(self.timestamp)
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
                        meta=None,
                        create_symlink=True):
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        linkname = osp.join(out_dir, 'latest.pth')

        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            mmcv.symlink(filename, osp.join(out_dir, 'latest.pth'))

    def train(self, data_loader, **kwargs):
        # The flag
        kwargs['validate'] = False
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(data_loader[0])
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
            batch_flags = kwargs.pop('batch_flags')

        if 'train_ratio' in kwargs:
            use_aux_per_niter = kwargs['train_ratio'][0]
        else:
            use_aux_per_niter = 1

        if len(data_loader) > 1:
            main_data_loader = data_loader[0]
            auxiliary_data_loaders = data_loader[1:]
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
                    data_dict = mixup(data_dict, spatial_mixup,
                                      temporal_mixup, mixup_beta)

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

                if not isinstance(outputs, dict):
                    raise TypeError('batch_processor() must return a dict')
                if 'log_vars' in outputs:
                    self.log_buffer.update(
                        outputs['log_vars'], outputs['num_samples'])
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
                            raise TypeError(
                                'batch_processor() must return a dict')
                        if 'log_vars' in outputs:
                            self.log_buffer.update(
                                outputs['log_vars'], outputs['num_samples'])
                        self.outputs = outputs
                        self.call_hook('after_train_iter')
                self._iter += 1

        else:
            main_data_loader = data_loader[0]
            for i, data_batch in enumerate(main_data_loader):
                runner_info['this_iter'] = self._iter
                if i % 100 == 0 and self.rank == 0:
                    memoStats()
                self._inner_iter = i
                self.call_hook('before_train_iter')
                outputs = self.batch_processor(
                    self.model, data_batch, train_mode=True, runner_info=runner_info, **kwargs)

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
        if 'batch_flags' in kwargs:
            batch_flag = kwargs.pop('batch_flags')[0]
        else:
            batch_flag = ''
        kwargs['batch_flag'] = batch_flag

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
                    fin = open(
                        osp.join(self.work_dir, '{}.pkl'.format(i)), 'rb')
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
                self.logger.info(
                    'VSummary: Epoch[{}] Top-1: {} Top-5: {}'.format(self._epoch, top1, top5))

    def resume(self,
               checkpoint,
               resume_optimizer=True,
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
        self._iter = -1
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info('resumed epoch %d, iter %d', self.epoch, -1)

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
        if self._iter == -1:
            # n gpu may change if you use resume
            self._iter = self._epoch * self.epoch_len
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
                self.logger.info(
                    "nstep of decay exceeded, training terminates")
                break

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def register_lr_hook(self, lr_config):
        if isinstance(lr_config, dict):
            assert 'policy' in lr_config
            hook_type = lr_config.pop('policy').title() + 'LrUpdaterHook'
            lr_config['type'] = hook_type
            hook = mmcv.build_from_cfg(lr_config, HOOKS)
        else:
            hook = lr_config
        self.register_hook(hook)

    def register_optimizer_hook(self, optimizer_config):
        if optimizer_config is None:
            return
        if isinstance(optimizer_config, dict):
            optimizer_config.setdefault('type', 'OptimizerHook')
            hook = mmcv.build_from_cfg(optimizer_config, HOOKS)
        else:
            hook = optimizer_config
        self.register_hook(hook)

    def register_checkpoint_hook(self, checkpoint_config):
        if checkpoint_config is None:
            return
        if isinstance(checkpoint_config, dict):
            checkpoint_config.setdefault('type', 'CheckpointHook')
            hook = mmcv.build_from_cfg(checkpoint_config, HOOKS)
        else:
            hook = checkpoint_config
        self.register_hook(hook)

    def register_logger_hooks(self, log_config):
        log_interval = log_config['interval']
        for info in log_config['hooks']:
            logger_hook = mmcv.build_from_cfg(
                info, HOOKS, default_args=dict(interval=log_interval))
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
        self.register_lr_hook(lr_config)
        self.register_optimizer_hook(optimizer_config)
        self.register_checkpoint_hook(checkpoint_config)
        self.register_hook(IterTimerHook())
        self.register_logger_hooks(log_config)
