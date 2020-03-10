# Copyright (c) Open-MMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

from ..hook import Hook
from ...utils import get_dist_info, get_host_info, get_time_str, obj_from_dict
import torch.distributed as dist
import torch


class LoggerHook(Hook):
    """Base class for logger hooks.

    Args:
        interval (int): Logging interval (every k iterations).
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`.
        reset_flag (bool): Whether to clear the output buffer after logging.
    """

    __metaclass__ = ABCMeta

    def __init__(self, interval=10, ignore_last=False, reset_flag=False):
        self.interval = interval
        self.ignore_last = ignore_last
        self.reset_flag = reset_flag
        _rank, _world_size = get_dist_info()
        self.rank = _rank
        self.world_size = _world_size

    @abstractmethod
    def log(self, runner):
        pass

    def before_run(self, runner):
        for hook in runner.hooks[::-1]:
            if isinstance(hook, LoggerHook):
                hook.reset_flag = True
                break

    def before_epoch(self, runner):
        runner.log_buffer.clear()  # clear logs of last epoch

    def after_train_iter(self, runner):
        _rank, _world_size = get_dist_info()
        # print('my rank is: ', _rank, 'after iteration')
        if self.every_n_inner_iters(runner, self.interval):
            runner.log_buffer.average(self.interval)
            self.sync_buffer_output(runner)
        elif self.end_of_epoch(runner) and not self.ignore_last:
            # not precise but more stable
            runner.log_buffer.average(self.interval)
            self.sync_buffer_output(runner)
        if runner.log_buffer.ready:
            self.log(runner, 'iter')
            if self.reset_flag:
                runner.log_buffer.clear_output()

    def after_train_epoch(self, runner):
        runner.log_buffer.average()
        self.sync_buffer_output(runner)
        if runner.log_buffer.ready:
            self.log(runner)
            if self.reset_flag:
                runner.log_buffer.clear_output()
        if 'batch_acc' in runner.log_buffer.output:
            runner.train_acc = runner.log_buffer.output['batch_acc']

    def after_val_epoch(self, runner):
        runner.log_buffer.average()
        self.sync_buffer_output(runner)
        if runner.log_buffer.ready:
            self.log(runner, 'epoch')
        # haodong mod, for ReduceLROnPlateau support
        if 'batch_acc' in runner.log_buffer.output:
            runner.val_acc = runner.log_buffer.output['batch_acc']

        if self.reset_flag:
            runner.log_buffer.clear_output()

    def after_val_iter(self, runner):
        if self.every_n_inner_iters(runner, self.interval):
            runner.log_buffer.average(self.interval)
            self.sync_buffer_output(runner)
        elif self.end_of_epoch(runner) and not self.ignore_last:
            runner.log_buffer.average(self.interval)
            self.sync_buffer_output(runner)
        if runner.log_buffer.ready:
            self.log(runner, 'iter')
            if self.reset_flag:
                runner.log_buffer.clear_output()

    def sync_buffer_output(self, runner):
        for k, v in runner.log_buffer.output.items():
            tmp_tensor = torch.Tensor([v]).cuda(torch.cuda.current_device())
            dist.all_reduce(tmp_tensor)
            tmp_tensor.div_(self.world_size)
            runner.log_buffer.output[k] = tmp_tensor.item()
