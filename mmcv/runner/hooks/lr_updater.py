# Copyright (c) Open-MMLab. All rights reserved.
from __future__ import division
from math import cos, pi

from .hook import HOOKS, Hook


class LrUpdaterHook(Hook):

    def __init__(self,
                 by_epoch=True,
                 warmup=None,
                 warmup_iters=0,
                 warmup_ratio=0.1,
                 warmup_byepoch=False,
                 **kwargs):
        # validate the "warmup" argument
        if warmup is not None:
            if warmup not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    '"{}" is not a supported type for warming up, valid types'
                    ' are "constant" and "linear"'.format(warmup))
        if warmup is not None:
            assert warmup_iters > 0, \
                '"warmup_iters" must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, \
                '"warmup_ratio" must be in range (0,1]'

        self.by_epoch = by_epoch
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_byepoch = warmup_byepoch
        self.warmup_ratio = warmup_ratio

        self.base_lr = []  # initial lr for all param groups
        self.regular_lr = []  # expected lr if no warming up is performed

    # ok
    def _set_lr(self, runner, lr_groups):
        for param_group, lr in zip(runner.optimizer.param_groups, lr_groups):
            param_group['lr'] = lr

    def get_lr(self, runner, base_lr):
        raise NotImplementedError

    # ok
    def get_regular_lr(self, runner):
        return [self.get_lr(runner, _base_lr) for _base_lr in self.base_lr]

    def get_warmup_lr(self, cur_iters):
        if self.warmup == 'constant':
            warmup_lr = [_lr * self.warmup_ratio for _lr in self.regular_lr]
        elif self.warmup == 'linear':
            k = (1 - cur_iters / self.warmup_iters) * (1 - self.warmup_ratio)
            warmup_lr = [_lr * (1 - k) for _lr in self.regular_lr]
        elif self.warmup == 'exp':
            k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
            warmup_lr = [_lr * k for _lr in self.regular_lr]
        return warmup_lr

    # ok
    def before_run(self, runner):
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params
        for group in runner.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.base_lr = [
            group['initial_lr'] for group in runner.optimizer.param_groups
        ]
        if self.warmup_byepoch:
            runner.logger.info('Warmup By Epoch, epoch_len is {}, total warmup iter is {}'.format(runner.epoch_len, self.warmup_iters * runner.epoch_len))
            self.warmup_iters *= runner.epoch_len
        else:
            runner.logger.info('Warmup By Iter, total warmup iter is {}'.format(self.warmup_iters))

    def before_train_epoch(self, runner):
        if not self.by_epoch:
            return
        self.regular_lr = self.get_regular_lr(runner)
        self._set_lr(runner, self.regular_lr)

    def before_train_iter(self, runner):
        cur_iter = runner.iter
        if not self.by_epoch:
            self.regular_lr = self.get_regular_lr(runner)
            if self.warmup is None or cur_iter >= self.warmup_iters:
                self._set_lr(runner, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(runner, warmup_lr)
        elif self.by_epoch:
            if self.warmup is None or cur_iter > self.warmup_iters:
                return
            elif cur_iter == self.warmup_iters:
                self._set_lr(runner, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(runner, warmup_lr)


@HOOKS.register_module
class FixedLrUpdaterHook(LrUpdaterHook):

    def __init__(self, **kwargs):
        super(FixedLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        return base_lr


@HOOKS.register_module
class StepLrUpdaterHook(LrUpdaterHook):

    def __init__(self, step, gamma=0.1, **kwargs):
        assert isinstance(step, (list, int))
        if isinstance(step, list):
            for s in step:
                assert isinstance(s, int) and s > 0
        elif isinstance(step, int):
            assert step > 0
        else:
            raise TypeError('"step" must be a list or integer')
        self.step = step
        self.gamma = gamma
        super(StepLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else runner.iter

        if isinstance(self.step, int):
            return base_lr * (self.gamma**(progress // self.step))

        exp = len(self.step)
        for i, s in enumerate(self.step):
            if progress < s:
                exp = i
                break
        return base_lr * self.gamma**exp


@HOOKS.register_module
class ExpLrUpdaterHook(LrUpdaterHook):

    def __init__(self, gamma, **kwargs):
        self.gamma = gamma
        super(ExpLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else runner.iter
        return base_lr * self.gamma**progress


@HOOKS.register_module
class PolyLrUpdaterHook(LrUpdaterHook):

    def __init__(self, power=1., min_lr=0., **kwargs):
        self.power = power
        self.min_lr = min_lr
        super(PolyLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters
        coeff = (1 - progress / max_progress)**self.power
        return (base_lr - self.min_lr) * coeff + self.min_lr


@HOOKS.register_module
class InvLrUpdaterHook(LrUpdaterHook):

    def __init__(self, gamma, power=1., **kwargs):
        self.gamma = gamma
        self.power = power
        super(InvLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else runner.iter
        return base_lr * (1 + self.gamma * progress)**(-self.power)


@HOOKS.register_module
class CosineLrUpdaterHook(LrUpdaterHook):

    def __init__(self, target_lr=0, **kwargs):
        self.target_lr = target_lr
        super(CosineLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters
        return self.target_lr + 0.5 * (base_lr - self.target_lr) * \
            (1 + cos(pi * (progress / max_progress)))


# Only Support
class CyclicCosineLrUpdaterHook(LrUpdaterHook):

    def __init__(self, target_lr=0, mode='uniform', ncycle=4, **kwargs):
        super(CyclicCosineLrUpdaterHook, self).__init__(**kwargs)
        self.target_lr = target_lr
        self.mode = mode
        self.ncycle = ncycle
        self.cosine_segs = None

    def init_cosine_segs(self, runner):
        self.cosine_segs = []
        if self.by_epoch:
            max_progress = runner.max_epochs
        else:
            max_progress = runner.max_iters
        if self.warmup is not None:
            n_warmup = self.warmup_iters
        else:
            n_warmup = 0

        if self.mode == 'uniform':
            for i in range(self.ncycle):
                st = n_warmup + int(i * (max_progress - n_warmup) / self.ncycles)
                ed = n_warmup + int((i + 1) * (max_progress - n_warmup) / self.ncycles)
                self.cosine_segs.append([st, ed])
        elif self.mode == 'exp':
            n_parts = sum(map(2 ** i, range(self.ncycle)))
            st, ed = 0, 0
            for i in range(self.ncycle):
                st = ed
                ed = st + 2 ** i
                st_epoch = n_warmup + int(st * (max_progress - n_warmup) / n_parts)
                ed_epoch = n_warmup + int(ed * (max_progress - n_warmup) / n_parts)
                self.cosine_segs.append([st, ed])



    def get_lr(self, runner, base_lr):
        if self.cosine_segs == None:
            self.init_cosine_segs(runner)

        this_seg = None
        if self.by_epoch:
            progress = runner.epoch
        else:
            progress = runner.iter
        for seg in self.cosine_segs:
            if seg[0] <= progress < seg[1]:
                this_seg = seg

        if this_seg == None:
            return base_lr
        else:
            return self.target_lr + 0.5 * (base_lr - self.target_lr) * \
                (1 + cos(pi * (progress / (this_seg[1] - this_seg[0]))))


# class PlateauLrUpdaterHook(LrUpdaterHook):
#     def __init__(self, tolerance, nstep, gamma, history, key, **kwargs):
#         self.tolerance = tolerance
#         self.nstep = nstep
#         self.gamma = gamma
#         self.step_history = history
#         self.decayed_steps = 0
#         self.key = key
#         super(PlateauLrUpdaterHook, self).__init__(**kwargs)
#
#
#     def get_lr(self, runner, base_lr):
#         return base_lr * (self.gamma) ** self.decayed_steps
#
#     def after_val_epoch(self, runner):
#         if not 'val' in self.key:
#             return
#         if hasattr(runner, self.key):
#             step_value = getattr(runner, self.key)
#         else:
#             runner.logger.info("runner not have designated step value")
#             exit(1)
#         self.step_history.append(step_value)
#         if len(self.step_history) > self.tolerance + 1:
#             max_value = max(self.step_history)
#             recent_max_value = max(self.step_history[-self.tolerance: ])
#             if recent_max_value < max_value:
#                 self.decayed_steps = self.decayed_steps + 1
#                 self.step_history = [max_value]
#                 runner.logger.info('tolerance exceeded, LR decayed.')
#         if self.decayed_steps > self.nstep:
#             runner.should_stop = True
#         runner.logger.info('history_step_values: {}, decayed_steps: {}'.format(self.step_history, self.decayed_steps))
#
#     def after_train_epoch(self, runner):
#         if not 'train' in self.key:
#             return
#         if hasattr(runner, self.key):
#             step_value = getattr(runner, self.key)
#         else:
#             runner.logger.info("runner not have designated step value")
#             exit(1)
#         self.step_history.append(step_value)
#         if len(self.step_history) > self.tolerance + 1:
#             max_value = max(self.step_history)
#             recent_max_value = max(self.step_history[-self.tolerance: ])
#             if recent_max_value < max_value:
#                 self.decayed_steps = self.decayed_steps + 1
#                 self.step_history = [max_value]
#                 runner.logger.info('tolerance exceeded, LR decayed.')
#         if self.decayed_steps > self.nstep:
#             runner.should_stop = True
#         runner.logger.info('history_step_values: {}, decayed_steps: {}'.format(self.step_history, self.decayed_steps))
