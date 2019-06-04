from torch.nn.utils import clip_grad

from .hook import Hook


class OptimizerHook(Hook):

    def __init__(self, grad_clip=None, iter_size=1):
        self.grad_clip = grad_clip
        self.iter_size = iter_size

    def clip_grads(self, params):
        clip_grad.clip_grad_norm_(
            filter(lambda p: p.requires_grad, params), **self.grad_clip)

    def before_run(self, runner):
        runner.optimizer.zero_grad()

    def after_train_iter(self, runner):
        iter_loss = runner.outputs['loss'] / self.iter_size
        iter_loss.backward()
        if (runner.iter % self.iter_size) == 0:
            if self.grad_clip is not None:
                self.clip_grads(runner.model.parameters())

            runner.optimizer.step()
            runner.optimizer.zero_grad()
