from .hook import Hook


class DistSamplerSeedHook(Hook):

    def before_epoch(self, runner):
        if isinstance(runner.data_loader, (list, tuple)):
            for i in range(len(runner.data_loader)):
                runner.data_loader[i].sampler.set_epoch(runner.epoch)
        else:
            runner.data_loader.sampler.set_epoch(runner.epoch)
