# Copyright (c) Open-MMLab. All rights reserved.

from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from apex.parallel import DistributedDataParallel as APEXDDP
except ImportError:
    pass
from .scatter_gather import scatter_kwargs
import torch

class MMDistributedDataParallel(DDP):

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids)
    def forward(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs,
                                      [torch.cuda.current_device()])
        return self.module(*inputs[0], **kwargs[0])


class MMAPEXDistributedDataParallel(APEXDDP):

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids)
    def forward(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs,
                                      [torch.cuda.current_device()])
        return self.module(*inputs[0], **kwargs[0])
