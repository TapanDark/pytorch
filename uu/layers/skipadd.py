import time
from typing import Dict
from uu.layers.sequential import mSequential
from uu.utils import ftensor as ft
from uu.utils import memory, padding_calc
from uu.utils.context_control import conv_2d_ctx

import numpy as np

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

myctx_dict = {}
# for correctness debug
USE_DEFAULT_CTX = False


class TiledSkipAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2):
        print("input1 : %s" % input1)
        print("input2 : %s" % input2)
        return input1 + input2


class TiledSkipAdd(torch.nn.Module):
    def __init__(
        self,
        *modules,
        is_ccheckpoint=False
    ):
        super(TiledSkipAdd, self).__init__()
        self.is_ccheckpoint = is_ccheckpoint
        self.skipLayers = mSequential(*modules[0])

    def forward(self, *inputs) -> Tensor:
        if type(inputs[0]) == tuple:
            # to remove additional packing in tuple
            inputs = list(inputs[0])

        if len(inputs) == 2:
            input, info = inputs
            self.is_ccheckpoint = False
        elif len(inputs) == 3:
            input, info, is_ccheckpoint = inputs
            self.is_ccheckpoint = is_ccheckpoint
        else:
            assert False, "missing info in cskipAdd"

        input2 = self.skipLayers(*inputs)
        out = input + input2[0]  # TODO: Can we allow the user to define this?
        return out, info, self.is_ccheckpoint
