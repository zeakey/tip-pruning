import numpy as np
import torch
import torch.nn as nn


class channel_selection(nn.Module):
    """
    Select channels from the output of BatchNorm2d layer. It should be put directly after BatchNorm2d layer.
    The output shape of this layer is determined by the number of 1 in `self.indices`.
    """
    def __init__(self, num_channels):
        """
        Initialize the `indices` with all one vector with the length same as the number of channels.
        During pruning, the places in `indices` which correpond to the channels to be pruned will be set to 0.
        """
        super(channel_selection, self).__init__()
        self.register_buffer("indices", torch.ones(num_channels, dtype=bool))

    def forward(self, input_tensor):
        """
        Parameter
        ---------
        input_tensor: (N,C,H,W). It should be the output of BatchNorm2d layer.
        """
        
        output = input_tensor[:, self.indices]
        return output