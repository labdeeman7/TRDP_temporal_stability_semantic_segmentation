# Code adapted from "TSM: Temporal Shift Module for Efficient Video Understanding"
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu
# We wrapping the conv. that is what we doing.

import torch
import torch.nn as nn
import torch.nn.functional as F

## resnet has the residual done for you automatically. 
##* we assuming 0 to 64.

class TemporalShift(nn.Module):
    def __init__(self, net, n_div=8):
        super(TemporalShift, self).__init__()
        self.net = net
        self.fold_div = n_div
        print('=> Using fold div: {}'.format(self.fold_div))
        self.shifted_features = []

    def forward(self, x):
        x, shifted_features = self.shift(x,  fold_div=self.fold_div)
        self.shifted_features = shifted_features
        return self.net(x)

    #** note, you are passing all of the images at once, through the system as if they were a single batch.
    @staticmethod
    def shift(x, fold_div=8):
        _, c, _, _ = x.size()
        fold = c // fold_div
        out = torch.zeros_like(x)
        out[1:, :fold, :] = x[:-1, :fold, :]
        out[:, fold:, :] = x[:, fold:, :]
        return out, x[:-1, :fold, :]


def make_temporal_shift(net, n_div=8):   
    n_round = 1
    if len(list(net.layer3.children())) >= 23:
        n_round = 2
        print('=> Using n_round {} to insert temporal shift'.format(n_round))

    def make_block_temporal(stage):
        blocks = list(stage.children())
        print('=> Processing stage with {} blocks residual'.format(len(blocks)))
        for i, b in enumerate(blocks):
            if i % n_round == 0:
                blocks[i].conv1 = TemporalShift(b.conv1, n_div=n_div)
        return nn.Sequential(*blocks)

    net.layer1 = make_block_temporal(net.layer1)
    net.layer2 = make_block_temporal(net.layer2)
    net.layer3 = make_block_temporal(net.layer3)
    net.layer4 = make_block_temporal(net.layer4)





