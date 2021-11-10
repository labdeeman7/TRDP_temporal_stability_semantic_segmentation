# Code adapted from "TSM: Temporal Shift Module for Efficient Video Understanding"
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch
import torch.nn as nn
import torch.nn.functional as F

## resnet has the residual done for you automatically. 

class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        self.feature_map = x
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        return self.net(x)

    #** note, you are passing all of the images at once, through the system as if they were a single batch.
    @staticmethod
    def shift(x, n_segment, fold_div=8, inplace=False):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  
        # t_1 replaced by t_0, from 0 till fold  #need to ensure that the image has position 0 and n_segments of previous images is 1 to n. 
        out[:, :, fold:] = x[:, :,fold:]  # not shift

        return out.view(nt, c, h, w)


def make_temporal_shift(net, n_segment=2, n_div=8):   
    n_round = 1
    if len(list(net.layer3.children())) >= 23:
        n_round = 2
        print('=> Using n_round {} to insert temporal shift'.format(n_round))

    def make_block_temporal(stage):
        blocks = list(stage.children())
        print('=> Processing stage with {} blocks residual'.format(len(blocks)))
        for i, b in enumerate(blocks):
            if i % n_round == 0:
                blocks[i].conv1 = TemporalShift(b.conv1, n_segment, n_div=n_div)
        return nn.Sequential(*blocks)

    net.layer1 = make_block_temporal(net.layer1)
    net.layer2 = make_block_temporal(net.layer2)
    net.layer3 = make_block_temporal(net.layer3)
    net.layer4 = make_block_temporal(net.layer4)





