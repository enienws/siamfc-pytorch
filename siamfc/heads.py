from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['SiamFC']


class SiamFC(nn.Module):

    def __init__(self, out_scale=0.001, alpha=None):
        super(SiamFC, self).__init__()
        self.out_scale = out_scale

        if alpha is not None:
            self.alpha = alpha
        else:
            self.alpha = nn.Parameter(torch.rand(1, requires_grad=True))
    
    def forward(self, z, x):
        resp1 = self._fast_xcorr(z[0], x[0]) * self.out_scale
        #Normalization
        # m1 = -3.1381075
        # s1 = 3.584974
        # resp1_norm = resp1 - m1
        # resp1_norm = resp1_norm / s1


        resp2 = self._fast_xcorr(z[1], x[1]) * self.out_scale
        #Normalization
        # m2 = 0.5537824
        # s2 = 1.715082
        # resp2_norm = resp2 - m2
        # resp2_norm = resp2_norm / s2

        #Weighted average
        # resp = torch.add(torch.mul(resp1_norm, 1-alpha), torch.mul(resp2_norm, alpha))
        if type(self.alpha) is torch.nn.Parameter:
            # resp = resp1 * (torch.ones(1).expand_as(resp1).to('cuda:1')- alpha.expand_as(resp1)) + resp2 * alpha.expand_as(resp2)
            resp = torch.add(torch.mul(resp1, 1 - self.alpha), torch.mul(resp2, self.alpha))
        else:
            resp = torch.add(torch.mul(resp1, 1 - self.alpha), torch.mul(resp2, self.alpha))
        # return resp1, resp2
        return resp
    
    def _fast_xcorr(self, z, x):
        # fast cross correlation
        nz = z.size(0)
        nx, c, h, w = x.size()
        x = x.view(-1, nz * c, h, w)
        out = F.conv2d(x, z, groups=nz)
        out = out.view(nx, -1, out.size(-2), out.size(-1))
        return out
