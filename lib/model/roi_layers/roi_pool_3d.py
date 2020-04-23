import time
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable

class roi_pooling(Function):

    @staticmethod
    def forward(ctx, input, rois, size, spatial_scale):
        output = []
        rois = rois.data.float()
        num_rois = rois.size(1)
        has_backward = True
        rois[:, :, 1:].mul_(spatial_scale)
        for i in range(num_rois):
            roi = rois[:, i, 1:]
            im = input[..., roi[:,0]:(roi[:,3] + 1), roi[:,1]:(roi[:,4] + 1), roi[:,2]:(roi[:,5] + 1)]
            output.append(F.adaptive_max_pool3d(im, size))
        output = torch.cat(output, 0)
        if has_backward:
            output.sum().backward()
        return output
roi_pooling = roi_pooling.apply

class ROIPool_3d(nn.Module):
    def __init__(self, output_size, spatial_scale):
        super(ROIPool_3d, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, input, rois):
        return roi_pooling(input, rois, self.output_size, self.spatial_scale)