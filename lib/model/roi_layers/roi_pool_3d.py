import time
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.autograd import Variable
import six
import numpy
class roi_pooling(Function):

    @staticmethod
    def forward(ctx, input, rois, size, spatial_scale):
        Ada_pooling = nn.AdaptiveMaxPool3d(size,True)
        ctx.output_size = size
        ctx.spatial_scale = spatial_scale
        output = []
        argsmax_data = []
        rois = rois.data.float()
        num_rois = rois.size(1)

        rois[:, :, 1:].mul_(spatial_scale)
        for i in range(num_rois):
            roi = rois[:, i, 1:]
            w_ = int(torch.round(roi[:,3] + 1)) - int(torch.round(roi[:,0]))
            h_ = int(torch.round(roi[:,4] + 1)) - int(torch.round(roi[:,1]))
            s_ = int(torch.round(roi[:,5] + 1)) - int(torch.round(roi[:,2]))
            im = input[..., int(torch.round(roi[:,0])):int(torch.round(roi[:,3] + 1)), \
                int(torch.round(roi[:,1])):int(torch.round(roi[:,4] + 1)), int(torch.round(roi[:,2])):int(torch.round(roi[:,5] + 1))].cuda()
            out = Ada_pooling(im)
            output.append(out[0])
            argsmax_data.append(out[1])
        argsmax_data = torch.cat(argsmax_data,0)
        ctx.save_for_backward(input,roi,argsmax_data)
        output = torch.cat(output, 0)
        #if has_backward:
        #    output.sum().backward()
        return output
    @staticmethod
    @once_differentiable
    def backward(ctx,grad_output):
        input, rois, argmax = ctx.saved_tensors
        size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        grad_input = numpy.zeros(input.shape,rois.dtype)
        channels, height, width, slices = input.shape[1:]

        for i in six.moves.range(rois.size()[1]):
            xmin, ymin, zmin, xmax, ymax, zmax = rois[0][i][1:]
            xmin = int(round(xmin*spatial_scale))
            xmax = int(round(xmax*spatial_scale))
            ymin = int(round(ymin*spatial_scale))
            ymax = int(round(ymax*spatial_scale))
            zmin = int(round(zmin*spatial_scale))
            zmax = int(round(zmax*spatial_scale))

            roi_width = max(xmax - xmin + 1, 1)
            roi_height = max(ymax - ymin + 1, 1)
            roi_slice = max(zmax - zmin + 1, 1)

            stridew = float(roi_width) / float(size[0])
            strideh = float(roi_height) / float(size[1])
            strides = float(roi_slice) / float(size[2])

            for w in six.moves.range(xmin, xmax + 1):
                for h in six.moves.range(ymin, ymax + 1):
                    for z in six.moves.range(zmin, zmax+1):
                        phstart = int(numpy.floor(float(h - ymin) / strideh))
                        phend = int(numpy.ceil(float(h - ymin + 1) / strideh))
                        pwstart = int(numpy.floor(float(w - xmin) / stridew))
                        pwend = int(numpy.ceil(float(w - xmin + 1) / stridew))
                        psstart = int(numpy.floor(float(z - zmin) / strides))
                        psend = int(numpy.ceil(float(z - zmin + 1) / strides))

                        phstart = min(max(phstart, 0), size[1])
                        phend = min(max(phend, 0), size[1])
                        pwstart = min(max(pwstart, 0), size[0])
                        pwend = min(max(pwend, 0), size[0])
                        psstart = min(max(psstart, 0), size[2])
                        psend = min(max(psend, 0), size[2])

                        for ph in six.moves.range(phstart, phend):
                            for pw in six.moves.range(pwstart, pwend):
                                for ps in six.moves.range(psstart, psend):
                                    max_idx_tmp = argmax[i, :, ph, pw, ps]
                                    for c in six.moves.range(channels):
                                        if max_idx_tmp[c] == (h * width + w):
                                            grad_input[i, c, h, w, z] += \
                                                grad_output[i, c, ph, pw, ps]

        return Variable(grad_input), None, None, None

roi_pooling = roi_pooling.apply

class ROIPool_3d(nn.Module):
    def __init__(self, output_size, spatial_scale):
        super(ROIPool_3d, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, input, rois):
        return roi_pooling(input, rois, self.output_size, self.spatial_scale)