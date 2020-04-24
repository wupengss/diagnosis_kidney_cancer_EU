from __future__ import absolute_import

import numpy as np
import torch

def nms_cpu_3d(dets, scores, thresh):
    #dets = dets.numpy()
    x1 = dets[:, 0].cpu()
    y1 = dets[:, 1].cpu()
    z1 = dets[:, 2].cpu()
    x2 = dets[:, 3].cpu()
    y2 = dets[:, 4].cpu()
    z2 = dets[:, 5].cpu()
    scores = scores

    areas = (x2 - x1 + 1) * (y2 - y1 + 1) * (z2 - z1 + 1)
    order = scores.argsort(descending=True)

    keep = []
    while order.size()[0] > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        zz1 = np.maximum(z1[i], z1[order[1:]])
        xx2 = np.maximum(x2[i], x2[order[1:]])
        yy2 = np.maximum(y2[i], y2[order[1:]])
        zz2 = np.maximum(z2[i], z2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        s = np.maximum(0.0, zz2 - zz1 + 1)
        inter = w * h * s
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return torch.IntTensor(keep)


