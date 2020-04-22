from __future__ import print_function
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import numpy as np
import pdb

# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

#array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def generate_anchors_3d(base_size=16, scales=2**np.arange(2, 5)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = np.array([1, 1, 1, base_size, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in xrange(ratio_anchors.shape[0])])
    return anchors

def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[3] - anchor[0] + 1
    h = anchor[4] - anchor[1] + 1
    z = anchor[5] - anchor[2] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    z_ctr = anchor[2] + 0.5 * (z - 1)
    return w, h, z, x_ctr, y_ctr, z_ctr

def _mkanchors(ws, hs, zs, x_ctr, y_ctr, z_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    zs = zs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         z_ctr - 0.5 * (zs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1),
                         z_ctr + 0.5 * (zs - 1)))
    return anchors

def _ratio_enum(anchor):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, z, x_ctr, y_ctr, z_ctr = _whctrs(anchor)
    size = w * h * z
    ratio1 = pow(size,1/3)
    ratio2 = pow(size,1/4)
    ws = np.round([ratio1, ratio2, ratio2, 2*ratio2])
    hs = np.round([ratio1, 2*ratio2, ratio2, ratio2])
    zs = np.round([ratio1, ratio2, 2*ratio2, ratio2])
    anchors = _mkanchors(ws, hs, zs, x_ctr, y_ctr, z_ctr)
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, z, x_ctr, y_ctr, z_ctr= _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    zs = z * scales
    anchors = _mkanchors(ws, hs, zs, x_ctr, y_ctr, z_ctr)
    return anchors

if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_anchors_3d()
    print(time.time() - t)
    print(a)
    from IPython import embed; embed()
