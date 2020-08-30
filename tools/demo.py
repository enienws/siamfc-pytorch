from __future__ import absolute_import

import os
import glob
import numpy as np

from siamfc import TrackerSiamFC


if __name__ == '__main__':
    seq_dir = os.path.expanduser('~/data/OTB/Crossing/')
    img_files = sorted(glob.glob(seq_dir + 'img/*.jpg'))
    anno = np.loadtxt(seq_dir + 'groundtruth_rect.txt')
    
    net_path = '..data/siamcolor_resnet_e1.pth'
    tracker = TrackerSiamFC(backbone_path=net_path)
    tracker.track(img_files, anno[0], visualize=True)
