from __future__ import absolute_import

import os
from got10k.datasets import *

from siamfc import TrackerSiamFC
import pickle

if __name__ == '__main__':
    #Use with preprocess_over
    # backbone_path = "../data/siamfc_alexnet_weighted_e50.pth"
    # module_path = '../data/SiamFCWeighted_Norm_v2_alpha_1_37.pth'

    #module_path -> resnet traiend in colorization framework
    backbone_path = "../data/v2/siamfc_alexnet_weighted_e50.pth"
    module_path = '../data/v2/state73879.pth'

    root_dir = os.path.expanduser('/opt/got10k')
    seqs = GOT10k(root_dir, subset='train', return_meta=True)

    # resp = (1 - alpha) * resp1(siamfc) + alpha * resp2(siamcolor)
    for i in reversed(range(0, 11)):

        # i = 0.1
        i = (10 - i) / 10
        tracker = TrackerSiamFC(name="Baseline_ResNet(Color)_v2_alpha_{}".format(i),
                                backbone_path=backbone_path,
                                module_path=module_path, alpha=i)
        # tracker.preprocess_over(seqs)
        tracker.train_over(seqs)
