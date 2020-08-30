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
    backbone_path = "../data/siamfc_alexnet_weighted_e50.pth"
    module_path = '../data/state73879.pth'

    root_dir = os.path.expanduser('/opt/got10k')
    seqs = GOT10k(root_dir, subset='train', return_meta=True)

    # resp = (1 - alpha) * resp1(siamfc) + alpha * resp2(siamcolor)
    # for i in (1, 0):

    i = 0.1
    tracker = TrackerSiamFC(name="SiamColor_Norm_alpha_trainable_freeze_none{}".format(i),
                            backbone_path=backbone_path,
                            module_path=module_path, alpha=None)
    # tracker.preprocess_over(seqs)
    tracker.train_over(seqs)
