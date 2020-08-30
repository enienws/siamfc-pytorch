from __future__ import absolute_import

import os
from got10k.experiments import *

from siamfc import TrackerSiamFC


if __name__ == '__main__':
    # backbone_path = "../data/v2/siamfc_alexnet_weighted_e50.pth"
    # module_path = None

    #Experiment 2
    backbone_path = "../data/v2/siamfc_alexnet_weighted_e50.pth"
    module_path = "../data/v2/state73879.pth"

    #SiamFC_SiamColor_Weighted (5099)
    #SiamFC_SiamColor_Weighted_57079
    # backbone_path = "../data/siamfc_alexnet_weighted_e50.pth"
    # module_path = '../data/alexnet2d_color_state57079.pth'

    # backbone_path = "../data/siamfc_alexnet_weighted_e50.pth"
    # module_path = '../data/SiamFCWeighted_Norm_v2_alpha_1_37.pth'

    # backbone_path = "../data/SiamColor_Norm_alpha_0.1_50.pth"
    # module_path = None

    # net_path = '../data/siamcolor_resnet2d_weighted_e1.pth'
    # module_path = '../data/siamcolor_resnet_e1.pth'
    # resp = (1 - alpha) * resp1 + alpha * resp2
    trackerNames = []
    for i in range(0,11):
        trackerNum = (10 - i) / 10
        # trackerNum = 0.0
        tracker1 = TrackerSiamFC(name="BaselineTracker".format(trackerNum),
                             backbone_path=backbone_path, module_path=module_path,
                             alpha=trackerNum)
        # tracker2 = TrackerSiamFC(name="SiamColor", backbone_path=net_path, module_path=module_path, alpha=1.0)

        root_dir = os.path.expanduser('/opt/otb/OTB2015')
        e = ExperimentOTB(root_dir, version='tb100')
        # root_dir = os.path.expanduser('/opt/vot/VOT2018')
        # e = ExperimentVOT(root_dir, version=2018, experiments=('supervised','realtime'))
        e.run(tracker1)
        # e.run(tracker2, visualize=False)
        trackerNames.append(tracker1.name)

    e.report(trackerNames)
