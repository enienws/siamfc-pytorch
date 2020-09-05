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

    #Experiment3
    backbone_path_list = [
        # "../data/v2/Baseline_ResNet(Color)_alpha_1.0_15.pth",
        # "../data/v2/Baseline_ResNet(Color)_alpha_0.9_15.pth",
        # "../data/v2/Baseline_ResNet(Color)_alpha_0.8_15.pth",
        # "../data/v2/Baseline_ResNet(Color)_alpha_0.7_15.pth",
        # "../data/v2/Baseline_ResNet(Color)_alpha_0.6_15.pth",
        # "../data/v2/Baseline_ResNet(Color)_alpha_0.5_15.pth",
        # "../data/v2/Baseline_ResNet(Color)_alpha_0.4_15.pth",
        # "../data/v2/Baseline_ResNet(Color)_alpha_0.3_15.pth",
        # "../data/v2/Baseline_ResNet(Color)_alpha_0.2_15.pth",
        "../data/v2/Baseline_ResNet(Color)_v2_alpha_0.1_15.pth",
        "../data/v2/Baseline_ResNet(Color)_v2_alpha_0.0_15.pth"
        ]

    module_path = None

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
    for i in range(0,2):
        trackerNum = (10 - i) / 10
        backbone_path = backbone_path_list[i]
        # trackerNum = 0.0
        tracker1 = TrackerSiamFC(name="Baseline_ResNet(Color)_v2_Trained_alpha_{}".format(trackerNum),
                             backbone_path=backbone_path, module_path=module_path,
                             alpha=trackerNum)
        # tracker2 = TrackerSiamFC(name="SiamColor", backbone_path=net_path, module_path=module_path, alpha=1.0)

        root_dir = os.path.expanduser('/opt/otb/OTB2015')
        e = ExperimentOTB(root_dir, version='tb50')
        # root_dir = os.path.expanduser('/opt/vot/VOT2018')
        # e = ExperimentVOT(root_dir, version=2018, experiments=('supervised','realtime'))
        e.run(tracker1)
        # e.run(tracker2, visualize=False)
        trackerNames.append(tracker1.name)

    e.report(trackerNames)
