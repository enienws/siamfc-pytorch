from siamfc.resnet import _resnet, BasicBlock
from siamfc.backbones import AlexNetV1
import torch
import torch.nn as nn
if __name__ == "__main__":
    net = _resnet('resnet18', BasicBlock, [2, 2, 2, 2], False, True),
    # net = AlexNetV1()
    input = torch.randn((3, 3,255,255), dtype=torch.float32)
    a = net[0](input)
    print(net)
    a = 0