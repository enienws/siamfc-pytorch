import torch.nn as nn
from .resnet import _resnet, BasicBlock
import torch
import torch.optim as optim
import numpy as np
import torchvision

# class BatchNorm3D(nn.Module):
#     """docstring for myBatchNorm3D"""
#
#     def __init__(self, inChannels):
#         super(BatchNorm3D, self).__init__()
#         self.inChannels = inChannels
#         self.bn = nn.BatchNorm2d(self.inChannels)
#
#     def forward(self, input):
#         out = input
#         N, C, D, H, W = out.size()
#         out = out.transpose(1,2).reshape(N*D, C, H, W)
#         out = self.bn(out.contiguous())
#         out = out.reshape(N,D,C,H,W).transpose(1,2)
#         return out



class Colorization(nn.Module):
    def __init__(self):
        super(Colorization, self).__init__()

        self.resnet = _resnet('resnet18', BasicBlock, [2, 2, 2, 2], False, True)

        self.conv3d_1 = self._make_3Dlayer(dilation=1, in_channel=258)
        self.conv3d_2 = self._make_3Dlayer(dilation=2)
        self.conv3d_3 = self._make_3Dlayer(dilation=4)
        self.conv3d_4 = self._make_3Dlayer(dilation=8)
        self.conv3d_5 = self._make_3Dlayer(dilation=16)
        self.features = nn.Conv3d(in_channels=256,
                                out_channels=64,
                                kernel_size=1,
                                dilation=1,
                                stride=1,
                                padding=(0,0,0))
        #TODO :: this is hard-coded cuda:0, should be cpu
        self.device = "cuda:1"

    def setdev(self, device):
        self.device = device
        # self.to(device)

    def create_spatial_features(self, B, W, S=1):
        ft_size = W
        row = []
        increment = 2.0 / float(ft_size - 1)
        for i in range(ft_size):
            row.append(-1 + i * increment)
        row = np.array(row)
        horizontal = np.stack([row] * ft_size)
        vertical = np.transpose(horizontal, [1, 0])
        spatial_feautre = np.stack([horizontal, vertical])
        # Sample spatial is not using since we don't have spatial concept
        # sample_spatial = np.stack([spatial_feautre] * S)
        batch_spatial = np.stack([spatial_feautre] * B)
        spatial_feautre = torch.from_numpy(batch_spatial).type(torch.float32)
        spatial_feautre = spatial_feautre.to(self.device)
        return spatial_feautre

    def _make_3Dlayer(self, in_channel=256, out_channel=256,
                      stride=1, padding=1, kernel_size=3, dilation=2):
        layers = []
        layers.append(nn.Conv3d(in_channels=in_channel,
                                out_channels=out_channel,
                                kernel_size=(1, kernel_size, kernel_size),
                                dilation=(1, dilation, dilation),
                                stride=stride,
                                padding=(0, dilation, dilation)))
        layers.append(nn.BatchNorm3d(out_channel))
        layers.append(nn.ReLU(inplace=True))
        if in_channel != out_channel:
            in_channel = out_channel
        layers.append(nn.Conv3d(in_channels=in_channel,
                                out_channels=out_channel,
                                kernel_size=(kernel_size, 1, 1),
                                dilation=(dilation, 1, 1),
                                stride=1,
                                padding=(dilation,0,0)))
        layers.append(nn.BatchNorm3d(out_channel))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        #Convert to grayscale image
        x = torch.mean(x, keepdim=True, dim=1)

        #number of batches, number of samples, channels, width, height
        N, C, W, H = x.shape

        #Calculate ResNet features
        x = self.resnet.forward(x)

        # number of batches, channels, height, width
        #output of resnet
        #for z: 1, 256, 16, 16
        N, C, W, H = x.shape
        spatial_features = self.create_spatial_features(N, W)
        x = torch.cat((x, spatial_features), dim=1)
        # number of batches, channels, number of samples, height, width
        x = x.unsqueeze(dim=2)

        #3DConv Network
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        x = self.conv3d_4(x)
        x = self.conv3d_5(x)

        # Calculate the features
        # number of batches, channels, number of samples, height, width
        features = self.features(x)
        features = features.squeeze(dim=2)

        return features

    def forward(self, x):
        return self._forward_impl(x)


if __name__ == "__main__":
    # create_spatial_features(16, 4, 32)

    model = Colorization()
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    dummy_input = torch.zeros([16,4,1,256,256], dtype=torch.float32)
    # features = number of batches, channels, number of samples, height, width
    features = model.forward(dummy_input)
    # features = number of batches, number of samples, channels, height * width
    features = features.transpose(1, 2).view(-1, 4, 64, 1024)

    #Create dummy labels
    labels = torch.randint(0, 16, (65536,)).view(16,4,32,32,1).type(torch.int64)

    features_splitted = torch.split(features, [3,1], dim=1)
    # reference_features = number of batches, number of reference, height * width, channels
    reference_features = features_splitted[0].transpose(2,3).view(-1,3072,64)
    # target_features = number of batches, number of target, height * width, channels
    target_features = features_splitted[1].transpose(2,3).view(-1,1024,64)

    labels_splitted = torch.split(labels, [3, 1], dim=1)
    reference_labels = labels_splitted[0].view(-1, 3072)
    target_labels = labels_splitted[1].view(-1, 1024)

    innerproduct = torch.matmul(target_features, reference_features.transpose(1,2))
    similarity = nn.functional.softmax(innerproduct, dim=2)
    dense_reference_labels = torch.nn.functional.one_hot(reference_labels)

    prediction = torch.matmul(similarity, dense_reference_labels.type(torch.float32))

    loss = criterion(prediction.transpose(1,2), target_labels)
    loss.backward()
    optimizer.step()

    a = 0