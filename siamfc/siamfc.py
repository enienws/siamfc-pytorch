from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import cv2
import sys
import os
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from got10k.trackers import Tracker
import pickle
from . import ops
from .backbones import AlexNetV1, AlexNetV1_modified
from .colorization import Colorization
from .resnet import ResNet, BasicBlock, _resnet
from .heads import SiamFC
from .losses import BalancedLoss
from .datasets import Pair
from .transforms import SiamFCTransforms
import copy
import matplotlib.pyplot as plt
from torchvision import transforms
from .transforms import ToPIL
__all__ = ['TrackerSiamFC']


class Net(nn.Module):

    def __init__(self, backbone, head, module=None):
        super(Net, self).__init__()
        self.backbone = backbone
        self.module = module
        self.head = head

        #Weight freezing logic.
        #Freeze the weights of module
        # for child in self.children():
        #     for param in child.parameters():
        #         param.requires_grad = False

    def calc_response(self, input):
        backbone_resp = self.backbone(input)
        module_resp = self.module(input)
        return (backbone_resp, module_resp)

    def forward(self, z, x):
        z = self.backbone(z), self.module(z)
        x = self.backbone(x), self.module(x)
        return self.head(z, x)


class TrackerSiamFC(Tracker):

    def __init__(self, backbone_path=None, module_path=None, alpha=None, **kwargs):
        super(TrackerSiamFC, self).__init__(kwargs['name'], True)
        self.cfg = self.parse_args(**kwargs)

        # Transformers from exemplar and search images
        self.inputTransformer = transforms.Compose([
            ToPIL(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:1' if self.cuda else 'cpu')


        # setup model
        self.net = Net(
            backbone=AlexNetV1(),
            module=_resnet('resnet18', BasicBlock, [2, 2, 2, 2], False, True),
            # module=AlexNetV1_modified(),
            head=SiamFC(self.cfg.out_scale, alpha=alpha))

        for param in self.net.parameters():
            print(param, type(param.data), param.size())

        ops.init_weights(self.net)
        
        # load checkpoint if provided
        # Load models for backbone and model
        # self.load_model(self.net, backbone_path)
        # self.load_model(self.net.module, module_path)
        if backbone_path is not None:
            self.load_model(self.net.backbone, backbone_path, filter_str="backbone.")
        # if backbone_path is not None:
        #     self.net.load_state_dict(torch.load(backbone_path))
        if module_path is not None:
            #use filter = module.resnet. when trained in a colorization framework
            self.load_model(self.net.module, module_path, filter_str = 'module.resnet.')
            # use filter = module. when trained in tracking framework
            # self.load_model(self.net.module, module_path, filter_str='module.')

        self.net = self.net.to(self.device)

        # setup criterion
        self.criterion = BalancedLoss()

        # setup optimizer
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)
        
        # setup lr scheduler
        gamma = np.power(
            self.cfg.ultimate_lr / self.cfg.initial_lr,
            1.0 / self.cfg.epoch_num)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma)

    def load_model(self, net, path, filter_str):
        # Load network
        if path is not None:
            model = torch.load(path)
            if "model_state_dict" in model:
                state_dict = torch.load(path)['model_state_dict']
            else:
                state_dict = torch.load(path)

            state_dict_v2 = copy.deepcopy(state_dict)

            for key in state_dict:
                #filter may be backbone or module
                #if it is module, one tries to load resnet related weights
                if filter_str in key:
                    modified_key = key.replace(filter_str, '')
                    state_dict_v2[modified_key] = state_dict_v2.pop(key)
                else:
                    state_dict_v2.pop(key)

            net.load_state_dict(state_dict_v2)
        return

    def parse_args(self, **kwargs):
        # default parameters
        cfg = {
            # basic parameters
            # 'out_scale': 0.001,
            'out_scale': 1.0,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
            # inference parameters
            'scale_num': 3,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 16,
            'total_stride': 8,
            # train parameters
            'epoch_num': 15,
            'batch_size': 8,
            'num_workers': 32,
            'initial_lr': 1e-2,
            'ultimate_lr': 1e-5,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0}
        
        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)
    
    @torch.no_grad()
    def init(self, img, box):
        # set to evaluation mode
        self.net.eval()

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]
        self.img_sz = (img.shape[0], img.shape[1])

        # create hanning window
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num)

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz
        
        # exemplar image
        self.avg_color = np.mean(img, axis=(0, 1))
        z = ops.crop_and_resize(
            img, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)

        #Transform and normalize the input image
        z = self.inputTransformer(z)

        # exemplar features
        z = z.to(self.device).unsqueeze(0).float()
        self.kernel = self.net.calc_response(z)
    
    @torch.no_grad()
    def update(self, img):
        # set to evaluation mode
        self.net.eval()

        # search images
        x = [ops.crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]

        #Transform and normalize the input image
        x = [self.inputTransformer(image) for image in x]
        x = torch.stack(x, dim=0)

        x = x.to(self.device).float()
        
        # responses
        x = self.net.calc_response(x)
        responses = self.net.head(self.kernel, x)
        responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)
        # print("loc: ", loc)

        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
            self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / self.cfg.instance_sz
        self.center += disp_in_image
        # print("disp:", disp_in_image)
        # print("center:", self.center)
        # update target size
        scale =  (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        #Correct tracking center
        self.correctCenter()

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box

    def correctCenter(self):
        if self.center[0] - self.target_sz[0] / 2 < 0:
            self.center[0] = self.target_sz[0] / 2
        if self.center[0] + self.target_sz[0] / 2 > self.img_sz[0]:
            self.center[0] = self.img_sz[0] - self.target_sz[1] / 2

        if self.center[1] - self.target_sz[1] / 2 < 0:
            self.center[1] = self.target_sz[1] / 2
        if self.center[1] + self.target_sz[1] / 2 > self.img_sz[1]:
            self.center[1] = self.img_sz[1] - self.target_sz[1] / 2

    def track(self, img_files, box, visualize=False):
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            img = ops.read_image(img_file)

            begin = time.time()
            if f == 0:
                self.init(img, box)
            else:
                boxes[f, :] = self.update(img)
            times[f] = time.time() - begin

            if visualize:
                ops.show_image(img, boxes[f, :])

        return boxes, times
    
    def preprocess_step(self, batch):
        # set network mode
        self.net.train(True)

        # parse batch data
        z = batch[0].to(self.device, non_blocking=self.cuda)
        x = batch[1].to(self.device, non_blocking=self.cuda)

        with torch.set_grad_enabled(True):
            # inference
            resp_alex, resp_resnet = self.net(z, x)
        
        return resp_alex, resp_resnet

    @torch.enable_grad()
    def preprocess_over(self, seqs, val_seqs=None,
                   save_dir='pretrained'):
        # set to train mode
        self.net.train(True)

        # create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # setup dataset
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)

        dataset = Pair(
            seqs=seqs,
            transforms=transforms)

        # setup dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            # num_workers=self.cfg.num_workers,
            num_workers=1,
            pin_memory=self.cuda,
            drop_last=True)

        responses_alex_list = []
        responses_resnet_list = []
        # for epoch in range(self.cfg.epoch_num):
        # loop over dataloader
        for it, batch in enumerate(dataloader):
            responses_alex, responses_resnet  = self.preprocess_step(batch)
            responses_alex = responses_alex.cpu().detach().numpy()
            responses_resnet = responses_resnet.cpu().detach().numpy()
            responses_alex_list.append(responses_alex)
            responses_resnet_list.append(responses_resnet)
            torch.cuda.empty_cache()
            print(it)

        def calc_mean_std(features):
            features = np.stack(np.array(features), axis=0)
            features = np.reshape(features, (-1, 1, 15, 15))
            mean = np.mean(features)
            variance = np.std(features)
            return mean, variance

        mean_alex, std_alex = calc_mean_std(responses_alex_list)
        mean_resnet, std_resnet = calc_mean_std(responses_resnet_list)

        with open("/opt/project/alex_v5.pickle", "wb") as f:
            pickle.dump(responses_alex_list, f)
        with open("/opt/project/resnet_v5.pickle", "wb") as f:
            pickle.dump(responses_resnet_list, f)

    def train_step(self, batch, backward=True):
        # set network mode
        self.net.train(backward)

        # parse batch data
        z = batch[0].to(self.device, non_blocking=self.cuda)
        x = batch[1].to(self.device, non_blocking=self.cuda)

        with torch.set_grad_enabled(backward):
            # inference
            responses = self.net(z, x)

            # calculate loss
            labels = self._create_labels(responses.size())
            loss = self.criterion(responses, labels)

            if backward:
                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return loss.item()

    @torch.enable_grad()
    def train_over(self, seqs, val_seqs=None,
                   save_dir='pretrained'):
        # set to train mode
        self.net.train()

        # create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # setup dataset
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)



        dataset = Pair(
            seqs=seqs,
            transforms=transforms)
        
        # setup dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            # num_workers=self.cfg.num_workers,
            num_workers=1,
            pin_memory=self.cuda,
            drop_last=True)
        
        # loop over epochs
        losses = []

        for epoch in range(self.cfg.epoch_num):
            # update lr at each epoch
            self.lr_scheduler.step(epoch=epoch)

            # loop over dataloader
            running_loss = 0
            for it, batch in enumerate(dataloader):
                loss = self.train_step(batch, backward=True)
                print('Epoch: {} [{}/{}] Loss: {:.5f}'.format(
                    epoch + 1, it + 1, len(dataloader), loss))
                sys.stdout.flush()
                running_loss += loss

            #Calculate average loss for this batch
            running_loss /= it
            losses.append(running_loss)
            
            # save checkpoint
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            net_path = os.path.join(
                save_dir, self.name + '_{}.pth'.format(epoch + 1))
            torch.save(self.net.state_dict(), net_path)

        #Save a plot
        plt.clf()
        plt.plot(np.array(losses), 'r')
        plt.savefig(os.path.join(save_dir, self.name + ".png"))
    
    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()
        
        return self.labels


