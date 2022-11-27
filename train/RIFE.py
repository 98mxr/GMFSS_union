import random
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from torch.nn.parallel import DistributedDataParallel as DDP
from gmflow.gmflow import GMFlow
from IFNet_HDv3 import IFNet
from MetricNet import MetricNet
from FusionNet import AnimeInterp
from discriminator import UNetDiscriminatorSN
import torch.nn.functional as F
from loss import *

from lpips import LPIPS

device = torch.device("cuda")
    
class Model:
    def __init__(self, local_rank=-1):
        self.flownet = GMFlow()
        self.ifnet = IFNet()
        self.metricnet = MetricNet()
        self.fusionnet = AnimeInterp()
        self.net_d = UNetDiscriminatorSN()
        self.device()
        # self.optimG = AdamW(self.fusionnet.parameters(), lr=1e-6, weight_decay=1e-4)
        self.optimG = AdamW(itertools.chain(
            self.metricnet.parameters(),
            self.fusionnet.parameters()), lr=1e-6, weight_decay=1e-4)
        self.optimD = AdamW(self.net_d.parameters(), lr=1e-6, weight_decay=1e-4)
        self.l1_loss = Charbonnier_L1().to(device)
        self.lpips = LPIPS(net='vgg').to(device)
        self.gan_loss = GANLoss(gan_type='wgan').to(device)

    def train(self):
        self.flownet.eval()
        self.ifnet.eval()
        self.metricnet.train()
        self.fusionnet.train()
        self.net_d.train()

    def eval(self):
        self.flownet.eval()
        self.ifnet.eval()
        self.metricnet.eval()
        self.fusionnet.eval()
        self.net_d.eval()

    def device(self):
        self.flownet.to(device)
        self.ifnet.to(device)
        self.metricnet.to(device)
        self.fusionnet.to(device)
        self.net_d.to(device)

    def load_model(self, path, rank=-1):
        def convert(param):
                return {
                    k.replace("module.", ""): v
                    for k, v in param.items()
                    if "module." in k
                }

        self.flownet.load_state_dict(torch.load('{}/flownet.pkl'.format(path)))
        self.ifnet.load_state_dict(torch.load('{}/rife.pkl'.format(path)))
        self.metricnet.load_state_dict(torch.load('{}/metric.pkl'.format(path)))
        self.fusionnet.load_state_dict(torch.load('{}/fusionnet.pkl'.format(path)), False)
    
    def save_model(self, path, rank=0):
        # torch.save(self.flownet.state_dict(), f'{path}/flownet.pkl')
        # torch.save(self.ifnet.state_dict(), f'{path}/rife.pkl')
        torch.save(self.metricnet.state_dict(), f'{path}/metric.pkl')
        torch.save(self.fusionnet.state_dict(), f'{path}/fusionnet.pkl')

    def inference(self, img0, img1, timestep=0.5, scale=1.0):
        with torch.no_grad():
            flow01 = self.flownet(img0, img1)
            flow10 = self.flownet(img1, img0)
            flow01 = F.interpolate(flow01, scale_factor = 0.5, mode="bilinear", align_corners=False) * 0.5
            flow10 = F.interpolate(flow10, scale_factor = 0.5, mode="bilinear", align_corners=False) * 0.5

            imgs = torch.cat((img0, img1), 1)
            merged = self.ifnet(imgs, timestep, scale_list=[8, 4, 2, 1])
            merged = F.interpolate(merged, scale_factor = 0.5, mode="bilinear", align_corners=False)

            imgm0 = F.interpolate(img0, scale_factor = 0.5, mode="bilinear", align_corners=False)
            imgm1 = F.interpolate(img1, scale_factor = 0.5, mode="bilinear", align_corners=False)

            # img0_chunks = torch.chunk(img0, chunks=2, dim=0)
            # img1_chunks = torch.chunk(img1, chunks=2, dim=0)
            # flow0_chunks = list()
            # flow1_chunks = list()
            # for s in range(2):
                # flow_gt0 = self.flownet(img0_chunks[s], img1_chunks[s])
                # flow_gt1 = self.flownet(img0_chunks[s], img1_chunks[s])
                # flow0_chunks.append(flow_gt0)
                # flow1_chunks.append(flow_gt1)
            # flow01 = torch.cat(flow0_chunks, dim=0)
            # flow10 = torch.cat(flow1_chunks, dim=0)

        metric0, metric1 = self.metricnet(imgm0, imgm1, flow01, flow10)

        reuse_things = [flow01, flow10, metric0, metric1]

        out, outl1 = self.fusionnet(img0, img1, reuse_things, merged, timestep)
        
        return flow01, flow10, metric0, metric1, out, outl1

    def update(self, imgs, gt, learning_rate=0, training=True, timestep=0.5, step=0, spe=1136):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        for param_group in self.optimD.param_groups:
            param_group['lr'] = learning_rate
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        if training:
            self.train()
        else:
            self.eval()

        accum_iter = 3

        simple_color_aug = SimpleColorAugmentation(enable=True)

        for p in self.net_d.parameters():
            p.requires_grad = False

        img0, img1, gt = simple_color_aug.augment(img0), simple_color_aug.augment(img1), simple_color_aug.augment(gt)
        flow01, flow10, metric0, metric1, merged, mergedl1 = self.inference(img0, img1, timestep, scale=1.0)
        merged, mergedl1, gt = simple_color_aug.reverse_augment(merged), simple_color_aug.reverse_augment(mergedl1), simple_color_aug.reverse_augment(gt)
        
        loss_l1 = self.l1_loss(mergedl1 - gt) 
        loss_lpips = (self.lpips.forward((merged - 0.5) / 0.5, (gt - 0.5) / 0.5).mean((1,2,3))).mean()
        loss_gan = self.gan_loss(self.net_d(merged), True, is_disc=False)

        if training:
            loss_G = (loss_l1 + loss_lpips + loss_gan) / accum_iter
            loss_G.backward()
            if ((step + 1) % accum_iter == 0) or ((step + 1) % spe == 0):
                self.optimG.step()
                self.optimG.zero_grad()

        for p in self.net_d.parameters():
            p.requires_grad = True

        real_d_pred = self.net_d(gt)
        l_d_real = self.gan_loss(real_d_pred, True, is_disc=True)
        if training:
            l_d_real = l_d_real / accum_iter
            l_d_real.backward()

        fake_d_pred = self.net_d(merged.detach().clone())
        l_d_fake = self.gan_loss(fake_d_pred, False, is_disc=True)
        if training:
            l_d_fake = l_d_fake / accum_iter
            l_d_fake.backward()

        if training:
            if ((step + 1) % accum_iter == 0) or ((step + 1) % spe == 0):
                self.optimD.step()
                self.optimD.zero_grad()

        return merged, torch.cat((flow01, flow10), 1), metric0, metric1, loss_l1, loss_lpips, l_d_real, l_d_fake

class SimpleColorAugmentation:
    def __init__(self, enable=True) -> None:
        self.seed = random.uniform(0, 1)
        if self.seed < 0.167:
            self.swap = [2, 1, 0]  # swap 1,3
            self.reverse_swap = [2, 1, 0]
        elif 0.167 < self.seed < 0.333:
            self.swap = [2, 0, 1]
            self.reverse_swap = [1, 2, 0]
        elif 0.333 < self.seed < 0.5:
            self.swap = [1, 2, 0]
            self.reverse_swap = [2, 0, 1]
        elif 0.5 < self.seed < 0.667:
            self.swap = [1, 0, 2]
            self.reverse_swap = [1, 0, 2]
        elif 0.667 < self.seed < 0.833:
            self.swap = [0, 2, 1]
            self.reverse_swap = [0, 2, 1]
        else:
            self.swap = [0, 1, 2]
            self.reverse_swap = [0, 1, 2]
        if not enable:
            self.swap = [0, 1, 2]  # no swap
            self.reverse_swap = self.swap
        pass

    def augment(self, img):
        """
        param: img, torch tensor, CHW
        """
        img = img[:, self.swap, :, :]
        return img

    def reverse_augment(self, img):
        img = img[:, self.reverse_swap, :, :]
        return img