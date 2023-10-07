# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 06:56:35 2023

@author: Administrator

应用迭代方法创建卷积网络

"""

import torch
import torch.nn as nn

from utils import idx2onehot
import numpy as np

import torch.nn.functional as F
from torch.nn import Linear
import torch
import utils as ut
from torch import autograd, nn, optim
from torch import nn
from torch.nn import Linear

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
from torchsummary import summary


################################################################################

def reparameterize(mu, log_var):  # 重参数技巧
    std = torch.exp(0.5 * log_var)  # 分布标准差std
    eps = torch.randn_like(std)  # 从标准正态分布中采样,(n,128)
    return mu + eps * std  # 返回对应正态分布中的采样值


# encoder
class causalNN_model(nn.Module):  # conditional VAE with conv2d
    def __init__(self, batch_size=4, img_length=224, latent_dim=128, z_dim=128, c_dim=18, y_dim=2, channel=1,
                 hiddens=[16, 32, 64, 128, 256]):
        super().__init__()
        self.batch_size = batch_size
        self.img_length = img_length
        self.latent_dim = latent_dim
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.y_dim = y_dim
        # self.label = label
        self.channel = channel
        self.hiddens = hiddens

        # self.a = torch.rand(z.size()[0],z.size()[1])
        self.A = nn.Parameter(torch.rand(batch_size, latent_dim), requires_grad=True).cuda()

        # prev_channels = 3

        # 编码器
        # def cencoder(self,x,y):
        modules_enc = []
        img_length = self.img_length
        prev_channels = self.channel
        for cur_channels in self.hiddens:
            modules_enc.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels,
                              cur_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1), nn.BatchNorm2d(cur_channels),
                    nn.LeakyReLU())
            )
            prev_channels = cur_channels
            img_length //= 2
        self.encoder = nn.Sequential(*modules_enc)
        if torch.cuda.is_available() == True:
            self.encoder = self.encoder.cuda()
        self.mean_linear = nn.Linear(prev_channels * img_length * img_length,
                                     self.latent_dim)
        self.var_linear = nn.Linear(prev_channels * img_length * img_length,
                                    self.latent_dim)
        # self.latent_dim = latent_dim

        self.mu = self.mean_linear
        self.log_var = self.var_linear
        self.liner_ez = nn.Linear(prev_channels * img_length * img_length,
                                     self.c_dim)
        # self.liner_ez = nn.Linear(prev_channels * img_length * img_length,
        #                              self.latent_dim)
        self.liner_cy = nn.Linear(self.c_dim, self.y_dim)
        # self.liner_cy = nn.Linear(self.z_dim, self.y_dim)



    def masked_sep(self, z):
        # z = z.view(self.batch_size, -1, self.z_dim)
        z = z.view(-1, self.z_dim)
        z = self.net(z)
        return z

    def loss_fn(self, y_pred, y):
        loss_fn = nn.MSELoss()
        # loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(y_pred, y)

        return loss

    def forward(self, x):
        # A = self.A
        attn = Attention(self.latent_dim, self.z_dim)
        # y = y.cuda()
        x = x.cuda()
        encoder = self.encoder(x)
        e = torch.flatten(encoder, 1)
        # mean = self.mean_linear(encoder)
        # logvar = self.var_linear(encoder) + 1e-8
        # eps = torch.rand_like(logvar)
        # std = torch.exp(logvar / 2)
        # e = eps * std + mean
        z = self.liner_ez(e)
        # c = attn.attention(z)
        # z_c = self.masked_sep(z)
        y_pred = self.liner_cy(z)

        return y_pred


    # def forward(self, x):
    #     # A = self.A
    #     attn = Attention(self.latent_dim, self.z_dim)
    #     # y = y.cuda()
    #     x = x.cuda()
    #     encoder = self.encoder(x)
    #     e = torch.flatten(encoder, 1)
    #     # mean = self.mean_linear(encoder)
    #     # logvar = self.var_linear(encoder) + 1e-8
    #     # eps = torch.rand_like(logvar)
    #     # std = torch.exp(logvar / 2)
    #     # e = eps * std + mean
    #     z = self.liner_ez(e)
    #     c = attn.attention(z)
    #     # z_c = self.masked_sep(z)
    #     y_pred = self.liner_cy(c)
    #
    #     return y_pred

# def loss_fn(self,x, x_rec, mean, logvar):
# 	kl_weight = 0.00025
# 	recons_loss = F.mse_loss(x_rec, x)
# 	kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar), 1), 0)
# 	loss = recons_loss + kl_loss * kl_weight
# 	return loss

# self.z = reparameterize(self.mu, self.log_var)
# self.z_c = self.dag(self.z)


def sample(self, device='cuda'):
    z = torch.randn(1, self.latent_dim).to(device)
    x = self.decoder_projection(z)
    x = torch.reshape(x, (-1, *self.decoder_input_chw))
    decoded = self.decoder(x)
    return decoded


# 分类模型
class model_classify(nn.Module):  # conditional VAE with conv2d
    def __init__(self, batch_size=4, img_length=224, y_dim=2, channel=1, hiddens=[16, 32, 64, 128, 256],
                 latent_dim=128):
        super().__init__()
        # self.x = x
        self.img_length = img_length
        self.y_dim = y_dim
        # self.label = label
        self.channel = channel
        self.hiddens = hiddens
        self.latent_dim = latent_dim
        # self.a = torch.rand(z.size()[0],z.size()[1])
        self.A = nn.Parameter(torch.rand(batch_size, latent_dim), requires_grad=True).cuda()

        # prev_channels = 3

        # 编码器
        # def cencoder(self,x,y):
        modules_enc = []
        img_length = self.img_length
        prev_channels = self.channel
        for cur_channels in self.hiddens:
            modules_enc.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels,
                              cur_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1), nn.BatchNorm2d(cur_channels),
                    nn.LeakyReLU())
            )
            prev_channels = cur_channels
            img_length //= 2
        self.encoder = nn.Sequential(*modules_enc)
        if torch.cuda.is_available() == True:
            self.encoder = self.encoder.cuda()
        self.mean_linear = nn.Linear(prev_channels * img_length * img_length,
                                     latent_dim)
        self.var_linear = nn.Linear(prev_channels * img_length * img_length,
                                    latent_dim)
        self.latent_dim = latent_dim

        self.mu = self.mean_linear
        self.log_var = self.var_linear


class Attention(nn.Module):
    def __init__(self, in_features, out_features, bias=False):  # in_features z2_dim
        super().__init__()
        self.M = nn.Parameter(torch.nn.init.normal_(torch.zeros(in_features, out_features), mean=0, std=1)).cuda()
        self.sigmd = torch.nn.Sigmoid()

    # self.M =  nn.Parameter(torch.zeros(in_features,in_features))
    # self.A = torch.zeros(in_features,in_features).to(device)

    def attention(self, z):
        # z: decode_m [64, z1_dim, z2_dim]
        # e: q_m [64, z1_dim, z2_dim]
        # M: [z1_dim, z1_dim]
        # breakpoint()
        # a = z.matmul(self.M).matmul(e.permute(0, 2, 1))
        c = z.matmul(self.M)
        c = self.sigmd(c)
        c = torch.softmax(c, dim=1)
        # print(self.M)
        # A = torch.softmax(c, dim=1)
        # e = torch.matmul(A, e)
        # zzz=0
        # return e, A
        return c


class Attention_ori(nn.Module):
    def __init__(self, in_features, bias=False):  # in_features z2_dim
        super().__init__()
        self.M = nn.Parameter(torch.nn.init.normal_(torch.zeros(in_features, in_features), mean=0, std=1))
        self.sigmd = torch.nn.Sigmoid()

    # self.M =  nn.Parameter(torch.zeros(in_features,in_features))
    # self.A = torch.zeros(in_features,in_features).to(device)

    def attention(self, z, e):
        # z: decode_m [64, z1_dim, z2_dim]
        # e: q_m [64, z1_dim, z2_dim]
        # M: [z1_dim, z1_dim]
        # breakpoint()
        a = z.matmul(self.M).matmul(e.permute(0, 2, 1))
        a = self.sigmd(a)
        # print(self.M)
        A = torch.softmax(a, dim=1)
        e = torch.matmul(A, e)
        return e, A
