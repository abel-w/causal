# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software;
# you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.
"""
纯VAE重建 重建图片比causalVAE好
没有y参与训练
"""

import torch

import myModels
from codebase import utils as ut
import argparse
from pprint import pprint

cuda = torch.cuda.is_available()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from torch.autograd import Variable
import numpy as np
import math
import time
from torch.utils import data
from utils import get_batch_unin_dataset_withlabel, _h_A
import matplotlib.pyplot as plt
import random
import torch.utils.data as Data
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from codebase import utils as ut
# from codebase.models.mask_vae_pendulum import CausalVAE
from myModels_VAE import cvae
import argparse
# from tqdm import tqdm
from pprint import pprint

cuda = torch.cuda.is_available()
from torchvision.utils import save_image
import myUtils as mut
import myModels as mms

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
import netron
import torch.onnx


###################################################################

class causalAE_model(nn.Module):  # conditional VAE with conv2d
    def __init__(self, batch_size=4, img_length=224, y_dim=2, channel=1, hiddens=[16, 32, 64, 128, 256],
                 latent_dim=128):
        super().__init__()
        # self.x = x
        self.img_length = img_length
        self.z_dim = 12544
        self.y_dim = y_dim
        # self.label = label
        self.channel = channel
        self.hiddens = hiddens
        self.latent_dim = latent_dim
        # self.a = torch.rand(z.size()[0],z.size()[1])
        self.A = nn.Parameter(torch.rand(batch_size, latent_dim), requires_grad=True).cuda()
        self.linerez = nn.Linear(12544, 2000)

        # self.prev_channels = 3

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

        self.Liner_condition = nn.Linear(prev_channels * img_length * img_length + self.y_dim,
                                         prev_channels * img_length * img_length)

        # summary(self.encoder,(1,224,224))

        # self.z = reparameterize(self.mu, self.log_var)

        # return self.m, self.v, self.z

        # decoder
        # def cdecoder(self):
        modules_dec = []
        self.decoder_projection = nn.Linear(prev_channels * img_length * img_length + self.y_dim,
                                            prev_channels * img_length * img_length)
        self.liner_encoder_x = nn.Linear(prev_channels * img_length * img_length,
                                            prev_channels * img_length * img_length)
        # self.decoder_projection = nn.Linear(
        # 	latent_dim + self.y_dim, prev_channels * img_length * img_length)

        self.decoder_input_chw = (prev_channels, img_length, img_length)
        for i in range(len(hiddens) - 1, 0, -1):
            modules_dec.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hiddens[i],
                                       hiddens[i - 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hiddens[i - 1]), nn.LeakyReLU())
            )
        modules_dec.append(
            nn.Sequential(
                nn.ConvTranspose2d(hiddens[0],
                                   hiddens[0],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(hiddens[0]), nn.LeakyReLU(),
                nn.Conv2d(hiddens[0], self.channel, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU()))
        self.decoder = nn.Sequential(*modules_dec)
        if torch.cuda.is_available() == True:
            self.decoder = self.decoder.cuda()

        self.net = nn.Sequential(
            nn.Linear(self.z_dim, 10000),
            nn.ELU(),
            nn.Linear(10000, self.z_dim),
        )

    def masked(self, z):
        z = z.view(-1, self.z_dim)
        z = self.net(z)
        return z

    # summary(self.decoder,(256,7,7))
    # ttt=0

    # def reparameterize(self, mu, log_var):  # 重参数技巧
    # 	std = torch.exp(0.5 * log_var)  # 分布标准差std
    # 	eps = torch.randn_like(std)  # 从标准正态分布中采样,(n,128)
    # 	return mu + eps * std  # 返回对应正态分布中的采样值

    def dag(self, z):
        # z_in = z
        # a = torch.rand(z.size()[0],z.size()[1])
        # A = nn.Parameter(torch.rand(z.size()[0],z.size()[1]), requires_grad=True)
        # bias = nn.Parameter(torch.Tensor(z_in.size()[0]).cuda())
        # z_c = F.linear(z_in, A, bias)
        z_c = torch.mul(z, self.A)
        return z_c, self.A

    # def forward(self, x, y):
    #     y = y.cuda()
    #     x = x.cuda().float()
    #     encoder = self.encoder(x)
    #     encoder = torch.flatten(encoder, 1)
    #     # encoder = self.Liner_condition(torch.cat([encoder, y], dim=1))
    #
    #     # mean = self.mean_linear(encoder)
    #     # logvar = self.var_linear(encoder) + 1e-8
    #     # eps = torch.rand_like(logvar)
    #     # std = torch.exp(logvar / 2)
    #     # z = eps * std + mean
    #     # z_c = z
    #
    #     # z_c, A = self.dag(z)
    #
    #     # z_c, self.A = torch.mul(z,self.A)
    #     x = self.decoder_projection(torch.cat((encoder, y), 1))
    #     x = torch.reshape(x, (-1, *self.decoder_input_chw))
    #     decoded = self.decoder(x)
    #
    #     return decoded

    # return decoded, mean, logvar, z_c, self.A

    def forward(self, x):
        # y = y.cuda()
        x = x.cuda().float()
        encoder = self.encoder(x)
        encoder = torch.flatten(encoder, 1)
        # encoder = self.Liner_condition(torch.cat([encoder, y], dim=1))

        # mean = self.mean_linear(encoder)
        # logvar = self.var_linear(encoder) + 1e-8
        # eps = torch.rand_like(logvar)
        # std = torch.exp(logvar / 2)
        # z = eps * std + mean
        # z_c = z

        # z_c, A = self.dag(z)

        # z_c, self.A = torch.mul(z,self.A)
        # e = self.decoder_projection(torch.cat((encoder, y), 1))
        e = self.liner_encoder_x(encoder)
        z_e = self.masked(e)
        # z_e = self.linerez(e)
        # z_m = self.masked(z_e)

        x = torch.reshape(z_e, (-1, *self.decoder_input_chw))
        decoded = self.decoder(x)

        return decoded

    def loss_fn(self, x, x_rec):
        # kl_weight = 0.00025
        recons_loss = F.mse_loss(x_rec, x)
        # kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean ** 2 - torch.exp(logvar), 1), 0)
        loss = recons_loss.float()
        # loss = recons_loss + kl_loss * kl_weight
        # loss = loss.float()
        return loss

##############################################################################################################
if __name__ == "__main__":
    path_modelviz = './modelviz.pth'

    # parameter######################
    nm_ds = 'breast'
    num_samples = 20000
    batch_size_su = 32
    total_epochs_su = 20
    channel_su = 3
    hiddens_su = [16, 32, 64, 128, 256]
    ###################
    path_ds = r"D:\pypro\data\breast"
    # path_out = './output'
    # if not os.path.exists(path_out):
    #     os.makedirs(path_out)

    path_result = os.path.join('./output/outputviz/causalAE/')
    if not os.path.exists(path_result):
        os.makedirs(path_result)

    path_savemodel = './output/model_save'
    if not os.path.exists(path_savemodel):
        os.makedirs(path_savemodel)

    # path_train = dpath
    # path_ds = 'D:/pypro/data/%s_enhance/test' % nm_ds
    train_be, var_be, test_be, train_ma, var_ma, test_ma = mut.datasetAll(path_ds)
    dataset_train = mut.getData_breast(train_be, train_ma, number_samples=num_samples)
    train_dataset = DataLoader(dataset=dataset_train, batch_size=batch_size_su, drop_last=True, shuffle=True)

    dataset_var = mut.getData_breast(var_be, var_ma, number_samples=1000)
    var_dataset = DataLoader(dataset=dataset_var, batch_size=batch_size_su, drop_last=True, shuffle=True)

    # dataset_val = mut.getData('D:/pypro/data/%s_enhance/var' % nm_ds, number_samples=1000)
    # val_data = DataLoader(
    #     dataset=dataset_val, batch_size=batch_size_su, drop_last=True, shuffle=True)
    dataset_test = mut.getData_breast(train_be, train_ma, number_samples=1000)
    test_dataset = DataLoader(dataset=dataset_test, batch_size=batch_size_su, drop_last=True, shuffle=True)

    if 'model_causalAE_breast.pt' in os.listdir(os.path.join(path_savemodel,'model_causalAE')):
        causalAE = torch.load(os.path.join(os.path.join(path_savemodel,'model_causalAE', 'model_causalAE_breast.pt')))
    else:
        causalAE = causalAE_model(batch_size=batch_size_su, img_length=224, channel=channel_su, hiddens=hiddens_su,
                      latent_dim=128).to(device)
    optimizer = torch.optim.Adam(causalAE.parameters(), lr=1e-3)

    # def save_model_by_name(model, global_step):
    # 	save_dir = os.path.join('checkpoints', model.name)
    # 	if not os.path.exists(save_dir):
    # 		os.makedirs(save_dir)
    # 	file_path = os.path.join(save_dir, 'model-{:05d}.pt'.format(global_step))
    # 	state = model.state_dict()
    # 	torch.save(state, file_path)
    # 	print('Saved to {}'.format(file_path))

    # 显示网络结构
    # x_onnx = torch.rand([2,1,224,224])
    # y_onnx = torch.rand([2,2])
    # torch.onnx.export(causalVAE_model, (x_onnx, y_onnx), path_modelviz)
    # netron.start(path_modelviz)
    list_loss_var = [100]
    for epoch in range(total_epochs_su):
        causalAE.train()
        totsl_loss = 0
        total_rec = 0
        total_kl = 0
        # n = 0
        # 设置当前epoch显示进度
        # pbar = tqdm(total=len(dataset), desc=f"Epoch {epoch + 1}/{total_epochs}", postfix=dict, miniters=0.3)
        # for x, y in train_dataset:
        # a_p = torch.ones(8, 128)

        for i, (x, y) in enumerate(train_dataset):  # 循环iter
            optimizer.zero_grad()
            x = x.float().to(device)
            x, max_x, min_x = mut.norm01(x)
            x_rec = causalAE.forward(x)

            loss = causalAE.loss_fn(x, x_rec)
            # loss = AE_model.loss_fn(x, x_rec, mean, logvar)
            # optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totsl_loss += loss
            loss0 = loss.item()

            print('causalAE_%s_epoch-%d / %d - iter - %d / %d: %f:' % (
                nm_ds, epoch + 1, total_epochs_su, i + 1, len(dataset_train) // batch_size_su, loss0))

            # if i % 100 == 0 or i == len(dataset) // batch_size_su - 1:
            if i == len(dataset_train) // batch_size_su - 1 or i % 300 == 0:
                x_play = x_rec[0, :, :, :].cpu()
                x_play = x_play.detach().cpu()
                x_play = x_play.cpu().detach().numpy()
                x_play = np.transpose(x_play, [1, 2, 0])
                x_play = (x_play - np.min(x_play)) / (np.max(x_play) - np.min(x_play))
                im = Image.fromarray(np.uint8(x_play * 255))
                im.save(os.path.join(path_result, '%02d%04d.png' % (epoch + 1, i)))


        loss_val = mut.AEeval(model=causalAE, data=var_dataset)

        if round(loss_val, 5) <= round(min(list_loss_var), 5):
            # torch.save(AE, os.path.join(path_savemodel, 'model_AE_%s_temp.pt' %nm_ds))
            # torch.save(AE.state_dict(), os.path.join(path_savemodel, 'model_AE_dic_%s_temp.pt' %nm_ds))
            model_best = causalAE
        list_loss_var.append(loss_val)

        #     pbar.set_postfix(**{"Loss": loss.item()})  # 显示当前iter的loss
        #     pbar.update(1)  # 步进长度
        # pbar.close()  # 关闭当前epoch显示进度

    t_f = mut.timeNow()
    # path_model_save = os.path.join('./output/model_save/VAE', 'VAE_model_%s.pt'%nm_ds)
    # if not os.path.exists('./output/model_save/VAE'):
    #     os.makedirs('./output/model_save/VAE')
    torch.save(model_best, os.path.join(path_savemodel, 'model_causalAE_%s_%s.pt' % (nm_ds, t_f)))
    torch.save(model_best.state_dict(), os.path.join(path_savemodel, 'model_dict_causalAE_%s_%s.pt' % (nm_ds, t_f)))

    print('......run is ok......')
