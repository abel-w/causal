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
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
from torchsummary import summary

################################################################################

def reparameterize(mu,log_var):  # 重参数技巧
	std = torch.exp(0.5 * log_var)  # 分布标准差std
	eps = torch.randn_like(std)  # 从标准正态分布中采样,(n,128)
	return mu + eps * std  # 返回对应正态分布中的采样值

# encoder
class cvae(nn.Module):#conditional VAE with conv2d
	def __init__(self, batch_size=4, img_length=224, y_dim=2, channel=1, hiddens=[16, 32, 64, 128, 256], latent_dim=128):
		super().__init__()
		# self.x = x
		self.img_length = img_length
		self.y_dim = y_dim
		# self.label = label
		self.channel = channel
		self.hiddens = hiddens
		self.latent_dim =latent_dim
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
		self.log_var= self.var_linear

		# summary(self.encoder,(1,224,224))
		# ttt = 0

		# self.z = reparameterize(self.mu, self.log_var)

		# return self.m, self.v, self.z

	# decoder
	# def cdecoder(self):
		modules_dec = []
		self.decoder_projection = nn.Linear(
			latent_dim + self.y_dim, prev_channels * img_length * img_length)
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

		# summary(self.decoder,(256,7,7))
		# ttt=0

	
	# def reparameterize(self, mu, log_var):  # 重参数技巧
	# 	std = torch.exp(0.5 * log_var)  # 分布标准差std
	# 	eps = torch.randn_like(std)  # 从标准正态分布中采样,(n,128)
	# 	return mu + eps * std  # 返回对应正态分布中的采样值

	def dag(self,z):
		# z_in = z
		# a = torch.rand(z.size()[0],z.size()[1])
		# A = nn.Parameter(torch.rand(z.size()[0],z.size()[1]), requires_grad=True)
		# bias = nn.Parameter(torch.Tensor(z_in.size()[0]).cuda())
		# z_c = F.linear(z_in, A, bias)
		z_c = torch.mul(z, self.A)
		return z_c, self.A
		
	
	def forward(self,x,y):
		y = y.cuda()
		x= x.cuda()
		encoder = self.encoder(x)
		encoder = torch.flatten(encoder,1)
		mean = self.mean_linear(encoder)
		logvar = self.var_linear(encoder) + 1e-8
		eps = torch.rand_like(logvar)
		std = torch.exp(logvar/2) 
		z = eps * std + mean
		z_c = z
		# z_c, A = self.dag(z)
        
		# z_c, self.A = torch.mul(z,self.A)
		x = self.decoder_projection(torch.cat((z_c,y),1))
		x = torch.reshape(x,(-1, *self.decoder_input_chw))
		decoded = self.decoder(x)

		return decoded, mean, logvar, z_c, self.A
	
	def loss_fn(self,x, x_rec, mean, logvar):
		kl_weight = 0.00025
		recons_loss = F.mse_loss(x_rec, x)
		kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar), 1), 0)
		loss = recons_loss + kl_loss * kl_weight
		return loss



		# self.z = reparameterize(self.mu, self.log_var)
		# self.z_c = self.dag(self.z)

def sample(self, device='cuda'):
    z = torch.randn(1, self.latent_dim).to(device)
    x = self.decoder_projection(z)
    x = torch.reshape(x, (-1, *self.decoder_input_chw))
    decoded = self.decoder(x)
    return decoded




# 分类模型
class model_classify(nn.Module):#conditional VAE with conv2d
	def __init__(self, batch_size=4, img_length=224, y_dim=2, channel=1, hiddens=[16, 32, 64, 128, 256], latent_dim=128):
		super().__init__()
		# self.x = x
		self.img_length = img_length
		self.y_dim = y_dim
		# self.label = label
		self.channel = channel
		self.hiddens = hiddens
		self.latent_dim =latent_dim
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
		self.log_var= self.var_linear

# Hyperparameters
# n_epochs = 10

# lr = 0.005
# def loss_fn(y, y_hat, mean, logvar):
# 	kl_weight = 0.00025
# 	recons_loss = F.mse_loss(y_hat, y)
# 	kl_loss = torch.mean(
#         -0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar), 1), 0)
# 	loss = recons_loss + kl_loss * kl_weight
# 	return loss


# class ConvEncoder(nn.Module):
# 	def __init__(self, out_dim=None):
# 		super().__init__()
# 		# init 96*96
# 		self.conv1 = torch.nn.Conv2d(3, 32, 4, 2, 1) # 48*48
# 		self.conv2 = torch.nn.Conv2d(32, 64, 4, 2, 1, bias=False) # 24*24
# 		self.conv3 = torch.nn.Conv2d(64, 1, 4, 2, 1, bias=False)
# 		#self.conv4 = torch.nn.Conv2d(128, 1, 1, 1, 0) # 54*44
   
# 		self.LReLU = torch.nn.LeakyReLU(0.2, inplace=True)
# 		self.convm = torch.nn.Conv2d(1, 1, 4, 2, 1)
# 		self.convv = torch.nn.Conv2d(1, 1, 4, 2, 1)
# 		self.mean_layer = nn.Sequential(
# 			torch.nn.Linear(8*8, 16)
# 			) # 12*12
# 		self.var_layer = nn.Sequential(
# 			torch.nn.Linear(8*8, 16)
# 			)
# 		# self.fc1 = torch.nn.Linear(6*6*128, 512)
# 		self.conv6 = nn.Sequential(
# 			nn.Conv2d(3, 32, 4, 2, 1),
# 			nn.ReLU(True),
# 			nn.Conv2d(32, 32, 4, 2, 1),
# 			nn.ReLU(True),
# 			nn.Conv2d(32, 64, 4, 2, 1),
# 			nn.ReLU(True),
# 			nn.Conv2d(64, 64, 4, 2, 1),
# 			nn.ReLU(True),
# 			nn.Conv2d(64, 64, 4, 2, 1),
# 			nn.ReLU(True),
# 			nn.Conv2d(64, 256, 4, 1),
# 			nn.ReLU(True),
# 			nn.Conv2d(256,128 , 1)
# 		)

# 	def encode(self, x):
# 		x = self.LReLU(self.conv1(x))
# 		x = self.LReLU(self.conv2(x))
# 		x = self.LReLU(self.conv3(x))
# 		#x = self.LReLU(self.conv4(x))
# 		#print(x.size())
# 		hm = self.convm(x)
# 		#print(hm.size())
# 		hm = hm.view(-1, 8*8)
# 		hv = self.convv(x)
# 		hv = hv.view(-1, 8*8)
# 		mu, var = self.mean_layer(hm), self.var_layer(hv)
# 		var = F.softplus(var) + 1e-8
# 		#var = torch.reshape(var, [-1, 16, 16])
# 		#print(mu.size())
# 		return  mu, var
# 	def encode_simple(self,x):
# 		x = self.conv6(x)
# 		m,v = ut.gaussian_parameters(x, dim=1)
# 		#print(m.size())
# 		return m,v
    
# class Decoder(nn.Module):
#     def __init__(self, z_dim, concept, z1_dim, channel, y_dim=1):
#         super().__init__()
#         self.z_dim = z_dim
#         self.z1_dim = z1_dim
#         self.concept = concept
#         self.y_dim = y_dim
#         self.channel = channel
#         self.elu = nn.ELU()
#         self.net1 = nn.Sequential(
#             nn.Linear(z1_dim + y_dim,300),
#             nn.ELU(),
#             nn.Linear(300, 300),
#             nn.ELU(),
#             nn.Linear(300, 1024),
#             nn.ELU(),
#             nn.Linear(1024, self.channel*224*224)
#             )
#         self.net2 = nn.Sequential(
#             nn.Linear(z1_dim + y_dim, 300),
#             nn.ELU(),
#             nn.Linear(300, 1024),
#             nn.ELU(),
#             nn.Linear(1024, self.channel*224*224)
#             )
#         self.net3 = nn.Sequential(
# 			nn.Linear(z1_dim + y_dim, 300),
# 			nn.ELU(),
# 			nn.Linear(300, 300),
# 			nn.ELU(),
# 			nn.Linear(300, 1024),
# 			nn.ELU(),
# 			nn.Linear(1024, self.channel*224*224)
#             )
#         self.net4 = nn.Sequential(
# 			nn.Linear(z1_dim + y_dim, 300),
# 			nn.ELU(),
# 			nn.Linear(300, 300),
# 			nn.ELU(),
# 			nn.Linear(300, 1024),
# 			nn.ELU(),
# 			nn.Linear(1024, self.channel*224*224)
#             )
#         self.net5 = nn.Sequential(
# 			nn.ELU(),
# 			nn.Linear(1024, self.channel*224*224)
#             )
#         self.net6 = nn.Sequential(
# 			nn.Linear(z_dim, 300),
# 			nn.ELU(),
# 			nn.Linear(300, 300),
# 			nn.ELU(),
# 			nn.Linear(300, 1024),
# 			nn.ELU(),
# 			nn.Linear(1024, 1024),
# 			nn.ELU(),
# 			nn.Linear(1024, self.channel*224*224)
#             )
        
#     def decoder_condition(self, z, u):
# 		#z = z.view(-1,3*4)
#         z = z.view(-1, 3*4)
#         z1, z2, z3 = torch.split(z, self.z_dim//4, dim = 1)
# 		#print(u[:,0].reshape(1,u.size()[0]).size())
#         rx1 = self.net1(torch.transpose(torch.cat((torch.transpose(z1, 1,0), u[:,0].reshape(1,u.size()[0])), dim = 0), 1, 0))
#         rx2 = self.net2(torch.transpose(torch.cat((torch.transpose(z2, 1,0), u[:,1].reshape(1,u.size()[0])), dim = 0), 1, 0))
#         rx3 = self.net3(torch.transpose(torch.cat((torch.transpose(z3, 1,0), u[:,2].reshape(1,u.size()[0])), dim = 0), 1, 0))
   
#         h = self.net4( torch.cat((rx1,rx2, rx3), dim=1))
        
#         return h


class ae(nn.Module):  # conditional VAE with conv2d
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

		self.Liner_condition = nn.Linear(prev_channels * img_length * img_length+self.y_dim,prev_channels * img_length * img_length)

		# summary(self.encoder,(1,224,224))

		# self.z = reparameterize(self.mu, self.log_var)

		# return self.m, self.v, self.z

		# decoder
		# def cdecoder(self):
		modules_dec = []
		self.decoder_projection = nn.Linear(prev_channels*img_length*img_length+self.y_dim,
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

	def forward(self, x, y):
		y = y.cuda()
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
		x = self.decoder_projection(torch.cat((encoder, y), 1))
		x = torch.reshape(x, (-1, *self.decoder_input_chw))
		decoded = self.decoder(x)

		return decoded
		# return decoded, mean, logvar, z_c, self.A

	def loss_fn(self, x, x_rec):
		# kl_weight = 0.00025
		recons_loss = F.mse_loss(x_rec, x)
		# kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean ** 2 - torch.exp(logvar), 1), 0)
		loss = recons_loss.float()
		# loss = recons_loss + kl_loss * kl_weight
		# loss = loss.float()
		return loss

	# def loss_fn(self, x, x_rec, mean, logvar):
	# 	kl_weight = 0.00025
	# 	recons_loss = F.mse_loss(x_rec, x)
	# 	kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean ** 2 - torch.exp(logvar), 1), 0)
	# 	loss = recons_loss + kl_loss * kl_weight
	# 	return loss



class causalAE(nn.Module):  # conditional VAE with conv2d
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
		self.z_dim = 12544

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

		self.Liner_condition = nn.Linear(prev_channels * img_length * img_length+self.y_dim,prev_channels * img_length * img_length)
		self.liner = nn.Linear(self.z_dim, self.z_dim)
		self.linerxy = nn.Linear(self.z_dim + self.y_dim,self.z_dim)
		# summary(self.encoder,(1,224,224))

		# self.z = reparameterize(self.mu, self.log_var)

		# return self.m, self.v, self.z

		# decoder
		# def cdecoder(self):
		modules_dec = []
		self.decoder_projection = nn.Linear(prev_channels*img_length*img_length+self.y_dim,
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
			nn.Linear(self.z_dim , 1024),
			nn.ELU(),
			nn.Linear(1024, self.z_dim),
		)


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

	def masked(self, z):
		z = z.view(-1, self.z_dim)
		z = self.net(z)
		return z

	def forward(self, x, y):
		y = y.cuda()
		x = x.cuda().float()
		encoder = self.encoder(x)
		encoder = torch.flatten(encoder, 1)
		z = self.masked(encoder)
		# x = self.liner(z)
		# z = self.linerxy(torch.cat([z,y],dim=1))

		# encoder = self.Liner_condition(torch.cat([encoder, y], dim=1))

		# mean = self.mean_linear(encoder)
		# logvar = self.var_linear(encoder) + 1e-8
		# eps = torch.rand_like(logvar)
		# std = torch.exp(logvar / 2)
		# z = eps * std + mean
		# z_c = z

		# z_c, A = self.dag(z)

		# z_c, self.A = torch.mul(z,self.A)
		z = self.decoder_projection(torch.cat((z, y), 1))
		z = torch.reshape(z, (-1, *self.decoder_input_chw))
		decoded = self.decoder(z)

		return decoded
		# return decoded, mean, logvar, z_c, self.A

	def loss_fn(self, x, x_rec):
		# kl_weight = 0.00025
		recons_loss = F.mse_loss(x_rec, x)
		# kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean ** 2 - torch.exp(logvar), 1), 0)
		loss = recons_loss.float()
		# loss = recons_loss + kl_loss * kl_weight
		# loss = loss.float()
		return loss

	# def loss_fn(self, x, x_rec, mean, logvar):
	# 	kl_weight = 0.00025
	# 	recons_loss = F.mse_loss(x_rec, x)
	# 	kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean ** 2 - torch.exp(logvar), 1), 0)
	# 	loss = recons_loss + kl_loss * kl_weight
	# 	return loss



class causalNN(nn.Module):  # conditional VAE with conv2d
	def __init__(self,model_in):
	# def __init__(self, batch_size=4, img_length=224, y_dim=2, channel=1, hiddens=[16, 32, 64, 128, 256],
	# 			 latent_dim=128):
		super().__init__()
		self.model_in = model_in
		self.z_dim = 2048
		# self.x = x
		# self.img_length = img_length
		# self.y_dim = y_dim
		# self.label = label
		# self.channel = channel
		# self.hiddens = hiddens
		# self.latent_dim = latent_dim
		# self.a = torch.rand(z.size()[0],z.size()[1])
		self.A = nn.Parameter(torch.rand(1, 1000), requires_grad=True).cuda()
		#
		# self.prev_channels = 3

		# 编码器
		# def cencoder(self,x,y):
		# modules_enc = []
		# img_length = self.img_length
		# prev_channels = self.channel
		# for cur_channels in self.hiddens:
		# 	modules_enc.append(
		# 		nn.Sequential(
		# 			nn.Conv2d(prev_channels,
		# 					  cur_channels,
		# 					  kernel_size=3,
		# 					  stride=2,
		# 					  padding=1), nn.BatchNorm2d(cur_channels),
		# 			nn.LeakyReLU())
		# 	)
		# 	prev_channels = cur_channels
		# 	img_length //= 2
		# self.encoder = nn.Sequential(*modules_enc)
		# if torch.cuda.is_available() == True:
		# 	self.encoder = self.encoder.cuda()
		# self.mean_linear = nn.Linear(prev_channels * img_length * img_length,
		# 							 latent_dim)
		# self.var_linear = nn.Linear(prev_channels * img_length * img_length,
		# 							latent_dim)
		# self.latent_dim = latent_dim
		#
		# self.mu = self.mean_linear
		# self.log_var = self.var_linear
		#
		# self.Liner_condition = nn.Linear(prev_channels * img_length * img_length+self.y_dim,prev_channels * img_length * img_length)

		# summary(self.encoder,(1,224,224))

		# self.z = reparameterize(self.mu, self.log_var)
		self.liner = nn.Linear(self.z_dim,self.z_dim)
		self.liner1000 = nn.Linear(self.z_dim, 1000)
		self.liner_y = nn.Linear(1000, 2)
		self.softmax = nn.Softmax(dim=1)
		self.liner_z = nn.Linear(self.z_dim,self.z_dim)
		self.linerez = nn.Linear(2048,1000)
		# self.sigmd = torch.sigmoid(100)
		# self.liner500_2 = nn.Linear(500,2)
		# self.sigmod = nn.

		# return self.m, self.v, self.z

		# decoder
		# def cdecoder(self):
		# modules_dec = []
		# self.decoder_projection = nn.Linear(prev_channels*img_length*img_length+self.y_dim,
		# 									prev_channels * img_length * img_length)
		# # self.decoder_projection = nn.Linear(
		# # 	latent_dim + self.y_dim, prev_channels * img_length * img_length)
		#
		# self.decoder_input_chw = (prev_channels, img_length, img_length)
		# for i in range(len(hiddens) - 1, 0, -1):
		# 	modules_dec.append(
		# 		nn.Sequential(
		# 			nn.ConvTranspose2d(hiddens[i],
		# 							   hiddens[i - 1],
		# 							   kernel_size=3,
		# 							   stride=2,
		# 							   padding=1,
		# 							   output_padding=1),
		# 			nn.BatchNorm2d(hiddens[i - 1]), nn.LeakyReLU())
		# 	)
		# modules_dec.append(
		# 	nn.Sequential(
		# 		nn.ConvTranspose2d(hiddens[0],
		# 						   hiddens[0],
		# 						   kernel_size=3,
		# 						   stride=2,
		# 						   padding=1,
		# 						   output_padding=1),
		# 		nn.BatchNorm2d(hiddens[0]), nn.LeakyReLU(),
		# 		nn.Conv2d(hiddens[0], self.channel, kernel_size=3, stride=1, padding=1),
		# 		nn.LeakyReLU()))
		# self.decoder = nn.Sequential(*modules_dec)
		# if torch.cuda.is_available() == True:
		# 	self.decoder = self.decoder.cuda()

	# summary(self.decoder,(256,7,7))
	# ttt=0

	# def reparameterize(self, mu, log_var):  # 重参数技巧
	# 	std = torch.exp(0.5 * log_var)  # 分布标准差std
	# 	eps = torch.randn_like(std)  # 从标准正态分布中采样,(n,128)
	# 	return mu + eps * std  # 返回对应正态分布中的采样值

		self.net = nn.Sequential(
			nn.Linear(self.z_dim , 200),
			nn.ELU(),
			nn.Linear(200, self.z_dim),
		)

	def masked(self, z):
		z = z.view(-1, self.z_dim)
		z = self.net(z)
		return z

	def dag(self, z):
		# z_in = z
		# a = torch.rand(z.size()[0],z.size()[1])
		# A = nn.Parameter(torch.rand(z.size()[0],z.size()[1]), requires_grad=True)
		# bias = nn.Parameter(torch.Tensor(z_in.size()[0]).cuda())
		# z_c = F.linear(z_in, A, bias)
		z_c = torch.mul(z, self.A)
		return z_c, self.A

	def forward(self, x):
		# y = y.cuda()
		x = x.cuda().float()
		# encoder = self.encoder(x)
		# encoder = torch.flatten(encoder, 1)
		# encoder = self.Liner_condition(torch.cat([encoder, y], dim=1))
		e = self.model_in(x)
		z_e = self.linerez(e)
		# A = self.A
		A = torch.sigmoid(self.A)
		z_a = torch.mul(z_e, A)
		# z_a = torch.mm(z_e, A)

		# self.A.retain_grad()
		# print('grad',self.A.grad)
		# self.A = A

		# z = self.liner_z(e)
		# self.z_dim = y.size()
		# z_m = self.masked(z_A)
		# y = self.liner(y)
		# z = self.liner1000(z_A)

		y = self.liner_y(z_a)
		y = self.softmax(y)
		# zzz=0
		# mean = self.mean_linear(encoder)
		# logvar = self.var_linear(encoder) + 1e-8
		# eps = torch.rand_like(logvar)
		# std = torch.exp(logvar / 2)
		# z = eps * std + mean
		# z_c = z

		# z_c, A = self.dag(z)

		# z_c, self.A = torch.mul(z,self.A)
		# x = self.decoder_projection(torch.cat((encoder, y), 1))
		# x = torch.reshape(x, (-1, *self.decoder_input_chw))
		# decoded = self.decoder(x)

		return y
		# return decoded, mean, logvar, z_c, self.A

	def loss_fn(self, x, x_rec):
		# kl_weight = 0.00025
		recons_loss = F.mse_loss(x_rec, x)
		# kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean ** 2 - torch.exp(logvar), 1), 0)
		loss = recons_loss.float()
		# loss = recons_loss + kl_loss * kl_weight
		# loss = loss.float()
		return loss

	# def loss_fn(self, x, x_rec, mean, logvar):
	# 	kl_weight = 0.00025
	# 	recons_loss = F.mse_loss(x_rec, x)
	# 	kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean ** 2 - torch.exp(logvar), 1), 0)
	# 	loss = recons_loss + kl_loss * kl_weight
	# 	return loss


def modifyModel(model):

	# print(model)
	# num_ftrs = model.fc.in_features
	# model.fc = nn.Sequential(nn.Linear(num_ftrs, 2),
	#                          nn.LogSoftmax(dim=1))
	# del model.fc
	# del model.avgpool
	# model.update()
	# print(model)
	model.fc = nn.Sequential()

	return model