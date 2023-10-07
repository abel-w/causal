# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 23:55:46 2023

@author: Administrator
"""
import random

from torch.utils.data import Dataset
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
from torchstat import stat
# from numpy import random
from PIL import Image
from torchvision import transforms
import torch.nn.utils.prune as prune
import torch.nn as nn
import time


#################################################


class getData_breast(Dataset):
    def __init__(self, train_be, train_ma, number_samples=20000):
        super(Dataset, self).__init__()

        self.number_samples = number_samples

        list_1 = random.sample(train_ma, int(number_samples / 2))
        list_0 = random.sample(train_be, int(number_samples / 2))
        # for i in os.listdir(self.path_tr_1):
        #     self.list_1.append(os.path.join(self.path_tr_1,i))
        # for i in os.listdir(self.path_tr_0):
        #     self.list_0.append(os.path.join(self.path_tr_0,i))
        #
        # if self.number_samples < min(len(self.list_0), len(self.list_1)):
        #     self.list_1 = random.sample(self.list_1,self.number_samples)
        #     self.list_0 = random.sample(self.list_0,self.number_samples)
        # else:
        #     self.list_1 = random.sample(self.list_1,len(self.list_1))
        #     self.list_0 = random.sample(self.list_0,len(self.list_0))

        self.ds_xy = []
        for i in list_0:
            self.ds_xy.append([i, [1, 0]])
        for i in list_1:
            self.ds_xy.append([i, [0, 1]])

        random.shuffle(self.ds_xy)

        # for i in range(len(self.list_ds)):
        #     if 'benign' in self.list_ds[i]:
        #         self.ds_xy_0.append([self.list_ds[i],[1,0]])
        #     elif 'malignant' in self.list_ds[i]:
        #         self.ds_xy_1.append([self.list_ds[i],[0,1]])
        #
        # self.ds_xy = (random.sample(self.ds_xy_0, int(number_samples/2))+
        #               random.sample(self.ds_xy_1, int(number_samples/2)))

        # zz=0

        # for i in self.list_0:
        #     self.dic_ds[i] = np.array([1,0])
        #
        # for i in self.list_1:
        #     self.dic_ds[i] = np.array([0, 1])

    def __len__(self):
        return len(self.ds_xy)

    def __getitem__(self, idx):
        # path_img = list(self.dic_ds.keys())[idx]
        # path_img = Image.open(self.ds_xy[idx][0])
        # random.shuffle(self.ds_xy)

        x = np.array(Image.open(self.ds_xy[idx][0]))
        y = np.array(self.ds_xy[idx][1])
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=0)

        # 归一化
        # x_mean, x_std = np.mean(x), np.std(x)

        # x_norm = (x - x_mean) / x_std
        # print(self.list_ds[idx])
        x_max = np.max(x)
        x_min = np.min(x)

        x_norm = (x - x_min) / (x_max - x_min)

        x_norm = np.transpose(x_norm, [2, 0, 1])

        return x_norm, y


class getData(Dataset):
    def __init__(self, path, number_samples):
        super(Dataset, self).__init__()
        self.path_tr_1 = os.path.join(path, '1')
        self.path_tr_0 = os.path.join(path, '0')
        self.number_samples = number_samples

        self.list_1 = []
        self.list_0 = []
        for i in os.listdir(self.path_tr_1):
            self.list_1.append(os.path.join(self.path_tr_1, i))
        for i in os.listdir(self.path_tr_0):
            self.list_0.append(os.path.join(self.path_tr_0, i))

        if self.number_samples < min(len(self.list_0), len(self.list_1)):
            self.list_1 = random.sample(self.list_1, self.number_samples)
            self.list_0 = random.sample(self.list_0, self.number_samples)
        else:
            self.list_1 = random.sample(self.list_1, len(self.list_1))
            self.list_0 = random.sample(self.list_0, len(self.list_0))

        self.dic_ds = {}
        for i in self.list_0:
            self.dic_ds[i] = np.array([1, 0])

        for i in self.list_1:
            self.dic_ds[i] = np.array([0, 1])

    def __len__(self):
        return len(self.dic_ds)

    def __getitem__(self, idx):
        path_img = list(self.dic_ds.keys())[idx]

        x = np.array(Image.open(path_img))
        y = self.dic_ds[path_img]
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=0)

        # 归一化
        # x_mean, x_std = np.mean(x), np.std(x)

        # x_norm = (x - x_mean) / x_std
        # print(self.list_ds[idx])
        x_max = np.max(x)
        x_min = np.min(x)

        x_norm = (x - x_min) / (x_max - x_min)

        x_norm = np.transpose(x_norm, [2, 0, 1])

        return x_norm, y


class DataFeed(Dataset):
    def __init__(self, path_ds, batch_size=16):
        super(Dataset, self).__init__()
        self.batch_size = batch_size
        self.path_tr_1 = os.path.join(path_ds, '1')
        self.path_tr_0 = os.path.join(path_ds, '0')
        self.list_1 = []
        self.list_0 = []

        # list_ds_y = []
        for i in os.listdir(self.path_tr_1):
            self.list_1.append(os.path.join(self.path_tr_1, i))
            # self.dic_ds[os.path.join(self.path_tr_1, i)] = [0, 1]
        for i in os.listdir(self.path_tr_0):
            self.list_0.append(os.path.join(self.path_tr_0, i))

    def datafeed(self):
        # i = 0
        list_feed_x = random.sample(self.list_1, int(self.batch_size / 2)) + random.sample(self.list_0,
                                                                                           int(self.batch_size / 2))
        arr_feed_y = np.concatenate((np.ones(int(self.batch_size / 2)), np.zeros(int(self.batch_size / 2))), 0)
        list_ds = []
        for j in range(len(list_feed_x)):
            x = np.load(list_feed_x[j])
            x = (x - np.min(x)) / (np.max(x) - np.min(x))
            # plt.imshow(x)
            # plt.show()
            list_ds.append(x)

        x = torch.tensor(list_ds)
        x = torch.permute(x, [0, 3, 1, 2])
        y = torch.tensor(arr_feed_y)

        return x, y


global acc_total
acc_total = 0


def varify(nm_ds, dataset, model):
    # self.acc0 = 0
    # acc_total = 0
    out_p_o_str_best = []
    # global acc_total
    # global model_best
    out_p_o = []
    # out_p_o_str = []
    acc_e = 0
    list_y = []
    list_y_pred = []
    for i, (x, y) in enumerate(dataset):
        # optimizer.zero_grad()
        model_val = model
        model_val.eval()
        x = x.float().to(device)
        y = y.float().cuda()

        with torch.no_grad():
            y_pred = model_val(x)

        list_y.extend(y.cpu().numpy())
        list_y_pred.extend(y_pred.cpu().numpy())

    # zzz=0
    for j in range(len(list_y)):
        out_p_o.append([np.argmax(list_y[j]), np.argmax(list_y_pred[j]), list_y_pred[j][0], list_y_pred[j][1]])

    out_p_o_arry = np.array(out_p_o)

    # out_p_o_str.append(
    #     [np.argmax(y[j].cpu().numpy()), torch.argmax(y_pred[j]).item(), '%.3f' % y_pred[j].cpu().numpy()[0],
    #      '%.3f' % y_pred[j].cpu().numpy()[1]])

    if not ((out_p_o_arry[:, 1] == 0).all() or (out_p_o_arry[:, 1] == 1).all()):
        for k in range(len(out_p_o_arry)):
            if out_p_o_arry[k, 0] == out_p_o_arry[k, 1]:
                acc_e += 1
    acc_epoch = acc_e / len(out_p_o_arry)

    print('the varify accurace: %f' % acc_epoch)

    return acc_epoch, out_p_o_arry, model


def test(nm_ds, dataset, model):
    out_p_o = []
    acc_e = 0
    list_y = []
    list_y_pred = []
    for i, (x, y) in enumerate(dataset):
        # optimizer.zero_grad()
        model_test = model
        model_test.eval()
        x = x.float().to(device)
        y = y.float().cuda()

        with torch.no_grad():
            y_pred = model_test(x)

        list_y.extend(y.cpu().numpy())
        list_y_pred.extend(y_pred.cpu().numpy())

    for j in range(len(list_y)):
        out_p_o.append([np.argmax(list_y[j]), np.argmax(list_y_pred[j]), list_y_pred[j][0], list_y_pred[j][1]])

    out_p_o_arry = np.array(out_p_o)

    # out_p_o_str.append(
    #     [np.argmax(y[j].cpu().numpy()), torch.argmax(y_pred[j]).item(), '%.3f' % y_pred[j].cpu().numpy()[0],
    #      '%.3f' % y_pred[j].cpu().numpy()[1]])

    if not ((out_p_o_arry[:, 1] == 0).all() or (out_p_o_arry[:, 1] == 1).all()):
        for k in range(len(out_p_o_arry)):
            if out_p_o_arry[k, 0] == out_p_o_arry[k, 1]:
                acc_e += 1
    acc_epoch = acc_e / len(out_p_o_arry)

    print('the test accurace: %f' % acc_epoch)

    return acc_epoch, out_p_o_arry


def output_process(list_out):
    acc = 0
    acc_list = []
    idx = 0
    for i in range(len(list_out)):
        acc_list.append(list_out[i][0])
        if list_out[i][0] > acc:
            acc = list_out[i][0]
            idx = i

    y_max = list_out[idx][1]
    model_max = list_out[idx][2]
    # zzz=0

    return acc_list, y_max, model_max


def check_net_parameter(model):
    with open('./parameter of causalResNet.txt', 'w+') as f:
        for name, parameters in model.named_parameters():
            f.write(name + ':' + str(parameters.size()) + '\n')


def net_stat(model, input_shape=(3, 224, 224)):
    stat_net = stat(model, input_shape)
    with open('./state of model.txt', 'w+') as f:
        for l in stat_net:
            f.write(l + '\n')


def summary_txt(model, input_shape=(3, 224, 224)):
    stat_net = summary(model, input_shape)
    with open('./state of model(summary).txt', 'w+') as f:
        for l in stat_net:
            f.write(l + '\n')


def dataPatch(ds_in, width_img, width_enhance, factor_enhance):
    list_img_patch = []
    list_coor_x = np.int32(random.normal(loc=112, scale=int((112 - width_enhance / 2) / 2), size=factor_enhance))
    print(max(list_coor_x), min(list_coor_x))
    list_coor_x = arrBox(list_coor_x, width_img=width_img, width_enhance=width_enhance)
    list_coor_x = arrBox(
        np.int32(random.normal(loc=112, scale=int((112 - width_enhance / 2) / 2), size=factor_enhance)),
        width_img=width_img, width_enhance=width_enhance)
    list_coor_y = arrBox(
        np.int32(random.normal(loc=112, scale=int((112 - width_enhance / 2) / 2), size=factor_enhance)),
        width_img=width_img, width_enhance=width_enhance)
    # list_coor_x = random.sample(range(int(0 + width_img/2+1),int(width_img-width_enhance/2-1)), factor_enhance)
    # list_coor_y = random.sample(range(int(0 + width_img/2+1),int(width_img-width_enhance/2)-1), factor_enhance)
    for c_x, c_y in zip(list_coor_x, list_coor_y):
        ds_patch = ds_in[int(c_x - width_enhance / 2):int(c_x + width_enhance / 2),
                   int(c_y - width_enhance / 2):int(c_y + width_enhance / 2)]

        # plt.imshow(ds_patch/255)
        # plt.close()

        list_img_patch.append(ds_patch)

    return list_img_patch


def arrBox(arr, width_img, width_enhance):
    # zzz=0
    for i in range(len(arr)):
        if arr[i] < int(width_enhance / 2 + 1):
            arr[i] = random.randint(int(width_enhance / 2 + 1), int(width_img - width_enhance / 2 - 1))
        if arr[i] > int(width_img / 2 + width_enhance / 2 - 1):
            arr[i] = random.randint(int(width_enhance / 2 + 1), int(width_img / 2 + width_enhance / 2 - 1))

    # zzz=0

    return arr


def dataEnhance(im_in, factor_enhance):
    # entropy_img = imgEntropy_1d(im_in)
    list_img_enhance = []
    list_img_enhance.append(im_in)
    im_1 = transforms.RandomHorizontalFlip()(im_in)
    list_img_enhance.append(im_1)
    for i in range(int(factor_enhance / 3) + 1):
        im_rc = transforms.RandomCrop(180)(im_in)
        im_2 = transforms.Resize(224)(im_rc)
        list_img_enhance.append(im_2)
        # plt.imshow(im_2)

        im_3 = transforms.RandomRotation(random.randint(360))(im_in)
        list_img_enhance.append(im_3)
        # plt.imshow(im_3)

        im_4 = transforms.RandomCrop(180)(im_3)
        im_4 = transforms.Resize(224)(im_4)
        list_img_enhance.append(im_4)
        # plt.imshow(im_4)

    return list_img_enhance


def dataEnhance_breast(im, factor_enhance):
    # entropy_img = imgEntropy_1d(im_in)
    list_img_enhance = []
    for i in range(int(factor_enhance / 3) + 1):
        im_224 = transforms.RandomCrop(224)(im)
        list_img_enhance.append(im_224)
        im_hf = transforms.RandomHorizontalFlip(p=1)(im_224)
        list_img_enhance.append(im_hf)
        im_vf = transforms.RandomVerticalFlip(p=1)(im_224)
        list_img_enhance.append(im_vf)

    return list_img_enhance


def imgEntropy_1d(img):
    arr_img = np.reshape(np.array(img), (np.array(img).size))
    arr_p = {}
    # arr_p = [i for i in range(256)]
    for j in range(255):
        arr_p[j] = np.sum(np.array(arr_img == j, dtype=int)) / (img.size[0] * img.size[1])

    e_i = 0
    for i in arr_p.keys():
        if arr_p[i] != 0:
            e_i += arr_p[i] * np.log2(arr_p[i])
    e_i = -e_i

    return e_i


def AEeval(model, data):
    # out_p_o = []
    # acc_e = 0
    list_loss = []
    # list_y_pred = []
    for i, (x, y) in enumerate(data):
        # optimizer.zero_grad()
        model_val = model
        model_val.eval()
        x = x.float().to(device)
        # y = y.float().cuda()

        with torch.no_grad():
            x_rec = model_val(x)
        loss = model_val.loss_fn(x, x_rec)

        list_loss.append(loss)

        # list_y.extend(y.cpu().numpy())
        # list_y_pred.extend(y_pred.cpu().numpy())
    loss_mean = sum(list_loss) / i
    loss_mean = loss_mean.item()
    # zzz=0

    return loss_mean


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


def timeNow():
    t = time.localtime()
    t_f = '%0d_%02d_%02d_%02d_%02d' % (t.tm_year - 2000, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min)
    return t_f


def dsNorm(ds):
    if type(ds) is np.ndarray:
        ds_out = (ds - np.min(ds)) / (np.max(ds) - np.min(ds))
    elif torch.is_tensor(ds):
        ds_out = (ds - torch.min(ds)) / (torch.max(ds) - torch.min(ds))
    return ds_out


def datasetAll(path_ds):
    list_ma = []
    list_be = []
    # fold_ds = os.listdir(path_ds)

    for filepath, dirnames, filenames in os.walk(os.path.join(path_ds)):
        if len(filenames) > 0 and 'benign' in filepath:
            for i_n in filenames:
                list_be.append(os.path.join(filepath, i_n))

        elif len(filenames) > 0 and 'malignant' in filepath:
            for i_n in filenames:
                list_ma.append(os.path.join(filepath, i_n))

    len_be = len(list_be)
    random.shuffle(list_be)
    train_be = list_be[0:int(len_be * 0.8)]
    var_be = list_be[int(len_be * 0.8):int(len_be * 0.9)]
    test_be = list_be[int(len_be * 0.9):len_be]

    len_ma = len(list_ma)
    random.shuffle(list_ma)
    train_ma = list_ma[0:int(len_ma * 0.8)]
    var_ma = list_ma[int(len_ma * 0.8):int(len_ma * 0.9)]
    test_ma = list_ma[int(len_ma * 0.9):len_ma]

    return train_be, var_be, test_be, train_ma, var_ma, test_ma


###数组转换网络输入tensor
def np2ten(x):  #
    x = np.transpose(x, [2, 0, 1])
    x = np.float32(x) / 255
    x = (x-np.min(x))/(np.max(x)-np.min(x))
    x = torch.tensor(x)
    x = torch.unsqueeze(x, 0)
    x = x.cuda()
    max_x = torch.max(x)
    min_x = torch.min(x)
    return x, max_x, min_x


def norm01(x):
    max_x=0
    min_x =0
    if type(x) is np.ndarray:
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        max_x = np.max(x)
        min_x = np.min(x)
    elif torch.is_tensor(x):
        x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
        max_x = torch.max(x)
        min_x = torch.min(x)
    return x, max_x, min_x


def ten2np(x):
    x= torch.squeeze(x)
    x= torch.permute(x,[1,2,0])
    x= x.detach().cpu().numpy()
    x,_,_=norm01(x)
    x=x*255
    return x