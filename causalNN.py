"""
torchvision的resnet50的预训练模型
增强breast数据集
"""

import numpy as np
import torch
import torch.nn as nn
import myUtils as mut
import os
from torch.utils.data import DataLoader
from torchsummary import summary
from torchstat import stat
import torch.nn.functional as F
import time

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# 分类数目
num_class = 2
# 各层数目
resnet18_params = [2, 2, 2, 2]
resnet34_params = [3, 4, 6, 3]
resnet50_params = [3, 4, 6, 3]
resnet101_params = [3, 4, 23, 3]
resnet152_params = [3, 8, 36, 3]


# 定义Conv1层
def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=places, kernel_size=7, stride=stride, padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )


# 浅层的残差结构
class BasicBlock(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=1):
        super(BasicBlock, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        # torch.Size([1, 64, 56, 56]), stride = 1
        # torch.Size([1, 128, 28, 28]), stride = 2
        # torch.Size([1, 256, 14, 14]), stride = 2
        # torch.Size([1, 512, 7, 7]), stride = 2
        self.basicblock = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(places * self.expansion),
        )

        # torch.Size([1, 64, 56, 56])
        # torch.Size([1, 128, 28, 28])
        # torch.Size([1, 256, 14, 14])
        # torch.Size([1, 512, 7, 7])
        # 每个大模块的第一个残差结构需要改变步长
        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(places * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 实线分支
        residual = x
        out = self.basicblock(x)

        # 虚线分支
        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


# 深层的残差结构
class Bottleneck(nn.Module):

    # 注意:默认 downsampling=False
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            # torch.Size([1, 64, 56, 56])，stride=1
            # torch.Size([1, 128, 56, 56])，stride=1
            # torch.Size([1, 256, 28, 28]), stride=1
            # torch.Size([1, 512, 14, 14]), stride=1
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            # torch.Size([1, 64, 56, 56])，stride=1
            # torch.Size([1, 128, 28, 28]), stride=2
            # torch.Size([1, 256, 14, 14]), stride=2
            # torch.Size([1, 512, 7, 7]), stride=2
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            # torch.Size([1, 256, 56, 56])，stride=1
            # torch.Size([1, 512, 28, 28]), stride=1
            # torch.Size([1, 1024, 14, 14]), stride=1
            # torch.Size([1, 2048, 7, 7]), stride=1
            nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places * self.expansion),
        )

        # torch.Size([1, 256, 56, 56])
        # torch.Size([1, 512, 28, 28])
        # torch.Size([1, 1024, 14, 14])
        # torch.Size([1, 2048, 7, 7])
        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(places * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 实线分支
        residual = x
        out = self.bottleneck(x)

        # 虚线分支
        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class causalNN(nn.Module):
    def __init__(self, blocks, blockkinds, num_classes=num_class):
        super(causalNN, self).__init__()

        self.blockkinds = blockkinds
        self.conv1 = Conv1(in_planes=3, places=64)
        self.z_dim = 2048

        self.net = nn.Sequential(
            nn.Linear(self.z_dim, 1000),
            nn.ELU(),
            nn.Linear(1000, self.z_dim),
          )

        # 对应浅层网络结构
        if self.blockkinds == BasicBlock:
            self.expansion = 1
            # 64 -> 64
            self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
            # 64 -> 128
            self.layer2 = self.make_layer(in_places=64, places=128, block=blocks[1], stride=2)
            # 128 -> 256
            self.layer3 = self.make_layer(in_places=128, places=256, block=blocks[2], stride=2)
            # 256 -> 512
            self.layer4 = self.make_layer(in_places=256, places=512, block=blocks[3], stride=2)

            self.fc = nn.Linear(512, num_classes)

        # 对应深层网络结构
        if self.blockkinds == Bottleneck:
            self.expansion = 4
            # 64 -> 64
            self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
            # 256 -> 128
            self.layer2 = self.make_layer(in_places=256, places=128, block=blocks[1], stride=2)
            # 512 -> 256
            self.layer3 = self.make_layer(in_places=512, places=256, block=blocks[2], stride=2)
            # 1024 -> 512
            self.layer4 = self.make_layer(in_places=1024, places=512, block=blocks[3], stride=2)

            self.fc = nn.Linear(2048, num_classes)

        self.avgpool = nn.AvgPool2d(7, stride=1)

        # 初始化网络结构
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 采用了何凯明的初始化方法
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):

        layers = []

        # torch.Size([1, 64, 56, 56])  -> torch.Size([1, 256, 56, 56])， stride=1 故w，h不变
        # torch.Size([1, 256, 56, 56]) -> torch.Size([1, 512, 28, 28])， stride=2 故w，h变
        # torch.Size([1, 512, 28, 28]) -> torch.Size([1, 1024, 14, 14])，stride=2 故w，h变
        # torch.Size([1, 1024, 14, 14]) -> torch.Size([1, 2048, 7, 7])， stride=2 故w，h变
        # 此步需要通过虚线分支，downsampling=True
        layers.append(self.blockkinds(in_places, places, stride, downsampling=True))

        # torch.Size([1, 256, 56, 56]) -> torch.Size([1, 256, 56, 56])
        # torch.Size([1, 512, 28, 28]) -> torch.Size([1, 512, 28, 28])
        # torch.Size([1, 1024, 14, 14]) -> torch.Size([1, 1024, 14, 14])
        # torch.Size([1, 2048, 7, 7]) -> torch.Size([1, 2048, 7, 7])
        # print("places*self.expansion:", places*self.expansion)
        # print("block:", block)
        # 此步需要通过实线分支，downsampling=False， 每个大模块的第一个残差结构需要改变步长
        for i in range(1, block):
            layers.append(self.blockkinds(places * self.expansion, places))

        return nn.Sequential(*layers)


    def masked(self, z):
        z = z.view(-1, self.z_dim)
        z = self.net(z)
        return z

    # def masked(self, z):
    #     z = z.view(-1, self.z_dim)
    #     z = self.net(z)
    #     return z

    def forward(self, x):

        # conv1层
        x = self.conv1(x)  # torch.Size([1, 64, 56, 56])

        # conv2_x层
        x = self.layer1(x)  # torch.Size([1, 256, 56, 56])
        # conv3_x层
        x = self.layer2(x)  # torch.Size([1, 512, 28, 28])
        # conv4_x层
        x = self.layer3(x)  # torch.Size([1, 1024, 14, 14])
        # conv5_x层
        x = self.layer4(x)  # torch.Size([1, 2048, 7, 7])
        x = self.avgpool(x)  # torch.Size([1, 2048, 1, 1]) / torch.Size([1, 512])
        #因果推断层
        x = self.masked(x)

        x = x.view(x.size(0), -1)  # torch.Size([1, 2048]) / torch.Size([1, 512])
        x = self.fc(x)  # torch.Size([1, 5])

        return x

    # def forward(self, x):
    #
    #     # conv1层
    #     x = self.conv1(x)  # torch.Size([1, 64, 56, 56])
    #
    #     # conv2_x层
    #     x = self.layer1(x)  # torch.Size([1, 256, 56, 56])
    #     # conv3_x层
    #     x = self.layer2(x)  # torch.Size([1, 512, 28, 28])
    #     # conv4_x层
    #     x = self.layer3(x)  # torch.Size([1, 1024, 14, 14])
    #     # conv5_x层
    #     x = self.layer4(x)  # torch.Size([1, 2048, 7, 7])
    #
    #     x = self.avgpool(x)  # torch.Size([1, 2048, 1, 1]) / torch.Size([1, 512])
    #     x = x.view(x.size(0), -1)  # torch.Size([1, 2048]) / torch.Size([1, 512])
    #     x = self.fc(x)  # torch.Size([1, 5])
    #
    #     return x


def causalNN18():
    return causalNN(resnet18_params, BasicBlock)


def causalNN34():
    return causalNN(resnet34_params, BasicBlock)


def causalNN50():
    return causalNN(resnet50_params, Bottleneck)


def causalNN101():
    return causalNN(resnet101_params, Bottleneck)


def causalNN152():
    return causalNN(resnet152_params, Bottleneck)


if __name__ == '__main__':
    # model = torchvision.models.resnet50()
    # malignant
    # benign
    # 模型测试
    # model = causalNN18()
    # model = causalNN34()
    # model = causalNN50()
    # model = model.to(device)

    ################################################################################
    batch_size_su = 32
    total_epochs_su = 20
    nm_ds = 'breast'
    #################################################################################
    model = causalNN50()
    # model = torch.load('./model_save/resnet50_pretrained.pt')
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_ftrs, 2),
                             nn.Softmax(dim=1))
    print(model)
    # summary(model,(3,64,64))
    mut.check_net_parameter(model)
    # stat(model,(3,224,224))

    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    path_savemodel = './output/model_save'
    if not os.path.exists(path_savemodel):
        os.makedirs(path_savemodel)

    path_ds = os.path.join('D:/pypro/data/breast')
    # list_mag = []
    # fold_ds = ['malignant', 'benign']
    train_be, var_be, test_be, train_ma, var_ma, test_ma = mut.datasetAll(path_ds)

    dataset_train = mut.getData_breast(train_be, train_ma, number_samples=20000)
    train_dataset = DataLoader(dataset=dataset_train, batch_size=batch_size_su, drop_last=True, shuffle=True)

    # path_var = os.path.join('D:/pypro/data/', nm_ds + '_enhance', 'var')
    dataset_var = mut.getData_breast(var_be, var_ma, number_samples=1000)
    var_dataset = DataLoader(dataset=dataset_var, batch_size=batch_size_su, drop_last=True, shuffle=True)

    # path_test = os.path.join('D:/pypro/data/', nm_ds + '_enhance', 'test')
    dataset_test = mut.getData_breast(train_be, train_ma, number_samples=1000)
    test_dataset = DataLoader(dataset=dataset_test, batch_size=batch_size_su, drop_last=True, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()

    global model_best
    model_best = model
    list_out = []
    list_acc_varify = [0]
    # dic_model = {}
    list_loss = []

    for epoch in range(total_epochs_su):
        model.train()
        totsl_loss = 0
        total_rec = 0
        total_kl = 0
        num_right = 0
        list_loss.append('epoch_%d' % epoch)
        for i, (x, y) in enumerate(train_dataset):  # 循环iter
            optimizer.zero_grad()
            x = x.float().to(device)
            y = y.float().cuda()

            y_pred = model(x)
            # a = model.layer4

            loss = loss_fn(y_pred, y)
            loss0 = loss.item()
            list_loss.append(loss0)

            # optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totsl_loss += loss

            # for j in range(len(y_pred)):
            #     if torch.argmax(y_pred[j]) == torch.argmax(y[j]):
            #         num_right += 1

            if i % 10 == 0:
                list_acc_dis = []
                for i_dis in range(batch_size_su):
                    if torch.argmax(y[i_dis]) == torch.argmax(y_pred[i_dis]):
                        list_acc_dis.append(1)
                acc_dis = sum(list_acc_dis) / batch_size_su
                print('resnet_%s_epoch-%d / %d - iter - %d / %d-loss: %f-accurace:%f' % (nm_ds,
                                                                                    epoch + 1, total_epochs_su, i + 1,
                                                                                    len(dataset_train) // batch_size_su,
                                                                                    loss0, acc_dis))

        ##################################################################################
        acc_varify, y_all, model_e = mut.varify(nm_ds, var_dataset, model)

        # list_out.append([acc_varify, y_all, model_e])
        if acc_varify >= max(list_acc_varify):
            model_best = model_e
            y_all_best = y_all
            torch.save(model, os.path.join(path_savemodel, 'model_temp.pt'))
            torch.save(model.state_dict(), os.path.join(path_savemodel, 'model_dic_temp.pt' ))
        list_acc_varify.append(acc_varify)  ####当前所有epoch的精确度

        with open('./accuracy_running.txt', 'w+') as f:
            f.write(str(list_acc_varify) + '\n')
            f.write('best accuracy is %f' % max(list_acc_varify) + '\n')
            for l in y_all_best:
                f.write('%f, %f, %.4f, %.4f' % (l[0], l[1], l[2], l[3]) + '\n')

    t_f = mut.timeNow()  # 获取现在时间
    with open('./accuracy_varify_causalNN_%s_%s.txt' % (nm_ds, t_f), 'w+') as f:
        f.write(str(list_acc_varify) + '\n')
        f.write('best accuracy: %f' % max(list_acc_varify) + '\n')
        for l in y_all_best:
            f.write('%f, %f, %.4f, %.4f' % (l[0], l[1], l[2], l[3]) + '\n')

    # acc_list, y_max, model_max = mut.output_process(list_out)
    with open('./loss_causalNN_%s_%s.txt' % (nm_ds, t_f), 'w+') as f:
        f.write(str(list_loss))

    torch.save(model_best, os.path.join(path_savemodel, 'model_causalNN_%s_%s.pt' % (nm_ds, t_f)))
    torch.save(model_best.state_dict(), os.path.join(path_savemodel, 'model_causalNN_dic_%s_%s.pt' % (nm_ds, t_f)))

    acc_test, y_test = mut.test(nm_ds, test_dataset, model_best)

    with open('./accuracy_causalNN_%s_%s.txt' % (nm_ds, t_f), 'w+') as f:
        f.write(str(list_acc_varify) + '\n')
        f.write('best varify accuracy: %f' % max(list_acc_varify) + '\n')
        # for l in y_max:
        #     f.write('%f, %f, %.4f, %.4f'%(l[0],l[1],l[2],l[3])+'\n')
        f.write('test accuracy: %f' % acc_test + '\n')
        for l in y_test:
            f.write('%f, %f, %.4f, %.4f' % (l[0], l[1], l[2], l[3]) + '\n')

    print('----------------------run finish---------------------')
