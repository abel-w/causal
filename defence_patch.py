import os
import PIL.Image as Image
import torch
from resnet import ResNet,Bottleneck
from  causalNN import causalNN
import myUtils as mut
import numpy as np

################################################

#读取元素图像和对抗图像进入列表，第一个是原始图像，后面是对抗图像,[origin,att,att,att.....]
path_im = r"D:\pypro\data\samples_attack\patch"
list_all = os.listdir(path_im)
f_all = []
for i in list_all:
    f_all.append(os.path.join(path_im,i))
f_ori_att = []
list_ori = []
for i in f_all:
    if  'ori' in i:
        list_ori.append(i)
for i_f in list_ori:
    tmp = []
    nm_i = i_f.split('_ori_')[0]
    for j_f in f_all:
        if nm_i in j_f:
            tmp.append(j_f)
    if len(tmp) > 1:
        f_ori_att.append(tmp)


##################################################
#利用对抗图像测试resnet模型
model_resnet = torch.load(r"D:\pypro\causal-CVAE-paper\output\model_save\model_resnet\model_resnet_breast.pt")
model_resnet.eval()

model_causalNN = torch.load(r"D:\pypro\causal-CVAE-paper\output\model_save\model_causalNN\model_causalNN_breast.pt")
model_causalNN.eval()

# y_list_resnet = []
list_y_all =[]
# list_y_res =[]
for i in f_ori_att:
    tmp_ori=[]
    tmp_res=[]
    tmp_cau = []
    img_ori, max_x, min_x = mut.np2ten(np.array(Image.open(i[-1])))
    with torch.no_grad():
        y_ori = model_resnet(img_ori)[0]
    # tmp_ori.append(torch.argmax(y_ori).detach().cpu().item())
    label_ori = torch.argmax(y_ori).detach().cpu().item()
    tmp_ori.append(label_ori)
    # with torch.no_grad():
    #     y_ori = model_resnet(mut.np2ten(np.array(Image.open(i[-1]))))[0].cpu().numpy()
    # tmp_ori.append(np.argmax(y_ori))

    for j in i[0:-1]:
        img_res, max_x, min_x = mut.np2ten(np.array(Image.open(j)))
        with torch.no_grad():
            y_res = model_resnet(img_res)[0]
        # tmp_ori.append(torch.argmax(y_ori).detach().cpu().item())
        label_res = torch.argmax(y_res).detach().cpu().item()
        tmp_res.append(label_res)
        # with torch.no_grad():
        #     y_res = model_resnet(mut.np2ten(np.array(Image.open(j))))[0].cpu().numpy()
        # tmp_res.append(np.argmax(y_res))

    # for j in i[0:-2]:
        img_cau, max_x, min_x = mut.np2ten(np.array(Image.open(j)))
        with torch.no_grad():
            y_cau = model_resnet(img_cau)[0]
        # tmp_ori.append(torch.argmax(y_ori).detach().cpu().item())
        label_cau = torch.argmax(y_cau).detach().cpu().item()
        tmp_cau.append(label_cau)
        # with torch.no_grad():
        #     y_cau = model_causalNN(mut.np2ten(np.array(Image.open(j))))[0].cpu().numpy()
        # tmp_cau.append(np.argmax(y_cau))

    list_y_all.append([tmp_ori,tmp_res,tmp_cau])
##########################################
#计算准确率
n_s_res = []
n_s_cau = []
for y_i in list_y_all:
    if  abs(1-y_i[0][0]) in y_i[1]:
        n_s_res.append(1)
    if y_i[0][0] in y_i[2]:
        n_s_cau.append(1)
        # zz=0
    # if all(y_i[1] != y_i[0][0]):
    #     n_s_res.append(1)
    # if any(y_i[2] == y_i[0][0]):
    #     n_s_cau.append(1)

suc_res = sum(n_s_res)/len(list_y_all)
print('攻击成功率：',suc_res)

suc_cau = sum(n_s_cau)/len(list_y_all)
print('攻击成功率：',suc_cau)


print('_____________finish__________________')
