import os
import PIL.Image as Image
import torch
from resnet import ResNet, Bottleneck
from causalNN import causalNN
import myUtils as mut
import numpy as np

################################################
if __name__ == '__main__':
    # 读取元素图像和对抗图像进入列表，第一个是原始图像，后面是对抗图像,[origin,att,att,att.....]
    path_im = r"D:\pypro\adversarial-attack-ww\output\pixels_attack\5_pix"
    list_all0 = os.listdir(path_im)
    sam_all = []
    for i in list_all0:
        if '@' in i:
            sam_all.append(os.path.join(path_im, i))

    y_true = []
    label_true = []
    for i in sam_all:
        f_c = i.split('@')[1].split('.png')[0].split('_')
        y_true.append([float(f_c[0]), float(f_c[1])])
        label_true.append(np.argmax(f_c))


    ##################################################
    # 利用对抗图像测试resnet模型
    model_resnet = torch.load(r"D:\pypro\causal-CVAE-paper\output\model_save\model_resnet\model_resnet_breast.pt")
    model_resnet.eval()

    model_causalNN = torch.load(r"D:\pypro\causal-CVAE-paper\output\model_save\model_causalNN\model_causalNN_breast.pt")
    model_causalNN.eval()

    # y_list_resnet = []
    y_res_cal = []
    # list_y_res =[]
    for i in range(len(sam_all)):
        img = np.array(Image.open(sam_all[i]))
        img, max_x, min_x = mut.np2ten(img)
        with torch.no_grad():
            y_res = model_resnet(img)[0].detach().cpu().numpy()
        label_res = np.argmax(y_res)
        with torch.no_grad():
            y_cal = model_causalNN(img)[0].detach().cpu().numpy()
        label_cal = np.argmax(y_cal)
        y_res_cal.append([label_true[i], label_res, label_cal])

    y_res_cal = np.array(y_res_cal)
    suc_res = 0
    suc_cal = 0
    for i in range(len(y_res_cal)):
        if y_res_cal[i][0] != y_res_cal[i][1]:
            suc_res += 1
        if y_res_cal[i][0] == y_res_cal[i][2]:
            suc_cal += 1

    acc_res = suc_res / len(y_res_cal)
    acc_cal = suc_cal / len(y_res_cal)

    print('attack: ', acc_res)
    print('defence: ', acc_cal)

    print('_____________finish__________________')
