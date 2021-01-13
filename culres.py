import os
import numpy as np
import math
from PIL import Image

import time

start = time.clock()


# 当中是你的程序


def psnr(img1, img2):
    mse = np.mean((img1 / 1. - img2 / 1.) ** 2)
    if mse < 1.0e-10:
        return 100 * 1.0
    return 10 * math.log10(255.0 * 255.0 / mse)


def mse(img1, img2):
    mse = np.mean((img1 / 1. - img2 / 1.) ** 2)
    return mse


def ssim(y_true, y_pred):
    u_true = np.mean(y_true)
    u_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    std_true = np.sqrt(var_true)
    std_pred = np.sqrt(var_pred)
    c1 = np.square(0.01 * 7)
    c2 = np.square(0.03 * 7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom


path1 = 'test2/'  # 指定输出结果文件夹
path2 = 'resinput/'  # 指定原图文件夹
f_nums = len(os.listdir(path1))
list_psnr = []
list_ssim = []
list_mse = []
for i in range(78,79):
    path = path1 + 'IM_DATA_' + str(i) + '.png'
    if os.path.isfile(path) == False:
                # print(path + " 不存在 ")
                continue
    img_a = Image.open(path)
    path_ = path2 + 'test_input_DATA_' + str(i) + '_y.png'
    if os.path.isfile(path_) == False:
                # print(path + " 不存在 ")
                continue
    img_b = Image.open(path_) 
    img_b = img_b.transpose(Image.ROTATE_180)
    img_a = np.array(img_a)
    img_b = np.array(img_b)

    psnr_num = psnr(img_a, img_b)
    print("------------------------------------")
    print(img_a)
    print("------------------------------------")
    print(img_b)
    ssim_num = ssim(img_a, img_b)
    mse_num = mse(img_a, img_b)
    list_ssim.append(ssim_num)
    list_psnr.append(psnr_num)
    list_mse.append(mse_num)
print("平均PSNR:", np.mean(list_psnr))   #,list_psnr)
print("平均SSIM:", np.mean(list_ssim))   #,list_ssim)
print("平均MSE:", np.mean(list_mse))   #,list_mse)

elapsed = (time.clock() - start)
print("Time used:", elapsed)
