import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import scipy.io as sio
import os

def standardization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


if __name__ == "__main__":
    matPath = '/home/cj/CSMRI_0325/data/val1/warm_up/warm_up/'
    savedir = '/home/cj/CSMRI_0325/data/val1/warm_up/warm_up_png/'
    npysave = '/home/cj/CSMRI_0325/data/val1/warm_up/npy/'
    for i in range(1811,18516):
            path = matPath + "DATA_"+str(i) + ".mat"
            if os.path.isfile(path) == False:
                # print(path + " 不存在 ")
                continue
            res = sio.loadmat(path)
            # res = sio.loadmat('/home/cj/CSMRI_0325/util/DATA_1855.mat')
            res = res['CS_K_Data']
            # im = imgFromSubF_pytorch(res)
            fshift = np.fft.ifftshift(res)  # 对整合好的频域数据进行逆变换
            img_back = np.fft.ifft2(fshift)
            # 出来的是复数，取绝对值，转化成实数
            img_back = np.abs(img_back)
            img_back[img_back<0]=0
            new_arr = standardization(img_back)*255
            
            y1 = Image.fromarray(new_arr)
            y1 = y1.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
            y1.save(savedir + "DATA_"+str(i) + ".png")
            np.save(npysave + "DATA_"+str(i) + ".npy",new_arr)
            # plt.subplot(144), plt.imshow(img_back, 'gray'), plt.title('inverse fourier')
            # plt.show()


    

    