 
import numpy as np
from PIL import Image
import torch.utils.data as data
import scipy.io as sio
import random
import os
import h5py
from skimage import transform
# import imageUtil
from util.imageUtil import *


def standardization(data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

matPath = "/home/cj/CSMRI_0325/data/val1/warm_up/warm_up/"
for i in range(18101,18110):
            path = matPath + "DATA_"+str(i) + ".mat"
            if os.path.isfile(path) == False:
                # print(path + " 不存在 ")
                continue
            mDic = sio.loadmat(path)
            name = "DATA_"+str(i) + ".mat"
            
            # CS_K_Dat --> subF
            # 网络输入的是subf，就是下采样的k空间数据，但是是iffshift后的
            # 所以ckd需要ifftshift转换一下,转换后的复数数据转成实数
            CS_K_Data = mDic['CS_K_Data']
            CS_K_Data = np.fft.ifftshift(CS_K_Data)
            w,h = CS_K_Data.shape
            ckd = np.zeros((2,w,h))
            ckd[0] = CS_K_Data.real
            ckd[1] = CS_K_Data.imag
            # 注意后面需要permute一下，现在的维度为(2,w,h)

            # CS_K_Data ---> 实部域png
            res = mDic['CS_K_Data']           
            fshift = np.fft.ifftshift(res) 
            img_back = np.fft.ifft2(fshift)
            img = img_back.real
            # img_back = np.abs(img_back)
            

            # show png
            img[img < 0] = 0
            img = standardization(img)*255
            png = Image.fromarray(img)
            png = png.convert('L')
            png.save('resoutput/ffttest_ini_'+ name.split('.')[0]+'.png')          
            
            

            # png = transform.resize(new_arr,(256,256))

            # 现在对图像做fft看subf是否一致
            fdata = np.fft.fft2(img_back)
            fftshi = np.fft.fftshift(fdata)
            w,h = fftshi.shape
            newckd = np.zeros((2,w,h))
            newckd[0] = fftshi.real
            newckd[1] = fftshi.imag 

            print(ckd == newckd)



            
            