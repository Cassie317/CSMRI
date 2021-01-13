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

fakeRandomPath = 'mask/mask_r30k_29.mat'
fakeRandomPath_15 = 'mask/mask_r10k_15.mat'
fakeRandomPath_10 = 'mask/mask_r4k_10.mat'
fakeRandomPath_5 = 'mask/mask_r4k_5.mat'
# 外部首先调用本函数
def getDataloader(dataType='1in1', mode='train', batchSize=8, crossValid=0):

    if('complex' in dataType):
        dataMode = 'complex'
    elif('abs' in dataType):
        dataMode = 'abs'
    else:
        dataMode = 'abs'
    if(mode == 'train'):
            shuffleFlag = True
            r = list(range(11,20))
            # r = list(range(11,170000))
    else:
            shuffleFlag = False
            r = list(range(170000,171724))
            # r = list(range(30,40))

    dataset = dataset_1in1_noImg(iRange=r,mode = dataMode)
    shuffleFlag = False
    data_loader = data.DataLoader(
        dataset, batch_size=batchSize, shuffle=shuffleFlag)
    datasize = len(dataset)
    return data_loader, datasize

def standardization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

# 获取本类型数据
class dataset_1in1_noImg(data.Dataset):
    # 读取数据集
    def __init__(self, iRange=range(11, 17), mode='abs', samplingMode='default', reduceMode="", staticRandom=False):
        # -----------------------------------------------------------------测试加载数据-------------------------------------------------------------------------
        
        # matPath = "/home/cj/CSMRI_0325/data/train/pngFormat2/"
        # matPath = "/home/cj/CSMRI_0325/data/val1/warm_up/warm_up/"
        # 
        # ！！！！改读取数据路径
        # matPath = "/home/cj/CSMRI_0325/data/val2/brain/"
        matPath = "/home/cj/CSMRI_0325/data/val3/final/abdomen/"
        # ylist是label，原图，mlist是mask，subk是降采样后的k_data
        self.mode = mode
        self.nameList = []
        self.CSkList = []
        self.PngList = []
        self.maskList = []
        index = 0
        # 1811,18516 170000,171724 1811,1850

        # ！！！！改读取数据范围
        for i in range(411,517):
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

            # CS_K_Data ---> png,复数二维图
            res = mDic['CS_K_Data']           
            fshift = np.fft.ifftshift(res) 
            img_back = np.fft.ifft2(fshift)
            # img_back = np.abs(img_back)
            w,h = img_back.shape
            png = np.zeros((2,w,h))
            png[0] = img_back.real
            png[1] = img_back.imag
            

            # show png_abs
            # 这个图片是反的
            # img_abs = np.abs(img_back)
            # img_abs[img_abs < 0] = 0
            # img_abs = standardization(img_abs)*255
            # png_abs = Image.fromarray(img_abs)
            # png_abs = png_abs.convert('L')
            # png_abs.save('resoutput/png_abs'+ name.split('.')[0]+'.png')          
 

            mask = mDic['mask'].astype(np.float32)
            mask = np.fft.ifftshift(mask)

            # 加载个IM保存一下图片
            # IM = mDic['IM']
            # IM = standardization(IM)*255
            # img = Image.fromarray(IM)
            # img = img.convert('L')
            # img.save('test2/IM_'+ name.split('.')[0]+'.png')
            
            self.nameList.append(name)
            self.CSkList.append(ckd)
            self.PngList.append(png)
            self.maskList.append(mask)
            index += 1

# --------------------------------------------------------------测试加载数据------------------------------------------------------


        # ---------------------训练加载数据--------------------------------------------
        # matPath = "/home/cj/CSMRI_0325/data/train/pngFormat2/"
        # self.mode = mode
        # self.cskList = []
        # self.yList = []
        # self.mList = []
        # self.subkList = []
        # self.nameList = []
        # self.PngList = []
        # index = 0
        # mDic = sio.loadmat(fakeRandomPath_15)
        # miList = mDic['RAll']
        
        
        # for i in iRange:
        #     path = matPath + "DATA_"+str(i) + ".mat"
        #     if os.path.isfile(path) == False:
        #         # print(path + " 不存在 ")
        #         continue
        #     mDic = sio.loadmat(path)
        #     name = "DATA_"+str(i) + ".mat"


        #     # ----------------------------------------------------------------------
        #     # CS_K_Dat --> subF
        #     # 网络输入的是subf，就是下采样的k空间数据，但是是iffshift后的
        #     # 所以ckd需要ifftshift转换一下,转换后的复数数据转成实数
        #     CS_K_Data = mDic['CS_K_Data']
        #     CS_K_Data = np.fft.ifftshift(CS_K_Data)
        #     w,h = CS_K_Data.shape
        #     ckd = np.zeros((2,w,h))
        #     ckd[0] = CS_K_Data.real
        #     ckd[1] = CS_K_Data.imag
        #     # 注意后面需要permute一下，现在的维度为(2,w,h)

        #     # CS_K_Data ---> png,复数二维图
        #     res = mDic['CS_K_Data']           
        #     fshift = np.fft.ifftshift(res) 
        #     img_back = np.fft.ifft2(fshift)
        #     # img_back = np.abs(img_back)
        #     w,h = img_back.shape
        #     png = np.zeros((2,w,h))
        #     png[0] = img_back.real
        #     png[1] = img_back.imag
            

        #     # show png_abs
        #     img_abs = np.abs(img_back)
        #     img_abs[img_abs < 0] = 0
        #     img_abs = standardization(img_abs)*255
        #     png_abs = Image.fromarray(img_abs)
        #     png_abs = png_abs.convert('L')
        #     png_abs.save('resoutput/png_abs'+ name.split('.')[0]+'.png')          
 

        #     mask = mDic['mask'].astype(np.float32)
        #     mask = np.fft.ifftshift(mask)

        #     # K-Data不需要的  复数
        #     K_Data = mDic['K_Data'].astype(np.float32)
        #     fshift = np.fft.ifftshift(K_Data)
        #     K_Data = fshift
        #     kd = transform.resize(K_Data,(256,256))

        #     # IM
        #     IM = (mDic['IM'].astype(np.float32))/255
        #     # standardization()
        #     # IM = standardization(IM)
        #     w,h = IM.shape
        #     # label = transform.resize(IM,(256,256))
        #     y = np.zeros((2,w,h))
        #     y[0]=IM

        #     # self.nameList.append(name)
        #     # self.CSkList.append(ckd)
        #     # self.PngList.append(png)
        #     # self.maskList.append(mask)
        #     # index +=1
        #     # name
        #     self.nameList.append(name)
        #     # CSKD
        #     self.cskList.append(ckd)
        #     # IM
        #     self.yList.append(y)
        #     # mask
        #     self.mList.append(mask)
        #     # k-data
        #     self.subkList.append(kd)
        #     # 欠采样图片
        #     self.PngList.append(png)
        #     index += 1
        #     # ----------------------------------------------------------------------
            
        #     # 复数\
        #     # CS_K_Data = mDic['CS_K_Data']
        #     # fshift = np.fft.ifft2(CS_K_Data)
            
            
        #     # # 下采样图片
        #     # img_back = np.fft.ifft2(fshift)
        #     #              # 出来的是复数，取绝对值，转化成实数
        #     # img_back = np.abs(img_back)
        #     # img_back[img_back < 0] = 0
        #     # new_arr = standardization(img_back)
        #     # png = transform.resize(new_arr,(256,256))
        #     # P = np.zeros((2,256,256))
        #     # P[0] = png

        #     # # png做傅里叶变化
        #     # subf = np.fft.fft2(png)
        #     # csk = subf.view('(2,)float')
            
        #     # # GT图
        #     # # IM 是整数型uint，原图的图像也为uint
        #     # IM = (mDic['IM'].astype(np.float32))/255
        #     # # standardization()
        #     # IM = standardization(IM)
        #     # label = transform.resize(IM,(256,256))
        #     # y = np.zeros((2,256,256))
        #     # y[0]=label


        #     # # K-Data不需要的  复数
        #     # K_Data = mDic['K_Data'].astype(np.float32)
        #     # fshift = np.fft.ifftshift(K_Data)
        #     # K_Data = fshift
        #     # kd = transform.resize(K_Data,(256,256))

        #     # # mask
        #     # # 0,1矩阵
        #     # mask = mDic['mask'].astype(np.float32)    
        #     # m = np.fft.ifftshift(mask)     
        #     # m = transform.resize(m,(256,256))
            
           

        #     # label *= 255  # 变换为0-255的灰度值
        #     # label0 = Image.fromarray(label)
        #     # label0 = label0.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
        #     # label0.save("train_test/label_"+name[0].split('.')[0]+'_img2.png')
     
        #     # self.nameList.append(name)
        #     # # CSKD
        #     # self.cskList.append(csk)
        #     # # IM
        #     # self.yList.append(y)
        #     # self.mList.append(m)
        #     # self.subkList.append(kd)
        #     # self.PngList.append(P)
        #     # index += 1
# ------------------------------------------------------训练数据加载--------------------------------------------------
    #                 dataloader将dataset封装成一个迭代器
    # dataloder的迭代器返回的是getitem里的内容
    # label就是原图



    def __getitem__(self, index):
        # i = index
        # mode = self.mode
        # name = self.nameList[i]# 名字
        # label = self.yList[i]  # IM
        # mask = self.mList[i]  # mask
        # png = self.PngList[i]  # K_data
        # CSk = self.cskList[i]  # CS_k_DATA
        # return mode,name, label, mask, png, CSk
        # ------------------------------------------------------------------------------
        i = index
        mode = self.mode
        name = self.nameList[i]
        subf = self.CSkList[i]
        png = self.PngList[i]
        mask = self.maskList[i]
        return mode,name,subf,png,mask

    def __len__(self):
        return len(self.nameList)


# if __name__ == '__main__':
#     a,b = getDataloader()
#     # print("batchsize = "+ str(a))
#     for  label, mask,subF, CSk in a:
#          print(type(label))
