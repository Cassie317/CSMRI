import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
import os
import numpy as np
import matplotlib.image as mpimg
from PIL import Image
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
import configparser
import time


from util import imshow, img2f, f2img, kspace_subsampling, Recoder, paramNumber, kspace_subsampling_pytorch, imgFromSubF_pytorch,  png2kspace
from network import getNet, getLoss, getOptimizer
from dataProcess import getDataloader


class core():
    def __init__(self, configPath, isEvaluate=False):
        self.config = configparser.ConfigParser()
        self.config.read(configPath)
        self.num = 0
        self.batchSize = int(self.config['train']['batchSize'])
        self.epoch = int(self.config['train']['epoch'])
        self.LR = float(self.config['train']['learningRate'])

        self.saveEpoch = int(self.config['log']['saveEpoch'])
        self.maxSaved = int(self.config['log']['maxSaved'])

        self.useCuda = self.config.getboolean('general', 'useCuda')
        self.needParallel = self.config.getboolean('general', 'needParallel')
        if(self.useCuda):
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

        self.mode = self.config['general']['mode']
        print("#Result Path:"+self.config['general']['path'])
        print("#Create network:"+self.config['general']['netType'])
        print("#Mode:"+self.mode)

        listDeviceStr = self.config['general']['device']
        listDevice = []
        for i in range(len(listDeviceStr)):
            if(listDeviceStr[i] == '1'):
                listDevice.append(str(i))
            devicesStr = ','.join(listDevice)
        os.environ["CUDA_VISIBLE_DEVICES"] = devicesStr

        self.net = getNet(self.config['general']['netType']).type(self.dtype)

        assert len(listDevice) > 0, "No device is selected"
        if(self.needParallel):
            print('parallel device:'+str(listDevice))
            self.net = nn.DataParallel(self.net, device_ids=list(
                range(len(listDevice)))).type(self.dtype)
        self.lossForward = getLoss(
            self.config['general']['lossType']).type(self.dtype)
        paramNum = paramNumber(self.net)
        print("#Number of network parameters:%d" % paramNum)

        self.crossValid = int(self.config['general']['crossValid'])

        if(not isEvaluate):
            # print("**************************************************enter")
            self.trainloader, self.trainsetSize = getDataloader(
                self.config['general']['dataType'], 'train', self.batchSize, self.crossValid)
        self.testloader, self.testsetSize = getDataloader(
            self.config['general']['dataType'], 'test', 1, self.crossValid)

        self.isFastMRI = 'FastMRI' in self.config['general']['dataType']
        if(self.isFastMRI):
            print("data range for PSNR: 0~12 (estimated)")
        else:
            print("data range for PSNR: 0~1")

        if('FromLSF' in configPath):
            self.record = Recoder(
                'resultFromLSF/' + self.config['general']['path'], self.saveEpoch*self.maxSaved)
        else:
            self.record = Recoder(
                self.config['general']['path'], self.saveEpoch*self.maxSaved)

        if(not isEvaluate):
            self.config.write(open(self.record.rootPath+"/config.ini", "w"))
            self.record.logNet(self.net)
            self.record.log("### Number of network parameters:%d" % paramNum)

        # self.optimizer = getOptimizer(self.net.parameters(), self.config['train']['optimizer'], self.LR)
        if(self.config['train']['optimizer'] == 'Adam_wd'):
            self.weightDecay = float(self.config['train']['weightDecay'])
        elif(self.config['train']['optimizer'] == 'Adam_DC_CNN'):
            self.weightDecay = 0.0000001
        elif(self.config['train']['optimizer'] == 'Adam_RDN'):
            self.weightDecay = 0.0001
        else:
            self.weightDecay = 0
        print('#Optimizer: '+self.config['train']['optimizer'] +
              ' LR = %.2e weightDecay = %.2e' % (self.LR, self.weightDecay))
        self.ckp_epoch = 0

    def train(self, epoch=0, listTarget=[0, 1], forwardStage=1, need_evaluate=True):
        if(epoch == 0):
            epoch = self.epoch
        self.optimizer = getOptimizer(self.net.parameters(
        ), self.config['train']['optimizer'], self.LR, self.weightDecay)
        msg = "start training: epoch = %d" % (epoch)
        self.record.log(msg)
        print(msg)
        for j in range(1, epoch + 1):
            self.net.train()
            i = 0
            total_loss = 0
            for mode, name, label, mask, png, CSk in self.trainloader:
                # self.optimizer.zero_grad()
                # # label获取满采样图片
                # # png 为降采样图片
                # # CSk为降采样k-data

                netLabel = Variable(label).type(self.dtype)
                mask_var = Variable(mask).type(self.dtype)
                netInput = Variable(png).type(self.dtype)
                subF0 = Variable(CSk).type(self.dtype)
                subF0 = subF0.permute(0,2,3,1)

                complexFlag = (mode[0] == 'complex')
                mask_var = Variable(mask).type(self.dtype)
                # 将满采样图片和mask做下面函数，得到降采样图的k-data
                subF = kspace_subsampling_pytorch(netLabel,mask_var)
                # 得到的subf是和提供的CSK一致的，即mask一致，netInput一致
                # 将降采样图k-data转化为图片
                netInput = imgFromSubF_pytorch(subF,complexFlag)
                # 将降采样图和k-data送入网络
                netOutput = self.net(netInput, subF, mask_var)
                # 满采图和网络输出做loss
                loss = self.lossForward(netOutput, netLabel)
                # -----------------------------------------------------------------------------------------------
                y = netLabel.cpu().data.numpy()
                y = y[0]
                y[0] = self.standardization(y[0])*255
                # y[0] *= 255  # 变换为0-255的灰度值
                y1 = Image.fromarray(y[0])
                y1 = y1.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
                y1.save("test2/label1_"+name[0].split('.')[0]+'_y.png')

                y = subF.cpu().data.numpy()
                y = y[0]
                y = self.standardization(y[:,:,0])*255
                # y[0] *= 255  # 变换为0-255的灰度值
                y1 = Image.fromarray(y)
                y1 = y1.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
                y1.save("test2/subF1_"+name[0].split('.')[0]+'_y.png')

                # subf逆

                x = netInput.cpu().data.numpy()
                x = x[0]
                x[0] = self.standardization(x[0])*255
                # y[0] *= 255  # 变换为0-255的灰度值
                x1 = Image.fromarray(x[0])
                x1 = x1.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
                x1.save("test2/input1_"+name[0].split('.')[0]+'_y.png')

                y = netOutput.cpu().data.numpy()
                y = y[0]
                y[0][y[0] < 0] = 0
                y[0] = self.standardization(y[0])*255
                # y[0] *= 255  # 变换为0-255的灰度值                
                y1 = Image.fromarray(y[0])
                y1 = y1.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
                y1.save("test2/output1_"+name[0].split('.')[0]+'_y.png')


                # ----------------------------------------------------------------------------------------------
                
                # netLabel = Variable(label).type(self.dtype)
                # netInput = Variable(png).type(self.dtype)
                # subF = Variable(CSk).type(self.dtype)
                # mask_var = Variable(mask).type(self.dtype)
                # subF = subF.permute(0,2,3,1) 
                # complexFlag = (mode[0] == 'complex')
                # netOutput = self.net(netInput, subF, mask_var)  
                # loss = self.lossForward(netOutput, netLabel)

                # # ----------------------------------------------------------------------------------------------

                # y = netLabel.cpu().data.numpy()
                # y = y[0]
                # y[0] = self.standardization(y[0])*255
                # # y[0] *= 255  # 变换为0-255的灰度值
                # y1 = Image.fromarray(y[0])
                # y1 = y1.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
                # y1.save("test2/label2_"+name[0].split('.')[0]+'_y.png')

                # y = subF0.cpu().data.numpy()
                # y = y[0]
                # y = self.standardization(y[:,:,0])*255
                # # y[0] *= 255  # 变换为0-255的灰度值
                # y1 = Image.fromarray(y)
                # y1 = y1.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
                # y1.save("test2/subF2_"+name[0].split('.')[0]+'_y.png')

                # # subf逆

                # x = netInput.cpu().data.numpy()
                # x = x[0]
                # x[0] = self.standardization(x[0])*255
                # # y[0] *= 255  # 变换为0-255的灰度值
                # x1 = Image.fromarray(x[0])
                # x1 = x1.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
                # x1.save("test2/input2_"+name[0].split('.')[0]+'_y.png')

                # y = netOutput.cpu().data.numpy()
                # y = y[0]
                # y[0][y[0] < 0] = 0
                # y[0] = self.standardization(y[0])*255
                # # y[0] *= 255  # 变换为0-255的灰度值                
                # y1 = Image.fromarray(y[0])
                # y1 = y1.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
                # y1.save("test2/output2_"+name[0].split('.')[0]+'_y.png')
                # ---------------------------------------------------------------------------------------------
                
                loss = self.lossForward(netOutput, netLabel)

                loss.backward()  # 计算梯度
                total_loss = total_loss + loss.item()*label.shape[0]
            
                self.optimizer.step()  # 反向传播，更新网络参数
                self.optimizer.zero_grad()  # 清空梯度

                i += label.shape[0]
                del mode, label, mask, png, CSk


            print('Epoch %05d [%04d/%04d] loss %.8f' %
                      (j+self.ckp_epoch, i, self.trainsetSize, loss.item()), '\r', '')



            self.record.log_train(j+self.ckp_epoch, total_loss/i)

            if j % self.saveEpoch == 0:
                print('Epoch %05d [%04d/%04d] loss %.8f SAVED' %
                      (j+self.ckp_epoch, i, self.trainsetSize, total_loss/self.trainsetSize))
                if need_evaluate:
                    l, p1, p2, s1, s2 = self.validation()
                    self.record.log_valid(j+self.ckp_epoch, l, p1, p2, s1, s2)
                    self.record.log(
                        "Evaluate psnr(before|after) =  %.2f|%.2f ssim = %.4f|%.4f" % (p1, p2, s1, s2))
                self.record.write_to_file(self.net.state_dict(), False)
        self.ckp_epoch += epoch
        self.record.write_to_file(self.net.state_dict(), True)

    def testValue(self, mode, name, label, mask,png,CSk):
        result = self.test(mode, name, label, mask,png,CSk)

        return result['loss'], result['psnr1'], result['psnr2'], result['ssim1'], result['ssim2']

    def test(self, mode, name, label, mask,png,CSk):
        y = label.numpy()
        netLabel = Variable(label).type(self.dtype)
        
        if(self.mode != 'inNetDC'):
            assert False, 'only for inNetDC mode'
        else:
            # netInput = Variable(png).type(self.dtype)
            # mask_var = Variable(mask).type(self.dtype)
            # subF = kspace_subsampling_pytorch(netInput,mask_var)
            # # subF = kspace_subsampling_pytorch(netLabel, mask_var)
            # complexFlag = (mode[0] == 'complex')
            # # netInput = imgFromSubF_pytorch(subF, complexFlag)

            mask_var = Variable(mask).type(self.dtype)
            subF = kspace_subsampling_pytorch(netLabel,mask_var)
            complexFlag = (mode[0] == 'complex')
            netInput = imgFromSubF_pytorch(subF,complexFlag)

        # 保存图片
        x = netInput.cpu().data.numpy()
        x = x[0]
        x[0] = self.standardization(x[0])*255
        # y[0] *= 255  # 变换为0-255的灰度值
        x1 = Image.fromarray(x[0])
        x1 = x1.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
        x1.save("train_test/input_"+name[0].split('.')[0]+'_y.png')


        # 检测一下netOutput_np的数据类型，保存为.npy
        netOutput = self.net(netInput, subF, mask_var)
        
        y1 = netOutput.cpu().data.numpy()
        y1 = y1[0]
        y1[0][y1[0] < 0] = 0
        y1[0] = self.standardization(y1[0])*255
        # y[0] *= 255  # 变换为0-255的灰度值                
        y2 = Image.fromarray(y1[0])
        y2 = y2.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
        y2.save("train_test/output_"+name[0].split('.')[0]+'_y.png')

        loss = self.lossForward(netOutput, netLabel)
        netOutput_np = netOutput.cpu().data.numpy()
        img1 = netOutput_np[0, 0:1].astype('float64')
        # im = Image.fromarray((img1[0]*255))
        # scipy.misc.imsave(name, img1[0])
        if(netOutput_np.shape[1] == 2):
            netOutput_np = abs(netOutput_np[:, 0:1]+netOutput_np[:, 1:2]*1j)
        img2 = netOutput_np[0].astype('float64')
        y = y[0]
        y2 = y[0:1]
        y2 = self.standardization(y2)

        img1 = np.clip(img1, 0, 1)
        img2 = np.clip(img2, 0, 1)

        if(self.isFastMRI):
            psnrBefore = psnr(y2, img1, 12)
            psnrAfter = psnr(y2, img2, 12)
        else:
            psnrBefore = psnr(y2, img1)
            psnrAfter = psnr(y2, img2)


        ssimBefore = ssim(y2[0], img1[0])
        ssimAfter = ssim(y2[0], img2[0])

        return {"loss": loss.item(), "psnr1": psnrBefore, "psnr2": psnrAfter, "ssim1": ssimBefore, "ssim2": ssimAfter, "result1": img1, "result2": img2, 'label': y2}

    def validation(self, returnList=False):
        self.net.eval()
        i = 0
        totalLoss = 0
        lpsnr1 = []
        lpsnr2 = []
        lssim1 = []
        lssim2 = []
        psnr1 = 0
        psnr2 = 0
        ssim1 = 0
        ssim2 = 0
        for mode, name, label, mask, png, CSk in self.testloader:
            loss0, psnrA, psnrB, ssimA, ssimB = self.testValue(
                mode, name, label, mask,png,CSk)

            totalLoss += loss0
            psnr1 += psnrA
            psnr2 += psnrB
            ssim1 += ssimA
            ssim2 += ssimB
            lpsnr1.append(psnrA)
            lpsnr2.append(psnrB)
            lssim1.append(ssimA)
            lssim2.append(ssimB)
            i += 1
            print('Evaluating %04d psnr(before|after) =  %.2f|%.2f ssim = %.4f|%.4f' % (
                i, psnrA, psnrB, ssimA, ssimB), '\r', '')
        print('Evaluating %04d psnr(before|after) =  %.2f|%.2f ssim = %.4f|%.4f Done' % (
            i, psnr1/i, psnr2/i, ssim1/i, ssim2/i))
        if(returnList):
            return lpsnr1, lpsnr2, lssim1, lssim2
        return totalLoss/i, psnr1/i, psnr2/i, ssim1/i, ssim2/i

    def standardization(self,data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    def myvalidation(self, returnList=False):
        self.net.eval()
        t0 = time.time()
        for mode, name, subF,label,mask in self.testloader:
            complexFlag = (mode[0] == 'complex')
            netInput = Variable(label).type(self.dtype)
            subF = Variable(subF).type(self.dtype)
            mask_var = Variable(mask).type(self.dtype)
            subF = subF.permute(0,2,3,1)       
            # netInput2 = imgFromSubF_pytorch(subF,complexFlag)
            complexFlag = (mode[0] == 'complex')
            netOutput = self.net(netInput, subF, mask_var)
            
            x = netInput.cpu().data.numpy()
            x = x[0]
            x[0] = self.standardization(x[0])*255
            x1 = Image.fromarray(x[0])
            x1 = x1.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
            x1.save("resinput/test_input_"+name[0].split('.')[0]+'_y.png')

            m = mask_var.cpu().data.numpy()
            m = m[0]
            m = self.standardization(m)*255
            m = Image.fromarray(m)
            m = m.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
            m.save("resinput/mask_"+name[0].split('.')[0]+'_y.png')

# 单个图像输出
            y1 = netOutput.cpu().data.numpy()
            y1 = y1[0]
            img_back = np.abs(y1[0])
            img_back[img_back < 0] = 0
            y1[0] = self.standardization(img_back)*255               
            y2 = Image.fromarray(y1[0])
            y2 = y2.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
            outpath = "resoutput/abdomen/final_res_"+name[0].split('.')[0]+'_y.png'
            y2.save(outpath)
  
            resout = Image.open(outpath)
            resout.rotate(180).save(outpath)
            resout = Image.open(outpath)
            np.save("out_npy/abdomen/"+name[0].split('.')[0]+'.npy',resout)
            
            # 组合图像输出
            res = np.concatenate((y1[0],x[0]),axis = 1)          
            y2 = Image.fromarray(res)
            y2 = y2.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
            path = "resoutput/abdomen/resout1_"+name[0].split('.')[0]+'_y.png'
            y2.save(path)
            y2 = Image.open(path)
            y2.rotate(180).save(path)

        t1 = time.time()
        t = t1 - t0
        print("time :"+ str(t))
        



            

    def standardization(self, data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    def loadCkpt(self, expectedEpoch, isChecked=False):
        self.record.load_from_file(self.net, expectedEpoch, isChecked, False)
        self.ckp_epoch = expectedEpoch
