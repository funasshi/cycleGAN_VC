import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from glob import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

#========================================================
#dataset用意
def load_data(dataset_name,trainA_name,trainB_name):
    import make_data
    Apath="datasets/"+dataset_name+"/"+trainA_name
    Bpath="datasets/"+dataset_name+"/"+trainB_name
    trainA=make_data.all_mcep(Apath)
    trainB=make_data.all_mcep(Bpath)
    return trainA,trainB

class Data(torch.utils.data.Dataset):
    def __init__(self,A,B):
        super(Data,self).__init__()
        self.A=A
        self.B=B
        self.len=min(A.size()[0],B.size()[0])
    def __len__(self):
        return self.len
    def __getitem__(self,index):
        index1,index2=index
        return self.A[index1],self.B[index2]

class Sampler(torch.utils.data.Sampler):
    def __init__(self,data_source,width,iteration):
        super(Sampler,self).__init__(data_source)
        self.data=data_source
        self.length=len(self.data)
        self.width=width
        self.iteration=iteration


    def __len__(self):
        return self.length//self.width

    def __iter__(self):
        self._check=0
        return self

    def __next__(self):
        if self._check > self.iteration: raise StopIteration
        index1=np.random.randint(0,self.length-self.width+1)
        index2=np.random.randint(0,self.length-self.width+1)
        self._check+=1
        return list(range(index1,index1+self.width)),list(range(index2,index2+self.width))

#========================================================
#テスト音声

def sample_data():#正しい
    sampleA_path="output/sample/sampleA.png"
    sampleA_img = load_img(sampleA_path,grayscale=False)
    sampleA_img=sampleA_img.resize((256,256))
    sampleA_img = np.uint8(sampleA_img)
    sampleA_img = sampleA_img /255
    sampleB_path="output/sample/sampleB.png"
    sampleB_img = load_img(sampleB_path,grayscale=False)
    sampleB_img=sampleB_img.resize((256,256))
    sampleB_img = np.uint8(sampleB_img)
    sampleB_img = sampleB_img /255
    return sampleA_img,sampleB_img

#========================================================
# 出力音声保存
def save(epoch,generatorAB,generatorBA):
    sampleA,sampleB=sample_data()
    if torch.cuda.is_available():
        generatorAB.to("cpu")
        generatorBA.to("cpu")
    plt.imsave("output/trueA/epoch_"+str(epoch)+".png",sampleA)
    sampleA=sampleA*2-1
    sampleA=numpy2tensor(sampleA)
    sampleA=sampleA.reshape((1,3,256,256))
    fakeB=generatorAB(sampleA)
    fakeB=fakeB.reshape((3,256,256))
    fakeB=tensor2numpy(fakeB)
    fakeB=(fakeB+1)/2
    plt.imsave("output/fakeB/epoch_"+str(epoch)+".png",fakeB)
    plt.imsave("output/trueB/epoch_"+str(epoch)+".png",sampleB)
    sampleB=sampleB*2-1
    sampleB=numpy2tensor(sampleB)
    sampleB=sampleB.reshape((1,3,256,256))
    fakeA=generatorBA(sampleB)
    fakeA=fakeA.reshape((3,256,256))
    fakeA=tensor2numpy(fakeA)
    fakeA=(fakeA+1)/2
    plt.imsave("output/fakeA/epoch_"+str(epoch)+".png",fakeA)
    if torch.cuda.is_available():
        generatorAB.cuda()
        generatorBA.cuda()



#========================================================
#進捗表示

def progress(p, l):
    sys.stdout.write("\rprocessing : %d %%" %(int(p * 100 / (l - 1))))
    sys.stdout.flush()
