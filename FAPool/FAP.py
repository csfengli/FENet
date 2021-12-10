##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Feng Li
## School of Computer Science & Engineering, South China University of Technology
## Email: csfengli@mail.scut.edu.cn
## Copyright (c) 2019
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import scipy.io as scio
# from DeepMFS.DeepMFS_copy_new import DeepMFSGenerator
from FAPool.LDEB import LDEB
from FAPool.PGB import PGB
from FAPool.GDCB import GDCB

import torch
import torch.nn as nn
from torch.autograd import Variable

class FAP(nn.Module): #paper20190804-1707
    def __init__(self,opt,D=3,K=32):
        super(FAP,self).__init__()
        self.D=D
        self.K=K
        # self.dim=dim
        self.deepmfs = nn.Sequential(
            LDEB(),
            nn.BatchNorm2d(1),)
        if opt.histogram_type == 'encoding':
            self.deephist=nn.Sequential(
                PGB(D=self.D,K=self.K),
                nn.BatchNorm2d(self.K),)
        elif opt.histogram_type == 'RBFpooling':
            self.deephist=nn.Sequential(
                PGB(self.D, 5, 2, opt.dim, stride=1, pading=2,
                             normalize_bins=True, normalize_count=False),
                nn.BatchNorm2d(self.K),)
        else:
            raise Exception('Unexpected type of histogram of {}'.format(opt.histogram_type))
        # self.deephist = DeepSoftHist(D=self.D,K=self.K)        
        self.boxfracdim = nn.Sequential(
            GDCB(mfs_dim=self.K,nlv_bcd=6),
        )

    def forward(self,input):
        minsample=torch.min(input)
        maxsample=torch.max(input)
        input=1+(input-minsample)/(torch.clamp(maxsample-minsample,10e-8,10e8))*255
        Densemap=self.deepmfs(input)             
        #将数据的range变换至[0，dim]区间
        # maxD=torch.max(Densemap)
        # minD=torch.min(Densemap)
        # scaleDensemap=((Densemap-minD)/(maxD-minD))*self.dim        
        # histmap=self.deephist(torch.floor(scaleDensemap))
        # print(self.D)
        # print(self.K)
        histmap=self.deephist(Densemap)
        FracDim=self.boxfracdim(histmap)           
        #[B,32,1,1]
        # 可视化FracDim
        # FracDim=FracDim[0,:,0,0].cpu()
        # print(FracDim)
        # plt.plot(FracDim.detach().numpy())
        return FracDim
