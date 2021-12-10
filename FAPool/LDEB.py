##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Feng Li
## School of Computer Science & Engineering, South China University of Technology
## Email: csfengli@mail.scut.edu.cn
## Copyright (c) 2019
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch
#import torchvision

import torch.nn as nn
from FAPool.padding_same_conv import Conv2d
from torch.autograd import Variable
import math
from FAPool.util import fspecial

class LDEB(nn.Module):
    def __init__(self,channel=1,nlv_dens=6):
        super(LDEB, self).__init__()
        self.conv=torch.nn.Sequential()
        self.channel=channel
        self.nlv_dens=nlv_dens
# nlv_dens=6
        
        for i in range(self.nlv_dens):
        # for i in range(self.nlv_dens):
            self.conv.add_module(str(i),Conv2d(self.channel,1,kernel_size=i+1,bias=False))
            # self.conv.add_module(str(i),Conv2d(self.channel,1,kernel_size=i+1,bias=False))
#            tmp1=self.conv[i].weight.data
#            self.conv[i].weight=nn.Parameter(self.conv[i].weight.data.normal_(i+1, (i+1)/2))
#            self.conv[i].weight.data.normal_(i+1, (i+1)/2)
            self.conv[i].weight=nn.Parameter(torch.FloatTensor(fspecial('gaussian',kernel_size=i+1,sigma=(i+1)/2)).repeat(1,self.channel,1,1),requires_grad=False)
            # self.register_parameter(,model[i].parameters)        
        self.ReLU = nn.ReLU()
    def forward(self,input):
        for i in range(self.nlv_dens):
            # print(input.dim())
            tmp=self.conv[i](input)*((i+1)**2)#input.dim()
            if i==0:
                output = tmp
            else:
                output = torch.cat((output,tmp),1)
        output=torch.log2(self.ReLU(output)+1)
        # Densemap=torch.mean(output, 1, keepdim=True)
        X=[math.log(i+1,2) for i in range(self.nlv_dens)]
        # X=Variable(torch.tensor(X).cuda(),requires_grad=False)
        X = torch.tensor(X).to(output.device)
        X=X.view(1,X.shape[0],1,1)
        meanX = torch.mean(X,1,True)
        meanY = torch.mean(output,1,True)
        Densemap = torch.div(torch.sum((output-meanY)*(X-meanX),1,True),torch.sum((X-meanX)**2,1,True))
        # print(torch.sum((X-meanX)**2,1,True))
        # Densemap = meanY-b*meanX
        return Densemap