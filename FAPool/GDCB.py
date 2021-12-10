#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Feng Li
## School of Computer Science & Engineering, South China University of Technology
## Email: csfengli@mail.scut.edu.cn
## Copyright (c) 2019
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
Created on Mon Mar 18 11:57:58 2019

@author: feng
"""
import torch
import torch.nn as nn
import math
from torch.autograd import Variable

class GDCB(nn.Module):
    def __init__(self,mfs_dim=25,nlv_bcd=6):
        super(GDCB,self).__init__()
        self.mfs_dim=mfs_dim
        self.nlv_bcd=nlv_bcd
        self.pool=nn.ModuleList()
        
        for i in range(self.nlv_bcd-1):
            #we set different mode by select "sliding(stride=1)"/"disjoint(stride=kernel_size)"
            self.pool.add_module(str(i),nn.MaxPool2d(kernel_size=i+2,stride=(i+2)//2))#sliding
        self.ReLU = nn.ReLU()
    def forward(self,input):
        tmp=[]
        for i in range(self.nlv_bcd-1):
            output_item=self.pool[i](input)
            tmp.append(torch.sum(torch.sum(output_item,dim=2,keepdim=True),dim=3,keepdim=True))
        output=torch.cat(tuple(tmp),2)#why 0 appear
#        print(output)
        output=torch.log2(self.ReLU(output)+1)
        X=[-math.log(i+2,2) for i in range(self.nlv_bcd-1)]
        # X=Variable(torch.tensor(X).cuda())
        X = torch.tensor(X).to(output.device)
        X=X.view([1,1,X.shape[0],1])
        meanX = torch.mean(X,2,True)
        meanY = torch.mean(output,2,True)
        Fracdim = torch.div(torch.sum((output-meanY)*(X-meanX),2,True),torch.sum((X-meanX)**2,2,True))
        # Fracdim=meanY-b*meanX
        return Fracdim    
            
# model=BoxFracDim()
# model.cuda()
# inputsample=Variable(torch.rand([1,25,280,280]).cuda(),requires_grad=False)
# Fracdim=model(inputsample)
# print(Fracdim.size())        
            