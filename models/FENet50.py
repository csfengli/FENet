##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Feng Li
## School of Computer Science & Engineering, South China University of Technology
## Email: csfengli@mail.scut.edu.cn
## Copyright (c) 2019
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models

import encoding
import torchvision.models as resnet
from FAPool.FAP import FAP

class Net(nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()
        nclass=opt.n_classes
        
        # self.dep = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                    # bias=False))

        # copying modules from pretrained models
        if opt.backbone.lower() == 'resnet18':
            self.backbone = models.resnet18(pretrained=opt.use_pretrained)
        elif opt.backbone.lower() == 'resnet50':
            self.backbone = models.resnet50(pretrained=opt.use_pretrained)
        elif opt.backbone.lower() == 'resnet101':
            self.backbone = models.resnet101(pretrained=opt.use_pretrained)
        elif opt.backbone.lower() == 'resnet152':
            self.backbone = models.resnet152(pretrained=opt.use_pretrained)#, dilated=False
        else:
            raise Exception('unknown backbone: {}'.format(opt.backbone.lower()))

        self.dim = opt.dim

        self.conv_before_mfs=nn.Sequential(
            nn.Conv2d(2048,3,kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU())
        self.mfs=FAP(opt,D=1,K=self.dim)

        self.pool = nn.Sequential(
            nn.AvgPool2d(7),
            encoding.nn.View(-1, 2048),
            nn.Linear(2048, self.dim*3),
            nn.BatchNorm1d(self.dim*3),
        )        

        self.fc= nn.Sequential(
            encoding.nn.Normalize(),
            nn.Linear((self.dim*3)*(self.dim*3), 128),
            encoding.nn.Normalize(),
            nn.Linear(128, nclass)#nclass
            )
        self.UP = nn.ConvTranspose2d(2048,2048,3,2,groups=2048)#group operation is important for compact model size 



    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            var_input = x 
            while not isinstance(var_input, Variable):
                var_input = var_input[0]
            _, _, h, w = var_input.size()
        else:
            raise RuntimeError('unknown input type: ', type(x))

        # if self.backbone == 'resnet18' or self.backbone == 'resnet50' or self.backbone == 'resnet101' \
        #     or self.backbone == 'resnet152':
            # pre-trained ResNet feature
        x = self.backbone.conv1(x)
        # x = self.dep(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        up = self.UP(x)
        # import pdb
        # pdb.set_trace()
        # print(up.shape)[32, 512, 15, 15]
        c = self.conv_before_mfs(up)
        # c = self.conv_before_mfs(x)
        c0=c[:,0,:,:].unsqueeze_(1)
        c1=c[:,1,:,:].unsqueeze_(1)
        c2=c[:,2,:,:].unsqueeze_(1)

        fracdim0=self.mfs(c0).squeeze_(-1).squeeze_(-1)
        fracdim1=self.mfs(c1).squeeze_(-1).squeeze_(-1)
        fracdim2=self.mfs(c2).squeeze_(-1).squeeze_(-1)
        x1 = self.pool(x) #avg pooling       
        x2 = torch.cat((fracdim0,fracdim1,fracdim2),1)
        x1 = x1.unsqueeze(1).expand(x1.size(0),x2.size(1),x1.size(-1))  #Bilinear Models
        x3 = x1*x2.unsqueeze(-1) #Bilinear Models
        x3 = x3.view(-1,x1.size(-1)*x2.size(1)) #Bilinear Models
        x  = self.fc(x3)

        # else:
        #     x = self.backbone(x)
        return x


def test():
    net = Net(nclass=23).cuda()
    print(net)
    
    test=net.cpu().state_dict()
    print('=============================================================================')
    for key,v in test.items():
        print(key)
    net.cuda()
    x = Variable(torch.randn(1,3,224,224)).cuda()
    y = net(x)
    print(y)
    params = net.parameters()
    sum = 0
    for param in params:
        sum += param.nelement()
    print('Total params:', sum)


if __name__ == "__main__":
    test()
