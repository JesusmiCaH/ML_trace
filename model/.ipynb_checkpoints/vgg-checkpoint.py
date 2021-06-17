import torch
import torch.nn as nn

class VGG19_for_deep_style(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1=nn.Conv2d(3,64,3,1,1)
        self.relu1_1=nn.ReLU(True)
        self.conv1_2=nn.Conv2d(64,64,3,1,1)
        self.relu1_2=nn.ReLU(True)

        self.aPool1to2=nn.MaxPool2d(2,2)

        self.conv2_1=nn.Conv2d(64,128,3,1,1)
        self.relu2_1=nn.ReLU(True)
        self.conv2_2=nn.Conv2d(128,128,3,1,1)
        self.relu2_2=nn.ReLU(True)

        self.aPool2to3=nn.MaxPool2d(2,2)

        self.conv3_1=nn.Conv2d(128,256,3,1,1)
        self.relu3_1=nn.ReLU(True)
        self.conv3_2=nn.Conv2d(256,256,3,1,1)
        self.relu3_2=nn.ReLU(True)
        self.conv3_3=nn.Conv2d(256,256,3,1,1)
        self.relu3_3=nn.ReLU(True)        
        self.conv3_4=nn.Conv2d(256,256,3,1,1)
        self.relu3_4=nn.ReLU(True)

        self.aPool3to4=nn.MaxPool2d(2,2)

        self.conv4_1=nn.Conv2d(256,512,3,1,1)
        self.relu4_1=nn.ReLU(True)
        self.conv4_2=nn.Conv2d(512,512,3,1,1)
        self.relu4_2=nn.ReLU(True)
        self.conv4_3=nn.Conv2d(512,512,3,1,1)
        self.relu4_3=nn.ReLU(True)        
        self.conv4_4=nn.Conv2d(512,512,3,1,1)
        self.relu4_4=nn.ReLU(True)

        self.aPool4to5=nn.MaxPool2d(2,2)

        self.conv5_1=nn.Conv2d(512,512,3,1,1)
        self.relu5_1=nn.ReLU(True)
        self.conv5_2=nn.Conv2d(512,512,3,1,1)
        self.relu5_2=nn.ReLU(True)
        self.conv5_3=nn.Conv2d(512,512,3,1,1)
        self.relu5_3=nn.ReLU(True)        
        self.conv5_4=nn.Conv2d(512,512,3,1,1)
        self.relu5_4=nn.ReLU(True)
        self.aPool_out=nn.MaxPool2d(2,2)
    
    def forward(self,x,out_layers):
        response={}  #用字典存放每一层的输出值，因为风格迁移的时候需要用每一层的响应算风格损失。
        response['1_0']=x
        response['1_1']=self.relu1_1(self.conv1_1(response['1_0']))
        response['1_2']=self.relu1_2(self.conv1_2(response['1_1']))

        response['2_0']=self.aPool1to2(response['1_2'])
        response['2_1']=self.relu2_1(self.conv2_1(response['2_0']))
        response['2_2']=self.relu2_2(self.conv2_2(response['2_1']))

        response['3_0']=self.aPool2to3(response['2_2'])
        response['3_1']=self.relu3_1(self.conv3_1(response['3_0']))
        response['3_2']=self.relu3_2(self.conv3_2(response['3_1']))
        response['3_3']=self.relu3_3(self.conv3_3(response['3_2']))
        response['3_4']=self.relu3_4(self.conv3_4(response['3_3']))

        response['4_0']=self.aPool3to4(response['3_4'])
        response['4_1']=self.relu4_1(self.conv4_1(response['4_0']))
        response['4_2']=self.relu4_2(self.conv4_2(response['4_1']))
        response['4_3']=self.relu4_3(self.conv4_3(response['4_2']))
        response['4_4']=self.relu4_4(self.conv4_4(response['4_3']))

        response['5_0']=self.aPool3to4(response['4_4'])
        response['5_1']=self.relu5_1(self.conv5_1(response['5_0']))
        response['5_2']=self.relu5_2(self.conv5_2(response['5_1']))
        response['5_3']=self.relu5_3(self.conv5_3(response['5_2']))
        response['5_4']=self.relu5_4(self.conv5_4(response['5_3']))

        return [response[o_lay] for o_lay in out_layers]