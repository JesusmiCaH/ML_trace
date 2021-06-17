import numpy as np
import torch
import torch.nn as nn
import cv2
import os
import torchvision.transforms as transforms
from model import vgg
    
#图片读取与预处理
transform_pretreat=transforms.Compose(\
    [transforms.ToTensor(),
    transforms.Resize(620)]) 
def read_pic(path:str):
    read_img=cv2.imread(path)
    tensor_img=transform_pretreat(read_img)
    tensor_img=tensor_img.unsqueeze(0)
    pic_size=tuple(tensor_img.shape[2:])
    return tensor_img,pic_size #实验改过，记得回来

def get_noise_img(mean,square,size):
    noise=np.random.normal(mean,square,size+(3,))
    noise=np.clip(noise,0.,1.)
    noi_img=np.uint8(noise*255)
    return noi_img

#图片准备
style_path='imgs/chaos.jpg'
style_pic,pic_shape=read_pic(style_path)
content_path='mix.jpg'
content_pic,pic_shape=read_pic(content_path)
'''
fix_img=get_noise_img(0.53,1e-8,pic_shape)
fix_pic=transforms.ToTensor()(fix_img)
fix_pic=fix_pic.unsqueeze(0)
'''
fix_pic=content_pic.clone()

style_pic=style_pic.cuda()*255
content_pic=content_pic.cuda()*255
fix_pic=fix_pic.cuda()*255


fix_pic.requires_grad_(True)

#模型导入
train_vgg=vgg.VGG19_for_deep_style()
train_vgg.load_state_dict(torch.load('model/deep_dream.pth'))
train_vgg.requires_grad_(False)

train_vgg.cuda()

def GramMatrix(feature_map):
    #该函数的功能是返回一层响应的格拉姆矩阵，一层响应的数组大小必然是：1*feature map个数*宽*长
    #按照格拉姆矩阵的定义，矩阵的元素为原矩阵与其转置的矩乘，但是在风格迁移的实例中，原矩阵（4维）将被拉成三维，即图片像素呈一维排列。
    #因此这里实质上是将原来的2、3维看作矩阵进行求格拉姆矩阵的运算。
    one,filter_num,length,width=feature_map.shape
    vector_fm=feature_map.view(one,filter_num,length*width)
    return (vector_fm@vector_fm.transpose(1,2))/(length*width*filter_num)

style_layers=['1_1','2_1','3_2','4_1','5_1']
content_layer=['4_2']
together_layer=style_layers+content_layer

weight_style=[3e3*j for j in [2,3,1,3,4]]
weight_content=[1e1]
together_weight=weight_style+weight_content

style_target=[GramMatrix(map) for map in train_vgg(style_pic,style_layers)]
content_target=train_vgg(content_pic,content_layer)

#训练区
loss_fun=nn.MSELoss()
epoch=[0]
optim_style=torch.optim.LBFGS([fix_pic])
#LBFGS算法不用for...in...结构，因为step(closure)中会多次调用closure方法，以至于一次epoch中会触发很多次closure效果，不方便计数，因此用下面的方法。
while epoch[0]<=1500:
    def closure():
        optim_style.zero_grad()
        #原来的程序把train_vgg调用了两遍，由于每次调用会把整个神经网络跑一遍，因此相当于跑了两遍整个网络，会让效率降低一倍，不可取！
        predicted=train_vgg(fix_pic,together_layer)
        style_predicted=predicted[0:len(style_layers)]
        content_predicted=predicted[len(style_layers):]

        loss_style=sum([loss_fun([GramMatrix(map) for map in style_predicted][i],style_target[i])*weight_style[i]\
            for i in range(len(style_layers))])
        loss_content=sum([loss_fun(content_predicted[i],content_target[i])*weight_content[i]\
            for i in range(len(content_layer))])
        total_loss=loss_style+loss_content
        total_loss.backward()
        #用于计数的是一个列表，因为单个数字无法传如函数作用域，因此用列表。
        epoch[0]+=1
        if epoch[0]%50==0:
            print(epoch[0],'  |  ',total_loss.item())
        return total_loss
    optim_style.step(closure)

#下面三部非常重要，正是因为没搞这三步，才会训练得像坨屎
fix_pic=fix_pic/255  #像素256->1(float)
fix_pic[fix_pic>1]=1
fix_pic[fix_pic<0]=0  #训练的时候没规定最大最小值，可能会练飞(大于255小于0)，这种像素显示出来是有问题的
result_pic=transforms.ToPILImage()(fix_pic.squeeze(0))
result_pic=np.array(result_pic)
cv2.imwrite('./fixxxx.jpg',result_pic)
