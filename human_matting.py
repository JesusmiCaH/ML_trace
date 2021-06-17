import time
import cv2
import torch 
import argparse
import numpy as np
import os 
import torch.nn.functional as F

# 图片尺寸
img_size=320
# 不要动模型！
torch.set_grad_enabled(False)

# 确定用什么装备训练——GPU/CPU
if torch.cuda.is_available():
    n_gpu = torch.cuda.device_count()
    print('GPU数量：{}'.format(n_gpu))
    device=torch.device('cuda')
else:
    print('你没有GPU！')
    device=torch.device('cpu')

#模型载入
def load_model(model_path):
    print('Loading model from {}...'.format(model_path))
    if device==torch.device('cpu'):
        myModel = torch.load(model_path, map_location=lambda storage, loc: storage)
    else:
        myModel = torch.load(model_path)
    
    myModel.eval()  #eval是模型的验证模式，和.train()相对
    myModel.to(device)
    
    return myModel


def img_restruction(fg_image,bg_image,size,net):
    # opencv
    origin_h, origin_w, c = fg_image.shape  #三通道，c就是3，用不上
    image_resize = cv2.resize(fg_image,(size,size),interpolation=cv2.INTER_CUBIC)
    image_resize = (image_resize - (104., 112., 121.,)) / 255.0
    #这里需要给图片加一维，因为神经网络的输入默认是原始数据维度+1维。
    tensor_4D = torch.FloatTensor(1,3,size,size)
    #原来是：长*宽*三通道，现在是：三通道*长*宽
    tensor_4D[0,:,:,:] = torch.FloatTensor(image_resize.transpose(2,0,1))
    inputs = tensor_4D.to(device)
    
    #用训练好的模型得出遮罩。
    trimap, alpha = net(inputs)
    #遮罩在显存中，需要移回去
    if device==torch.device('cuda'):
        alpha_np = alpha[0,0,:,:].cpu().data.numpy()
    else:
        alpha_np = alpha[0,0,:,:].data.numpy()
    
    #背景遮罩alpha是一个二维的数组
    alpha_np = cv2.resize(alpha_np, (origin_w, origin_h), interpolation=cv2.INTER_CUBIC)
    #维度为3，因为原图的维度为3，不然没法乘
    alpha_np=alpha_np[..., np.newaxis]
    
    # 背景缩放一下
    bg_h,bg_w,c = bg_image.shape  #三通道，c就是3，用不上
    if bg_h/origin_h<bg_w/origin_w:
        bg_image = cv2.resize(bg_image,(origin_h*bg_w//bg_h,origin_h),interpolation=cv2.INTER_CUBIC)
        bg_image=bg_image[:,(origin_h*bg_w//bg_h-origin_w)//2:(origin_h*bg_w//bg_h+origin_w)//2]
    else:
        bg_image = cv2.resize(bg_image,(origin_w,origin_w*bg_h//bg_w),interpolation=cv2.INTER_CUBIC)
        bg_image=bg_image[(origin_w*bg_h//bg_w-origin_h)//2:(origin_w*bg_h//bg_w+origin_h)//2,:]
    
    fg_image=fg_image.astype(np.int16)
    bg_image=bg_image.astype(np.int16)
    #透明图层叠加公式，这部分可以见论文。
    result_img=bg_image+(fg_image-bg_image)*alpha_np

    result_img[result_img<0] = 0
    result_img[result_img>255] = 255
    out = result_img.astype(np.uint8)
    return out

    
def image_seg(net):
    human_img_path='imgs/wbb.jpg'
    background_img_path='imgs/bg1.jpg'
    size=320

    human_img = cv2.imread(human_img_path)
    background_img = cv2.imread(background_img_path)
    
    frame_seg = img_restruction(human_img, background_img, size, net)

    # show a frame
    cv2.imwrite("./mix.jpg", frame_seg)
    
def main():
    model_path='model/SHMmodel.pth'
    myModel = load_model(model_path)
    image_seg(myModel)

if __name__ == "__main__":
    main()
