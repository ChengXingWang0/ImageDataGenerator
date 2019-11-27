import cv2
from imgaug import augmenters as iaa
import os
import time
#对batch中的一部分图片应用一部分Augmenters,剩下的图片应用另外的Augmenters。
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
# 定义一组变换方法.
seq = iaa.Sequential([
    # 选择0到5种方法做变换
    iaa.SomeOf((0, 5),
        [
                iaa.Fliplr(0.5), # 对50%的图片进行水平镜像翻转
                iaa.Flipud(0.5), # 对50%的图片进行垂直镜像翻转	
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), #高斯滤波
                    iaa.AverageBlur(k=(2, 7)),  #均值滤波，k指核的大小
                    iaa.MedianBlur(k=(3, 11)),  #中值滤波
                ]),
                # iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB",children=iaa.WithChannels(0, iaa.Add(10))), #颜色空间转换
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), #锐化图片
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),      #与锐化类似，但是具有压纹效果
                iaa.AdditiveGaussianNoise(      #增加高斯噪声
                    loc=0, scale=(0.0, 0.05*255)
                ),
                iaa.Invert(0.05, per_channel=True), # 几率反转每个通道
                iaa.Add((-10, 10), per_channel=0.5),   # 为每个像素上加浮动10
                iaa.AddElementwise((-40, 40)),
                iaa.Multiply((0.5, 1.5)), #更改原来图像的亮度
                iaa.MultiplyElementwise((0.5, 1.5)),    #为每个像素乘于一个值
                iaa.ContrastNormalization((0.5, 2.0)),  #更改对比度
        ],
        random_order=True #以随机的方式执行上述扩充
    )
],random_order=True) 
# 图片文件相关路径
abs_path=os.path.dirname(os.path.abspath(__file__))
epoch=2
def DataGenerator(seq,abs_path,epoch):
    for root,dirs,files in os.walk(abs_path):
        for dir_item in dirs:
            try:
                new_dirname=os.path.join(abs_path,'trans_'+dir_item)
                old_dirname=os.path.join(abs_path,dir_item)
                os.mkdir(new_dirname)
            except FileExistsError:
                pass
            filelist = os.listdir(old_dirname)
            imglist=[]
            for item in filelist:
                img = cv2.imread(os.path.join(old_dirname,item))
                imglist.append(img)
            for count in range(epoch):
                print(dir_item)
                images_aug = seq.augment_images(imglist)
                for index in range(len(images_aug)):
                    filename = str(count) +str(time.time()) +'.jpg'
                    cv2.imwrite(os.path.join(new_dirname,filename),images_aug[index])
DataGenerator(seq,abs_path,3)
#https://blog.csdn.net/u012897374/article/details/80142744#commentBox