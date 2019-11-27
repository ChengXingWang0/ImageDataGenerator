# ImageDataGenerator
Keras和OpenCv数据增强API的封装版

1.Keras自带的数据增强API奇坑无比，对文件结构有很奇怪的要求，所以对其进行了一层封装

Folder
  --class1
    --1.jpg
    --2.jpg
    .......
  --class2
    --3.jpg
    --4.jpg
    .......
  ..........
  
  放在Folder文件夹下，运行后，会生成与class相对应的trans_classN文件夹在Folder文件夹外
  
  
 2.OpenCv的API,与上述类似，但是生成的trans_classN在Folder内部
  
 
 两个代码的末尾提供了相应的官方文档，大家使用时各取所需，调整api内参数和对象即可
 epoch是每张图片增强的倍数，如epoch为20,则一张图片会增强到20张图片
