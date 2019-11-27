from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil
epoch=20
train_path=os.path.dirname(os.path.abspath(__file__))
datagen = ImageDataGenerator(rotation_range = 30,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)
def DataGenerator(datagen,train_path,batch_size,epoch):
    tras_path = os.path.dirname(train_path)
    for root,dirs,files in os.walk(train_path):
        for dir_item in dirs:
            try:
                os.mkdir(os.path.join(train_path,dir_item+'/test_'+dir_item))
            except FileExistsError:
                pass
            for filename in os.listdir(os.path.join(train_path,dir_item)):
                if not os.path.isdir(os.path.join(train_path,dir_item+'/'+filename)):
                    shutil.move(os.path.join(train_path,dir_item+'/'+filename),os.path.join(train_path,dir_item+'/test_'+dir_item+'/'+filename))
            gen = datagen.flow_from_directory(os.path.join(train_path,dir_item),target_size=(256, 256),batch_size=batch_size,save_to_dir=os.path.join(tras_path,'trans_'+dir_item),save_prefix='xx',save_format='jpg')#生成后的图像保存路径
            for i in range(epoch):
                try:
                    gen.next()
                except FileNotFoundError:
                    os.mkdir(os.path.join(tras_path,'trans_'+dir_item))
                    gen.next()
            for filename in os.listdir(os.path.join(train_path,dir_item+'/test_'+dir_item)):
                    if not os.path.isdir(os.path.join(train_path,dir_item+'/'+filename)):
                        shutil.move(os.path.join(train_path,dir_item+'/test_'+dir_item+'/'+filename),os.path.join(train_path,dir_item+'/'+filename))
            os.rmdir(os.path.join(train_path,dir_item+'/test_'+dir_item))
        break
#千万不要手动中断
DataGenerator(datagen,train_path,100,epoch)
# https://keras.io/zh/preprocessing/image/