import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import imageio



def load_minst(path="data/MINST_data"):
    '''
    导入minst数据集
    :param path: 数据路径
    :return: 数据字典{train_x,train_y,test_x,test_y}
    '''

    minst = input_data.read_data_sets(path,one_hot=True)
    data_dict = {
        "train_x":minst.train.images,
        "train_y":minst.train.labels,
        "test_x":minst.test.images,
        "test_y":minst.test.labels
    }
    return data_dict

def read_img2numpy(batch_size=64,img_h=64,img_w=64,path="data/faces/faces"):
    '''
    读取磁盘图像，并将图片重新调整大小
    :param batch_size: 每次读取图片数量
    :param img_h: 图片重新调整后的高度
    :param img_w: 图片重新调整宽度
    :param path: 数据存放路径
    :return: 图像numpy数组
    '''
    file_list=os.listdir(path)#图像名称列表
    data=np.zeros([batch_size,img_h,img_w,3],dtype=np.uint8)#初始化numpy数组
    mask=np.random.choice(len(file_list),batch_size,replace=True)
    for i in range(batch_size):
        mm=Image.open(path+"/"+file_list[mask[i]])
        tem=mm.resize((img_w,img_h))#重新调整图片大小
        data[i,:,:,:]=np.array(tem)
    #数据归一化-1---1
    data=(data-127.5)/127.5#-1到1之间
    return data
def img2gif(img_path="out/dcgan/",gif_path="out/dcgan/"):
    #获取图像文件列表
    file_list=os.listdir(img_path)
    imges=[]
    for file in file_list:
        if file.endswith(".png"):
            img_name=img_path+file
            imges.append(imageio.read(img_name))
    imageio.mimsave(gif_path+"result.gif",imges,fps=2)

def load_mnist_resize(path="data/MNIST_data",img_w=32,img_h=32):
    data ={}
    with tf.Session() as sess:
        mnist = input_data.read_data_sets(train_dir=path,one_hot=True,reshape=[])
        images = mnist.train.images
        data['data'] = tf.image.resize_images(images=images,size=(img_w,img_h)).eval()
        data['label'] =mnist.train.labels
    return data
def read_all_facade2numpy(img_h=64,img_w=64,path='data/facade',data_file='train_picture',label_file='train_label'):
    """
    读取磁盘图片，并调整尺寸。
    :param img_h: 调整后的高度
    :param img_w: 调整后的宽度
    :param path: 文件路径
    :param data_file: 训练图片文件夹
    :param label_file:训练目标图片文件夹
    :return: 训练数据numpy，训练数据目标值numpy
    """
    file_list = os.listdir(path+'/'+data_file)
    file_len = len(file_list)
    train_data = np.zeros([file_len,img_h,img_w,3],dtype=np.uint8)
    for i in range(file_len):
        name = file_list[i]
        mm = Image.open(path+'/'+data_file+'/'+name)
        tem =mm.resize((img_h,img_w))
        train_data[i,:,:,:] = np.array(tem)
    #数据归一化
    train_data = (train_data-127.5)/127.5  #[-1,1]

    label_data = np.zeros([file_len,img_w,img_h,3],dtype=np.uint8)
    for i in range(file_len):
        name = file_list[i]
        name = name.split('.')[0]+'.jpg'
        mm =Image.open(path+'/'+label_file+'/'+name)
        tem = mm.resize((img_h,img_w))
        label_data[i,:,:,:] = np.array(tem)
    label_data = (label_data-127.5)/127.5
    return train_data,label_data


if __name__=='__main__':
   # data = load_minst()
    #print(data['test_y'].shape)
   data=read_img2numpy()
   print(data.shape)#(64, 64, 64, 3)代表batch_size为64，图片大小为64*64*3
   #data=(data+1)/2. #数据范围为0~1
   data=((data*127.5)+127.5).astype(np.uint8)
   plt.imshow(data[10])
   plt.show()

