import matplotlib.image as mpimg
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.ops import array_ops

def load_image(path):
    # 读取图片，rgb
    img = mpimg.imread(path)
    # 将图片修剪成中心的正方形
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    return crop_img

def resize_image(image, size):
    with tf.name_scope('resize_image'):
        images = []
        #print("图片图片图片",image)
        for i in image:
            #print("爱爱爱爱爱爱爱爱",i.shape)
            i = cv2.resize(i, size)
            images.append(i)
        images = np.array(images)
        return images

def print_answer(argmax):
    with open("/home/cv/下载/lixiao/Transfer-Learning-master/VGG16-gcn/data/model/index_word.txt","r",encoding='utf-8') as f:
        synset = [l.split(";")[1].replace("\n","") for l in f.readlines()]
    print(synset[argmax])
    return synset[argmax]

def multi(x):

    a=np.array([[0,1,1],[1,0,1],[1,1,0]])
    d=np.array([[2,0,0],[0,2,0],[0,0,2]])
    c=np.linalg.inv(d**0.5)*a*np.linalg.inv(d**0.5)
    c=np.expand_dims(c,axis=2)
    c=np.expand_dims(c,axis=0)
    c=np.tile(c, (batch_size,1,1,512))
    x=c*x
    return x
