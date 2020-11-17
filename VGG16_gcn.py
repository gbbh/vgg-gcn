import tensorflow as tf
from tensorflow import keras
from keras import Model,Sequential
from keras.layers import Flatten, Dense, Conv2D, GlobalAveragePooling2D,Multiply,Add
from keras.layers import Input, MaxPooling2D, GlobalMaxPooling2D, Concatenate
from keras.layers import *
from keras.activations import softmax
import numpy as np
K.set_image_dim_ordering('tf')

def gcn(a):

    bs = 50
    [q,w,e,r]=a[0].shape#权重矩阵bs×18×18×1
    [t,y,u,i]=a[1].shape#特征图
    h = tf.reshape(a[1],(-1,y*u,1,i))
    h = tf.reshape(h,(-1,i,y*u,1))
    a = tf.reshape(a[0],(-1,r,w,e))
    d = np.identity(w)
    d = np.linalg.inv(d**0.5)
    d = tf.convert_to_tensor(d,dtype=tf.float32)
    d = tf.expand_dims(d,axis=0)
    d = tf.expand_dims(d,axis=3)
    d = tf.tile(d,[bs,1,1,1])
    d = tf.reshape(d,(-1,r,w,e))
    c = tf.matmul(d,a)
    c = tf.matmul(c,d)
    c = tf.tile(c,[1,i,1,1])
    x = tf.matmul(c,h)
    x = tf.reshape(x,[-1,y*u,1,i])
    x = tf.reshape(x,[-1,y,u,i])
    print('图卷及图卷及图卷及',x)

    return x

def awe(input):

    [d,a,b,c] = input.shape#50*6*3*256
    upper1 = np.zeros((a*b, a*b))
    upper2 = np.zeros((a*b, a*b))
    z=int((a*b+1)*(a*b))/2
    #bs = 100
    #print('是哦你是是地方地方地方的个',a*b)
    input = tf.reshape(input,(-1,1,a*b,c))
    for i0 in range(50):
        for i1 in range(a*b):
            for i2 in range(i1,a*b):
                x1 = input[i0,0,i1,:]*input[i0,0,i2,:]
                x1 = tf.reduce_sum(x1)
                x1 = x1/(int(c)**0.5)
                x1 = tf.reshape(x1,(1,1,1,1))
                if i2==i1:
                    x2=x1
                else:
                    x2 = Concatenate(axis=2)([x2,x1])
            #print('爱克斯开克斯爱克斯iiiii',x2)
            #x2 = softmax(x2,axis=2)
            print('爱克斯开克斯爱克斯2222',x2)
            if i1==0:
                x3=x2
            else:
                x3=Concatenate(axis=2)([x3,x2])
            #print('爱克斯开克斯爱克斯3333',x3)
        #x3=tf.reshape(x3,(171))
        #x3=tf.squeeze(x3)
        #print('爱克斯开克斯爱克斯3333',x3)
        with tf.Session() as sess:
            print('爱克斯开克斯爱克斯3333',x3.eval())
            upper1[np.triu_indices(int(a*b), 0)] = x3
        #upper2[np.tril_indices(a*b, 0)] = x3
        #upper2[np.tril_indices(a*b, 2)] = 0
        #x2=upper1+upper2
        if i0==0:
            x4=x3
        else:
            x4=Concatenate(axis=0)([x4,x3])

    return x4

def VGG16_gcn(num_classes):
    # 96，48，3
    image_input = Input(shape = (96,48,3))
    # 第一个卷积部分
    # 48，24，64
    net={}
    x = Conv2D(32,(3,3),activation = 'relu',padding = 'same',name = 'block1_conv1')(image_input)
    x = Conv2D(32,(3,3),activation = 'relu',padding = 'same', name = 'block1_conv2')(x)
    x = MaxPooling2D((2,2), strides = (2,2), name = 'block1_pool')(x)
    #net['block1_conv3'] = Conv2D(32,(3,3),activation = 'relu',padding = 'same', name = 'block1_conv3')(x)
    #x = MaxPooling2D((2,1),strides = (2,1),name = 'block5_pool')(x)
    #x = Lambda(awe,name ='block1_awe1')(net['block1_conv3'])
    #x = Conv2D(1,(3,3),activation = 'relu',padding = 'same', name = 'block1_conv4')(x)
    #x = Lambda(gcn,name ='block1_gcn1')([x,net['block1_conv3']])
    #x = Conv2D(32,(3,3),activation = 'relu',padding = 'same', name = 'block1_conv5')(x)
    # 第二个卷积部分
    # 24,12,128
    x = Conv2D(64,(3,3),activation = 'relu',padding = 'same',name = 'block2_conv1')(x)
    x = Conv2D(64,(3,3),activation = 'relu',padding = 'same',name = 'block2_conv2')(x)
    x = MaxPooling2D((2,2),strides = (2,2),name = 'block2_pool')(x)

    # 第三个卷积部分
    # 12,6,256
    x = Conv2D(128,(3,3),activation = 'relu',padding = 'same',name = 'block3_conv1')(x)
    x = Conv2D(128,(3,3),activation = 'relu',padding = 'same',name = 'block3_conv2')(x)
    x = Conv2D(128,(3,3),activation = 'relu',padding = 'same',name = 'block3_conv3')(x)
    x = MaxPooling2D((2,2),strides = (2,2),name = 'block3_pool')(x)

    # 第四个卷积部分
    # 6,3,512
    x = Conv2D(256,(3,3),activation = 'relu',padding = 'same', name = 'block4_conv1')(x)
    x = Conv2D(256,(3,3),activation = 'relu',padding = 'same', name = 'block4_conv2')(x)
    x = Conv2D(256,(3,3),activation = 'relu',padding = 'same', name = 'block4_conv3')(x)
    x = MaxPooling2D((2,2),strides = (2,2),name = 'block4_pool')(x)

    # 第五个卷积部分
    # 3,3,512
    x = Conv2D(256,(3,3),activation = 'relu',padding = 'same', name = 'block5_conv1')(x)
    x = Conv2D(256,(3,3),activation = 'relu',padding = 'same', name = 'block5_conv2')(x)
    net['block5_conv3'] = Conv2D(256,(3,3),activation = 'relu',padding = 'same', name = 'block5_conv3')(x)
    x = Lambda(awe,name ='block5_awe1')(net['block5_conv3'])
    x = Conv2D(1,(3,3),activation = 'relu',padding = 'same', name = 'block5_conv4')(x)
    x = Lambda(gcn,name ='block5_gcn1')([x,net['block5_conv3']])
    x = Conv2D(256,(3,3),activation = 'relu',padding = 'same', name = 'block2_conv3')(x)
    x = MaxPooling2D((2,1),strides = (2,1),name = 'block5_pool')(x)
    # 提取特征

    # 分类部分
    x = Flatten(name = 'flatten')(x)
    x = Dense(2048,activation = 'relu',name = 'fullc1')(x)
    x = Dense(2048,activation = 'relu',name = 'fullc2')(x)
    x = Dense(num_classes,activation = 'softmax',name = 'fullc3')(x)
    model = Model(image_input,x,name = 'vgg16')

    return model
