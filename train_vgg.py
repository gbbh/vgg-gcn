from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import np_utils,get_file
from keras.optimizers import Adam
from model.VGG16 import VGG16
import numpy as np
import utils
import cv2
from keras import backend as K
import tensorflow as tf

import pandas as pd
#在终端完整显示矩阵
np.set_printoptions(threshold=np.inf)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
#K.set_image_dim_ordering('tf')
#下载训练权重
#WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                      # 'releases/download/v0.1/'
                      # 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')


def generate_arrays_from_file1(lines,batch_size):
    # 获取总长度
    n = len(lines)
    i = 0

    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            # 从文件中读取图像
            img = cv2.imread(r"/home/cv/下载/lixiao/Transfer-Learning-master/VGG16-gcn/data/image/train1" + '/' + name)
            #print('原图尺寸',img.shape)
            #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            #print('原图尺寸',img.shape)
            #img = img/255
            X_train.append(img)
            Y_train.append(lines[i].split(';')[1])
            # 读完一个周期后重新开始
            i = (i+1) % n

        # 处理图像
        X_train = utils.resize_image(X_train,(48,96))
        print('输入的大小',X_train.shape)
        #X_train = X_train.reshape(-1,96,48,3)

        #X_train = np.expand_dims(X_train, axis=3)
        #X_train = np.array(X_train)
        #print("火车火车火车火车火车",X_train.shape)
        Y_train = np_utils.to_categorical(np.array(Y_train),num_classes=8)
        #print('标签标签标签标签',Y_train)
        yield (X_train, Y_train)

def generate_arrays_from_file2(lines,batch_size):
    # 获取总长度
    n = len(lines)
    i = 0


    while 1:
        X_train = []
        Y_train = []

        # 获取一个batch_size大小的数据
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            # 从文件中读取图像
            img = cv2.imread(r"/home/cv/下载/lixiao/Transfer-Learning-master/VGG16-gcn/data/image/test" + '/' + name)
            #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            #print('原图尺寸',img.shape)
            #img = img/255
            X_train.append(img)
            Y_train.append(lines[i].split(';')[1])
            # 读完一个周期后重新开始
            i = (i+1) % n

        # 处理图像
        X_train = utils.resize_image(X_train,(48,96))
        #print('输入的大小但是分公司的方式地方是的地方',X_train.shape)
        X_train = X_train.reshape(-1,96,48,3)
        #print('输入的大小但是分公司的方式地方是的地方',X_train.shape)

        #X_train = np.expand_dims(X_train, axis=3)
        #X_train = np.array(X_train)
        #print("火车火车火车火车火车",X_train.shape)
        Y_train = np_utils.to_categorical(np.array(Y_train),num_classes=8)
        #print('标签标签标签标签',Y_train)
        yield (X_train, Y_train)

if __name__ == "__main__":
    # 模型保存的位置
    log_dir = "/home/cv/下载/lixiao/Transfer-Learning-master/VGG16-gcn/logs/"

    # 打开数据集的txt
    with open(r"/home/cv/下载/lixiao/Transfer-Learning-master/VGG16-gcn/data/train1.txt","r") as f1:
        lines1 = f1.readlines()
    with open(r"/home/cv/下载/lixiao/Transfer-Learning-master/VGG16-gcn/data/test.txt","r") as f2:
        lines2 = f2.readlines()
    # 打乱行，这个txt主要用于帮助读取数据来训练
    # 打乱的数据更有利于训练
    np.random.seed(10101)
    np.random.shuffle(lines1)
    np.random.shuffle(lines2)
    np.random.seed(None)

    # 90%用于训练，10%用于估计。
    num_train = int(len(lines1)*0.9)
    num_val = len(lines1)-num_train
    #print('验证',num_train)
    #num_val = len(lines1)-num_train
    # 建立VGG模型
    model = VGG16(8)
    #weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',WEIGHTS_PATH_NO_TOP,cache_subdir='models',file_hash='6d6bbae143d832006294945121d1f1fc')

    #model.load_weights(weights_path,by_name=True)

    # 保存的方式，3世代保存一次
    checkpoint_period1 = ModelCheckpoint(
                                    log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='acc',
                                    save_weights_only=False,
                                    save_best_only=True,
                                    period=3
                                )
    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
                            monitor='acc',
                            factor=0.5,
                            patience=3,
                            verbose=1
                        )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
                            monitor='val_loss',
                            min_delta=0,
                            patience=10,
                            verbose=1
                        )
    #trainable_layer = 19
    #for i in range(trainable_layer):
    #    model.layers[i].trainable = False

    # 交叉熵
    model.compile(loss = 'categorical_crossentropy',
            optimizer = Adam(lr=1e-4),
            metrics = ['accuracy'])

    # 一次的训练集大小
    batch_size =50

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))



    #print('freeze the first {} layers of total {} layers.'.format(trainable_layer, len(model.layers)))

    # 开始训练
    model.fit_generator(generate_arrays_from_file1(lines1[:num_train], batch_size),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=generate_arrays_from_file1(lines1[num_train:], batch_size),
            validation_steps=max(1, num_val//batch_size),
            epochs=50,
            initial_epoch=0,
            callbacks=[checkpoint_period1, reduce_lr])

    model.save_weights(log_dir+'middle_one.h5')
    #for i in range(len(model.layers)):
    #    model.layers[i].trainable = True
    # 交叉熵
    #model.compile(loss = 'categorical_crossentropy',
    #        optimizer = Adam(lr=1e-4),
    #        metrics = ['accuracy'])

    #model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
    #        steps_per_epoch=max(1, num_train//batch_size),
    #        validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
    #        validation_steps=max(1, num_val//batch_size),
    #        epochs=6,
    #        initial_epoch=3,
    #        callbacks=[checkpoint_period1, reduce_lr])

    #model.save_weights(log_dir+'last_one.h5')
