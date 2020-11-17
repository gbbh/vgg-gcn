import tensorflow as tf
from tensorflow import keras
from keras import Model,Sequential
from keras.layers import Flatten, Dense, Conv2D, GlobalAveragePooling2D,Multiply,Add
from keras.layers import Input, MaxPooling2D, GlobalMaxPooling2D
from keras.layers import *
import numpy as np
i=0
a=1
b=2
def jian(a,b):
    #print('就开始发动机考虑发的')
    return a-b
def jia(a,b):
    c=jian(a,b)
    return c
d=jia(a,b)
if i:
    K.set_image_dim_ordering('tf')
    sess = tf.InteractiveSession()
    a=np.array([[[[1,2,3,3,3,5]],[[2,3,4,6,7,7]]],[[[1,2,3,4,5,6]],[[3,9,3,5,8,2]]]])
    b = tf.convert_to_tensor(a, dtype=tf.float64)
    c=np.array([[[[1],[2],[4],[4],[2],[3]],[[2],[4],[3],[6],[6],[9]]],[[[1],[6],[4],[7],[3],[5]],[[4],[3],[2],[3],[6],[8]]]])
    d = tf.convert_to_tensor(c, dtype=tf.float64)
    #x_exp = tf.exp(b)
    #x_sum = tf.reduce_sum(x_exp)#(x_exp, axis = 1, keepdims = True)
    #s = x_exp / x_sum
    #s=tf.reduce_sum(s)
    s=tf.matmul(b,d)
    #print('你好你啊你爱好你',d,b)
    s0=b[0,:,0,0]*b[0,:,0,1]
    s1=b[0,:,0,0]*b[0,:,0,2]
    s0=tf.reduce_sum(s0)
    s1=tf.reduce_sum(s1)
    s0=tf.reshape(s0,(1,1,1,1))
    s1=tf.reshape(s1,(1,1,1,1))
    #s = Concatenate()([s0,s1])
    #s=tf.reduce_sum(s)
    #s=tf.reshape(s,(1,1,1,1))
    #print('是地方第三方第三方的',s.eval())

    #x1=Dot(axes=0)([b[0,:,0,0],b[0,:,0,1]])
    #print('是地方第三方第三方的',x1.eval())
    #print('是地方第三方第三方的',x1)
    #x1=tf.reduce_sum(x1)
    #d=tf.sum(d)
    #x2=Concatenate()([b,d])
    #print('是地方第三方第三方的',x1.eval())
    #d=tf.transpose(d[0])
    #print('是地方第三方第三方的',x1)
    #x1 = Dot()([b,d])
    #print('是的方式地方但是方式',d.eval())
    #print("的公司的公司的方式的",d)
def softmax(input):

    x_exp = tf.exp(input)
    x_sum = tf.reduce_sum(x_exp)#(x_exp, axis = 1, keepdims = True)
    s = x_exp / x_sum

    return s
    
def create_upper_matrix(values, size):
    upper = np.zeros((size, size))
    upper[np.tril_indices(5, 1)] = values
    return(upper)

ma = create_upper_matrix([1,2,3,4,5,6], 5)
print('是地方是否健康老师的',ma)
