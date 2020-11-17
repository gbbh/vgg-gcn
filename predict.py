import numpy as np
import utils
import cv2
from keras import backend as K
from model.VGG16_gcn import VGG16_gcn

K.set_image_dim_ordering('tf')

if __name__ == "__main__":
    model = VGG16_gcn(8)
    model.load_weights("/home/cv/下载/lixiao/Transfer-Learning-master/VGG16-gcn/logs/middle_one.h5")
    img = cv2.imread("/home/cv/下载/lixiao/Transfer-Learning-master/VGG16-gcn/data/image/test/00001.png")
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #img = img/255
    print("福建省抵抗哦附近皇帝iosdjfiosdjo",img.shape)
    #img = np.array(img)
    img = np.expand_dims(img,axis = 0)
    img = utils.resize_image(img,(48,96))
    print(utils.print_answer(np.argmax(model.predict(img))))
