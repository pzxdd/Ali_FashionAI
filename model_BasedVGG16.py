import os
import keras
import numpy as np
from PIL import Image
import pandas as pd
#from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Concatenate,MaxPooling2D,Flatten
from keras import backend as K
#from keras.utils import plot_model


#用python的yield预处理图片
#一张图片对应72个点 24 * 3 = 72


def listDived(anno_list):
    anno_temp = []
    for land in anno_list:
        for i in land:
            anno_temp.append(i)

    return anno_temp


def readImage(annoPath,numPic):
    csv_handle = pd.read_csv(annoPath)
    count = numPic
    Img =[]
    Anno = []
    for row in csv_handle.iterrows():
        r = row[1]
        fileName = r['image_id']
        Image = np.array(Image.open(fileName))

        split_axis = r.ix[2:].str.split(pat='_', expand=True)
        split_axis = np.array(split_axis, dtype='float32')
        anno_list = split_axis.tolist()
        anno_temp = listDived(anno_list)
        Anno.append(anno_temp)
        Img.append(Image)
        count -= 1
        if count == 0:
            x_train,y_train = Img,Anno
            count = numPic
            Img,Anno = [],[]
            yield(np.array(x_train),np.array(y_train))


def change_vgg16(input_shape):
    #改变vgg最后一层，然后添加一个vis
    img_input = Input(shape=input_shape)
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    #使用1x1卷积
    x = Conv2D(128, (1, 1), activation='relu', padding='same', name='block5_conv4')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x_final_dense = Dense(4096, activation='relu', name='fc2')(x)

    #regress block
    x_landmark=  Dense(72,name = 'prediction_landmark')(x_final_dense)

    #x = [Dense(3, activation='softmax', name='p%d' % (i + 1))(x_final_dense) for i in range(24)]
    output = [x_landmark]
    #output.extend(x)
    # x_v1 = Dense(2,name = 'visibility_1')(x_final_dense)
    # x_v2 = Dense(2,name = 'visibility_2')(x_final_dense)
    # x_v3 = Dense(2,name = 'visibility_3')(x_final_dense)
    # x_v4 = Dense(2,name= 'visibility_4')(x_final_dense)
    # x_v5 = Dense(2,name= 'visibility_5')(x_final_dense)
    # x_v6 = Dense(2, name='visibility_6')(x_final_dense)

    #x_final = Concatenate([x_landmark,x_v1,x_v2,x_v3,x_v4,x_v5,x_v6])

    model = Model(img_input, outputs=output, name='change_vgg16')

    return model


    #x = Dense(classes, activation='softmax', name='predictions')(x)
def euc_dist_keras(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True))


if __name__ == "__main__":
    input_shape=(512,512,3)
    model = change_vgg16(input_shape)

    model.summary()
    #plot_model(model, to_file='change_vgg16.png')
    # weight_path=''
    model.load_weights(weight_path,by_name=True)

    model.compile(optimizer='sgd',loss=euc_dist_keras)

    for x_train,y_train in readImage('Annotations/train.csv',32):
        print(x_train.shape,y_train.shape)
        model.fit(x_train,y_train,batch_size=10,epochs=100)

    model.save('model_1.h5')





