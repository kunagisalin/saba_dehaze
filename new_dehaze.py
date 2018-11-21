#coding=utf-8#
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate,Activation, ZeroPadding2D,UpSampling2D
from keras.layers import add, Flatten
from keras.utils import plot_model
from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import os
from keras import optimizers
from PIL import Image
import numpy as np
from keras.preprocessing.image import img_to_array
#load data
from keras.preprocessing.image import load_img # load an image from file 
from keras.preprocessing.image import img_to_array # convert the image pixels to a numpy array 
from keras.callbacks import TensorBoard
import math
from keras import backend as K

directory = './outdoor/hazy/'
arr = []
result = os.listdir(directory)
result.sort()
for imgname in result:
	img = Image.open(directory + imgname)
	img = img.resize((500,500))
	arr.append(np.array(img))
x_train = np.array(arr)
print(x_train.shape)
import matplotlib.pyplot as plt
plt.imshow(x_train[1])
plt.show()

directoryy = './outdoor/gt/'
arry = []
result = os.listdir(directoryy)
result.sort()
for imgname in result:
	imgy = Image.open(directoryy + imgname)
	imgy = imgy.resize((500,500))
	arry.append(np.array(imgy))
y_train = np.array(arry)
print(y_train.shape)

plt.imshow(y_train[1])
plt.show()

x_train = x_train.astype('float32')/255.0
y_train = y_train.astype('float32')/255.0


def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
	if name is not None:
		bn_name = name + '_bn'
		conv_name = name + '_conv'
	else:
		bn_name = None
		conv_name = None

	x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
	x = BatchNormalization(axis=3, name=bn_name)(x)
	return x

def identity2_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
	x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
	x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
	if with_conv_shortcut:#shortcut的含义是：将输入层x与最后的输出层y进行连接，如上图所示
		shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
		x = add([x, shortcut])
		return x
	else:
		x = add([x, inpt])
		return x
def identity3_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
	x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
	x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
	x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
	if with_conv_shortcut:#shortcut的含义是：将输入层x与最后的输出层y进行连接，如上图所示
		shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
		x = add([x, shortcut])
		return x
	else:
		x = add([x, inpt])
		return x
def identity4_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
	x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
	x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
	x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
	x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
	if with_conv_shortcut:#shortcut的含义是：将输入层x与最后的输出层y进行连接，如上图所示
		shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
		x = add([x, shortcut])
		return x
	else:
		x = add([x, inpt])
		return x

def GAWN(width,height,channel):
	x = inpt = Input(shape=(width,height,channel))
#x = ZeroPadding2D((3, 3))(inpt)
#conv1
	#x=Conv2d_BN(x,nb_filter=64,kernel_size=(3,3),padding='same')
	#x=Conv2d_BN(x,nb_filter=64,kernel_size=(3,3),padding='same')
	x = Conv2D(nb_filter=64, kernel_size=(3,3), padding='same', strides=(1, 1), activation='relu')(x)
	x = Conv2D(nb_filter=64, kernel_size=(3,3), padding='same', strides=(1, 1), activation='relu')(x)

#conv2_x
	x=MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
	#x=Conv2d_BN(x,nb_filter=128,kernel_size=(3,3),padding='same')
	x=MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)

#conv3_x
	x = identity2_Block(x, nb_filter=64, kernel_size=(3, 3))
	x = identity2_Block(x, nb_filter=64, kernel_size=(3, 3))
	x = identity3_Block(x, nb_filter=64, kernel_size=(3, 3))
	x = identity4_Block(x, nb_filter=64, kernel_size=(3, 3))

#conv4_x
	x = UpSampling2D(size=(2, 2), data_format=None)(x)
	x = UpSampling2D(size=(2, 2), data_format=None)(x)

#conv5_x
	x = Conv2D(nb_filter=3, kernel_size=(3,3), padding='same', strides=(1, 1), activation='relu')(x)

	#x = Flatten()(x)
	#x = Dense(1024, activation='relu')(x)
	model = Model(inputs=inpt, outputs=x)
	return model

from keras.utils import plot_model
model = GAWN(500,500,3)
model.summary()
#plot_model(model, to_file='model.png')

#psnr
def psnr(y_ture,y_red):
	return -10*K.log(K.mean(K.flatten(y_ture - y_red))**2)/np.log(10)

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
'''
tb = TensorBoard(log_dir='./logs',  # log 目录
	histogram_freq=1,  # 按照何等频率（epoch）来计算直方图，0为不计算
	batch_size=32,     # 用多大量的数据计算直方图
	write_graph=True,  # 是否存储网络结构图
	write_grads=False, # 是否可视化梯度直方图
	write_images=False,# 是否可视化参数
	embeddings_freq=0, 
	embeddings_layer_names=None, 
	embeddings_metadata=None)
callbacks = [tb]
'''
model.fit(x_train,y_train,batch_size=32,epochs = 20)

model.save('9.h5')

