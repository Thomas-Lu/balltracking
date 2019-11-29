import os
import sys
lmu_path = os.path.abspath("../lmu")
sys.path.append(lmu_path)

from lmu import LMUCell

import keras as K
from keras.applications import ResNet50
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input, GlobalAveragePooling2D, Conv2DTranspose, Reshape, Lambda, Conv2D, UpSampling2D
from keras.layers import TimeDistributed
from keras.layers.recurrent import RNN
from keras.models import Sequential, Model
from keras.initializers import Constant
from keras.utils import multi_gpu_model, to_categorical

import numpy as np

def lmu_layer(return_sequences=False,**kwargs):
    return RNN(LMUCell(units=6,
                       order=6,
                       theta=15,
                       input_encoders_initializer=Constant(1),
                       hidden_encoders_initializer=Constant(0),
                       memory_encoders_initializer=Constant(0),
                       input_kernel_initializer=Constant(0),
                       hidden_kernel_initializer=Constant(0),
                       memory_kernel_initializer='glorot_normal',
                      ),
               return_sequences=return_sequences,
               **kwargs)

def Lmu_stack(input_tensor, return_sequences):
    t1 = []
    t2 = []
    shape = K.backend.int_shape(input_tensor)
    input_tensor = Reshape([shape[1], -1, shape[-1]])(input_tensor)
    for i in range(shape[-1]):
        x = Lambda(lambda x: x[...,i])(input_tensor)
        x = lmu_layer(return_sequences=return_sequences)(x)
        t1.append(Dense(1)(x))
        t2.append(Dense(49)(x))
    return [Lambda(lambda x: K.backend.stack(x, axis=-1))(t1), Lambda(lambda x: K.backend.stack(x, axis=-1))(t2)]

def velocity_layer(t):
    t = TimeDistributed(Reshape([-1]))(t)
    return TimeDistributed(Dense(2))(t)

def deconvolution_layer(t):
    heat = TimeDistributed(Reshape([7,7,-1]))(t)
    heat = TimeDistributed(Conv2DTranspose(filters=32,kernel_size=3,strides=2, dilation_rate=1,activation='relu', padding='same'))(heat)
    heat = TimeDistributed(UpSampling2D( interpolation='bilinear'))(heat)
    heat = TimeDistributed(Conv2DTranspose(filters=8,kernel_size=3,strides=2, dilation_rate=1,activation='relu', padding='same'))(heat)
    heat = TimeDistributed(UpSampling2D( interpolation='bilinear'))(heat)
    heat = TimeDistributed(Conv2DTranspose(filters=1,kernel_size=3,strides=2, dilation_rate=1,activation='relu', padding='same'))(heat)
    return heat

seq_len = 15

resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))

for layer in resnet_model.layers:
    layer.trainable=False
    
input_layer = Input(shape=(seq_len, 224, 224, 3))

x = TimeDistributed(resnet_model)(input_layer)
x = TimeDistributed(Conv2D(kernel_size=1, filters=64))(x) #reduce channels

out1, out2 = Lmu_stack(x, return_sequences=True)

v = velocity_layer(out1, name="velocity")

heat = deconvolution_layer(out2, name="heatmap")

model = Model(inputs=input_layer, outputs=[heat,v])

model.summary()

#Data
X = np.load('X_width_224_15_2048.npy')
Y = np.load('Y_width_224_15_2048.npy')
V = np.load('V_width_224_15_2048.npy')
Y = np.expand_dims(Y, (-1))
print(X.shape, Y.shape, V.shape)

losses = {
	"velocity": "mean_squared_error",
	"heatmap": "binary_crossentropy",
}

model.compile(loss=losses, optimizer='adam', metrics=['accuracy'])
model.fit(X, [Y,V], epochs=30, batch_size=16, validation_split=0.05, 
        callbacks=[EarlyStopping(restore_best_weights=True, patience=2)])