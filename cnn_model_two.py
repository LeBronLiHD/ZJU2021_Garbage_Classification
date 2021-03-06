# -*- coding: utf-8 -*-
from tensorflow.keras.layers \
    import Input, Dense, Flatten, Dropout, Activation, Conv2D, ReLU, MaxPool2D
# from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
import time
import parameter


def cnn_model_two(input_shape):

    inputs = Input(shape=input_shape)

    cnn = Conv2D(32, kernel_size=3, padding='same', input_shape=input_shape)(inputs)
    cnn = (BatchNormalization())(cnn)
    cnn = (ReLU())(cnn)
    cnn = (MaxPool2D(pool_size=2, strides=2))(cnn)

    cnn = (Conv2D(64, kernel_size=3, padding='same'))(cnn)
    cnn = (BatchNormalization())(cnn)
    cnn = (ReLU())(cnn)
    cnn = (MaxPool2D(pool_size=2, strides=2))(cnn)

    cnn = (Conv2D(128, kernel_size=3, padding='same'))(cnn)
    cnn = (BatchNormalization())(cnn)
    cnn = (ReLU())(cnn)
    cnn = (MaxPool2D(pool_size=2, strides=2))(cnn)

    cnn = (Conv2D(128, kernel_size=3, padding='same'))(cnn)
    cnn = (BatchNormalization())(cnn)
    cnn = (ReLU())(cnn)
    cnn = (MaxPool2D(pool_size=2, strides=2))(cnn)

    cnn = (Conv2D(128, kernel_size=3, padding='same'))(cnn)
    cnn = (BatchNormalization())(cnn)
    cnn = (ReLU())(cnn)
    cnn = (MaxPool2D(pool_size=2, strides=2))(cnn)

    cnn = (Conv2D(128, kernel_size=3, padding='same'))(cnn)
    cnn = (BatchNormalization())(cnn)
    cnn = (ReLU())(cnn)
    cnn = (MaxPool2D(pool_size=2, strides=2))(cnn)

    cnn = (Conv2D(128, kernel_size=3, padding='same'))(cnn)
    cnn = (BatchNormalization())(cnn)
    cnn = (ReLU())(cnn)
    cnn = (MaxPool2D(pool_size=2, strides=2))(cnn)

    cnn = (Flatten())(cnn)
    cnn = (Dense(256))(cnn)
    cnn = (Dropout(0.4))(cnn)
    cnn = (Dense(parameter.CLASS_NUM))(cnn)

    outputs = cnn

    model = Model(inputs=inputs, outputs=outputs)

    return model

