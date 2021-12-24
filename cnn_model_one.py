# -*- coding: utf-8 -*-

import filter
import numpy as np
import tensorflow as tf
import cv2
import os
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import img_to_array
from keras.callbacks import EarlyStopping
from imutils import paths
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import tkinter
import matplotlib
import matplotlib.pyplot as plt
import parameter
from keras.callbacks import TensorBoard
matplotlib.use('TkAgg')


def cnn_model_one(size):
    """
    训练CNN模型
    :param data:  图像矩阵, rbg
    :param label: 图像对应的标签, one-hot
    :param size:  图片尺寸
    :return: 返回训练后的模型
    """
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                     input_shape=(size, size, 3), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                     input_shape=(size, size, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                     input_shape=(size, size, 3), padding='same'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                     input_shape=(size, size, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
                     input_shape=(size, size, 3), padding='same'))
    model.add(Dropout(0.3))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                     input_shape=(size, size, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(parameter.CLASS_NUM, activation='softmax'))

    return model
