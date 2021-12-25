# -*- coding: utf-8 -*-

"""
datagen = ImageDataGenerator(
    featurewise_center=False,  # 是否使输入数据去中心化（均值为0），
    samplewise_center=False,  # 是否使输入数据的每个样本均值为0
    featurewise_std_normalization=False,  # 是否数据标准化（输入数据除以数据集的标准差）
    samplewise_std_normalization=False,  # 是否将每个样本数据除以自身的标准差
    zca_whitening=False,  # 是否对输入数据施以ZCA白化
    rotation_range=15,  # 数据提升时图片随机转动的角度(范围为0～180)
    width_shift_range=0.15,  # 数据提升时图片水平偏移的幅度（单位为图片宽度的占比，0~1之间的浮点数）
    height_shift_range=0.15,  # 同上，只不过这里是垂直
    horizontal_flip=False,  # 是否进行随机水平翻转
    vertical_flip=False)  # 是否进行随机垂直翻转
"""

import numpy as np
import parameter
import generate
import load_image
import cnn_model_one, cnn_model_two
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import os
import random
import prediction
import datetime


def flatten(data):
    data_flat = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            data_flat.append(data[i][j])
    return np.array(data_flat)


def model_evaluate(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=1, batch_size=parameter.BATCH_SIZE)
    save_dir = parameter.MODEL
    model_name = "model_" + str(parameter.EPOCH_NUM) + "_" + str(score[1]) + '.h5'
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('save model at %s ' % model_path)
    print("train model done!")


def plot_training_history(history):
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def train_model_fit(data, label, size, x_test, y_test, model_select=1):
    model = None
    if model_select == 1:
        model = cnn_model_one.cnn_model_one(size)
    elif model_select == 2:
        model = cnn_model_two.cnn_model_two((size, size, 3))
    else:
        print("model_select error!")

    model.compile(loss="categorical_crossentropy",
                  optimizer="Adam",
                  metrics=["accuracy"])
    tensorboard = TensorBoard(parameter.LOG + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                              histogram_freq=1)

    res = model.fit(data, label,
                    batch_size=parameter.BATCH_SIZE,
                    epochs=parameter.EPOCH_NUM,
                    callbacks=[tensorboard],
                    shuffle=True,
                    verbose=1,
                    # steps_per_epoch=len(data) / 32,
                    # validation_steps=(5 * parameter.GEN_RATE * parameter.CLASS_NUM)/32,
                    validation_split=0.15)
    plot_training_history(res.history)
    print(model.summary())
    model_evaluate(model, x_test, y_test)
    prediction.model_prediction(parameter.MODEL, x_test, y_test)


def train_model_gen(data, label, size, x_test, y_test, model_select=1):
    model = None
    if model_select == 1:
        model = cnn_model_one.cnn_model_one(size)
    elif model_select == 2:
        model = cnn_model_two.cnn_model_two((size, size, 3))
    else:
        print("model_select error!")
    model.compile(loss="categorical_crossentropy",
                  optimizer="Adam",
                  metrics=["accuracy"])
    tensorboard = TensorBoard(parameter.LOG, histogram_freq=1)

    data_gen = ImageDataGenerator(
        featurewise_center=False, samplewise_center=False,
        featurewise_std_normalization=False, samplewise_std_normalization=False,
        zca_whitening=True, zca_epsilon=1e-06, rotation_range=90, width_shift_range=0.0,
        height_shift_range=0.0, brightness_range=None, shear_range=0.0, zoom_range=0.0,
        channel_shift_range=0.0, fill_mode='nearest', cval=0.0,
        horizontal_flip=False, vertical_flip=False, rescale=None,
        preprocessing_function=None, data_format=None, validation_split=0.0, dtype=None)

    data_gen.fit(data)
    res = model.fit_generator(data_gen.flow(data, label,
                                            batch_size=parameter.BATCH_SIZE,
                                            shuffle=True,
                                            subset="training"),
                              epochs=parameter.EPOCH_NUM,
                              validation_data=data_gen.flow(x_test, y_test,
                                                            batch_size=8,
                                                            shuffle=True,
                                                            subset="validation"),
                              shuffle=True,
                              verbose=1,
                              steps_per_epoch=len(data) / 32,
                              validation_steps=(5 * parameter.GEN_RATE * parameter.CLASS_NUM) / 32,
                              # validation_batch_size=parameter.BATCH_SIZE,
                              callbacks=[tensorboard])
    plot_training_history(history=res.history)
    print(model.summary())
    model_evaluate(model, x_test, y_test)
    prediction.model_prediction(parameter.MODEL, x_test, y_test)


if __name__ == '__main__':
    all_train_images, all_train_labels = load_image.load_gen_data(parameter.GEN_TRAIN)
    all_val_images, all_val_labels = load_image.load_gen_data(parameter.GEN_VAL)
    all_train_images = np.array(all_train_images)
    all_train_labels = np.array(all_train_labels)
    all_val_images = np.array(all_val_images)
    all_val_labels = np.array(all_val_labels)

    print("original ->")
    print(type(all_train_images), type(all_train_labels))
    print(all_train_images.shape, all_train_labels.shape)
    print(type(all_val_images), type(all_val_labels))
    print(all_val_images.shape, all_val_labels.shape)

    all_train_images, all_train_labels, all_val_images, all_val_labels = \
        flatten(all_train_images), flatten(all_train_labels), flatten(all_val_images), flatten(all_val_labels)
    all_train_labels, all_val_labels = \
        to_categorical(all_train_labels, parameter.CLASS_NUM), to_categorical(all_val_labels, parameter.CLASS_NUM)
    shu_index = np.arange(all_train_images.shape[0])
    np.random.shuffle(shu_index)
    all_train_images = all_train_images[shu_index, :, :, :]
    all_train_labels = all_train_labels[shu_index, :]
    shu_index = np.arange(all_val_images.shape[0])
    np.random.shuffle(shu_index)
    all_val_images = all_val_images[shu_index, :, :, :]
    all_val_labels = all_val_labels[shu_index, :]

    print("after preprocess ->")
    print(type(all_train_images), type(all_train_labels))
    print(all_train_images.shape, all_train_labels.shape)
    print(type(all_val_images), type(all_val_labels))
    print(all_val_images.shape, all_val_labels.shape)

    train_model_fit(all_train_images, all_train_labels,
                    parameter.IMG_SIZE,
                    all_val_images, all_val_labels, model_select=1)
    train_model_fit(all_train_images, all_train_labels,
                    parameter.IMG_SIZE,
                    all_val_images, all_val_labels, model_select=2)
    # train_model_gen(all_train_images, all_train_labels,
    #                 parameter.IMG_SIZE,
    #                 all_val_images, all_val_labels, model_select=1)
