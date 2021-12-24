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

import parameter
import generate
import load_image
import cnn_model_one
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import random


def model_evaluate(model, x_test, y_test):
    score = model.evaluate(x_test, y_test)
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
        model = cnn_model_one.cnn_model_one(size)
    else:
        print("model_select error!")

    model.compile(loss="categorical_crossentropy",
                  optimizer="Adam",
                  metrics=["accuracy"])
    tensorboard = TensorBoard(parameter.LOG)

    history = model.fit(data, label,
                        batch_size=parameter.BATCH_SIZE,
                        epochs=parameter.EPOCH_NUM,
                        callbacks=[tensorboard],
                        shuffle=True,
                        verbose=1,
                        steps_per_epoch=(95 * parameter.GEN_RATE * parameter.CLASS_NUM)/32,
                        validation_steps=(5 * parameter.GEN_RATE * parameter.CLASS_NUM)/32,
                        validation_batch_size=parameter.BATCH_SIZE,
                        validation_split=0.2)
    plot_training_history(history)
    model_evaluate(model, x_test, y_test)


def train_model_gen(data, label, size, x_test, y_test, model_select=1):
    model = None
    if model_select == 1:
        model = cnn_model_one.cnn_model_one(size)
    elif model_select == 2:
        model = cnn_model_one.cnn_model_one(size)
    else:
        print("model_select error!")
    model.compile(loss="categorical_crossentropy",
                  optimizer="Adam",
                  metrics=["accuracy"])
    tensorboard = TensorBoard(parameter.LOG)

    data_gen = ImageDataGenerator(
        featurewise_center=False,  # 是否使输入数据去中心化（均值为0），
        samplewise_center=False,  # 是否使输入数据的每个样本均值为0
        featurewise_std_normalization=False,  # 是否数据标准化（输入数据除以数据集的标准差）
        samplewise_std_normalization=False,  # 是否将每个样本数据除以自身的标准差
        zca_whitening=False,  # 是否对输入数据施以ZCA白化
        rotation_range=90,  # 数据提升时图片随机转动的角度(范围为0～180)
        width_shift_range=0.25,  # 数据提升时图片水平偏移的幅度（单位为图片宽度的占比，0~1之间的浮点数）
        height_shift_range=0.25,  # 同上，只不过这里是垂直
        horizontal_flip=False,  # 是否进行随机水平翻转
        vertical_flip=False)  # 是否进行随机垂直翻转

    data_gen.fit(data)
    res = model.fit_generator(generator=data_gen.flow(data, label, batch_size=parameter.BATCH_SIZE),
                              epochs=parameter.EPOCH_NUM,
                              validation_split=0.2,
                              shuffle=True,
                              verbose=1,
                              steps_per_epoch=(95 * parameter.GEN_RATE * parameter.CLASS_NUM) / 32,
                              validation_steps=(5 * parameter.GEN_RATE * parameter.CLASS_NUM) / 32,
                              validation_batch_size=parameter.BATCH_SIZE,
                              callbacks=[tensorboard])
    plot_training_history(history=res.history)
    model_evaluate(model, x_test, y_test)


if __name__ == '__main__':
    all_train_images, all_train_labels = load_image.load_data(parameter.GEN_TRAIN)
    all_val_images, all_val_labels = load_image.load_data(parameter.GEN_VAL)
    print(type(all_train_images), type(all_train_labels))
