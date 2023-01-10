import logging

logging.getLogger("tensorflow").setLevel(logging.WARNING)

import cv2
import numpy as np
import os
import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import math


# def define_model():
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(100, 100, 1)))
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Flatten())
#     model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
#     model.add(Dense(26, activation='softmax'))
#     # compile model
#     opt = SGD(learning_rate=0.01, momentum=0.9)
#     model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
#     return model


def LeNet5():
    model = Sequential([
        Conv2D(filters=32, kernel_size=5, strides=1, activation='relu', input_shape=(50, 50, 1),
               kernel_regularizer=l2(0.0005), name='convolution_1'),
        Conv2D(filters=32, kernel_size=5, strides=1, name='convolution_2', use_bias=False),
        BatchNormalization(name='batchnorm_1'),
        Activation("relu"),
        MaxPooling2D(pool_size=2, strides=2, name='max_pool_1'),
        Dropout(0.25, name='dropout_1'),
        Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', kernel_regularizer=l2(0.0005),
               name='convolution_3'),
        Conv2D(filters=64, kernel_size=3, strides=1, name='convolution_4', use_bias=False),
        BatchNormalization(name='batchnorm_2'),
        Activation("relu"),
        MaxPooling2D(pool_size=2, strides=2, name='max_pool_2'),
        Dropout(0.25, name='dropout_2'),
        Flatten(name='flatten'),
        Dense(units=256, name='fully_connected_1', use_bias=False),
        BatchNormalization(name='batchnorm_3'),
        Activation("relu"),
        Dense(units=128, name='fully_connected_2', use_bias=False),
        BatchNormalization(name='batchnorm_4'),
        Activation("relu"),
        Dense(units=84, name='fully_connected_3', use_bias=False),
        BatchNormalization(name='batchnorm_5'),
        Activation("relu"),
        Dropout(0.25, name='dropout_3'),
        Dense(units=26, activation='softmax', name='output')
    ])
    model._name = 'LeNet5v2'
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def prep_pixels(img):
    return img.astype('float32') / 255.0


def load_data():
    Y = []
    X = []

    for i in tqdm(os.listdir("./data")):
        for j in os.listdir("./data/" + str(i)):
            img = cv2.imread("./data/" + str(i) + "/" + str(j), 0)
            img = cv2.resize(img, (50, 50))
            img = prep_pixels(img)
            Y.append(i)
            X.append(img)

    encoder = LabelBinarizer()
    Y = encoder.fit_transform(Y)
    X = np.array(X)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle=True)
    _, X_val, _, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)
    return X_train, X_test, X_val, y_train, y_test, y_val


X_train, X_test, X_val, y_train, y_test, y_val = load_data()
X_train = tf.expand_dims(X_train, axis=-1)
X_test = tf.expand_dims(X_test, axis=-1)
X_val = tf.expand_dims(X_val, axis=-1)

print(X_train.shape)
print(X_test.shape)
print(X_val.shape)
print(y_train.shape)
print(y_test.shape)
# model = define_model()
model = LeNet5()

# model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)
model.save("typing_model_2")

reconstructed_model = tf.keras.models.load_model("typing_model")
_, acc = reconstructed_model.evaluate(X_val, y_val, verbose=1)
print(reconstructed_model.predict(X_val).argmax(axis=-1))
print(acc)