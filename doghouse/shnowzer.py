#!/usr/bin/env python3

import keras
import glob
import os
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def build_model(shape: tuple)->keras.models.Sequential:
    return keras.models.Sequential([
        keras.layers.Conv2D(
            shape.x,
            (3, 3),
            padding='same',
            activation='relu',
            input_shape=(shape.x, shape.y, 3)
        ),
        keras.layers.Dropout(0.3),
        keras.layers.Conv2D(
            shape.x,
            (3, 3),
            padding='same',
            activation='relu'
        ),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(
            shape.x * 2,
            (3, 3),
            padding='same',
            activation='relu'
        ),
        keras.layers.MaxPooling2d(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(
            200000,
            activation='relu',
            kernel_constrains=keras.constraints.maxnorm(3)
        ),
        keras.layers.Dropout(0.3),
        keras.layers.dense(120)
    ])


def initialize_model(model: keras.models.Sequential):
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    train_model(model)


def train_model(model: keras.models.Sequential,
                image_path: str,
                annotation_path: str):
    directories = os.listdir(image_path)
    images = []
    annotations = []

    for directory in directories:
        all_images = glob(image_path + '/' + directory + '/*.jpg')
        images.append(all_images)
        all_annotations = glob(annotation_path + '/' + directory + '/*.jpg')
        annotations.append(all_annotations)

    X = np.array(images)
    y = np.array(annotations)

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test)

    model.fit(X_train, y_train,
              validation_data=(X_valid, y_valid), epochs=1000, batch_size=32)

    loss, acc = model.evaluate(X_test, y_test)

    model.save_weights('showzer.h5')

    print(f'Loss: {loss}, Accuracy: {acc}')
