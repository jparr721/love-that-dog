#!/usr/bin/env python3

import keras
import glob
import os
import sys
import random
import numpy as np
import xml.etree.ElementTree as ET
from itertools import chain
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Doofus(Exception):
    pass


def build_model(shape: tuple)->keras.models.Sequential:
    print('Using input shape: {}'.format(shape))
    return keras.models.Sequential([
        keras.layers.Conv2D(
            shape[0],
            (3, 3),
            padding='same',
            activation='relu',
            input_shape=(shape[0], shape[1], shape[2])
        ),
        keras.layers.Dropout(0.3),
        keras.layers.Conv2D(
            shape[0],
            (3, 3),
            padding='same',
            activation='relu'
        ),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(
            shape[0],
            (3, 3),
            padding='same',
            activation='relu'
        ),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(
            shape[0] * shape[1],
            activation='relu'
        ),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(120)
    ])


def initialize_model(model: keras.models.Sequential,
                     image_path: str,
                     annotation_path: str):
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    train_model(model, image_path, annotation_path)


def flatten(ls: list):
    return [item for subitem in ls for item in subitem]


def fix_images(image_path: str):
    directories = os.listdir(image_path)
    images = []

    for d in directories:
        all_img = glob.glob(image_path + '/' + d + '/*.jpg')
        images.append(all_img)

    images = flatten(images)

    for i in range(len(images)):
        print('Fixing image: {}'.format(images[i]))
        img = Image.open(images[i])
        img = img.resize((50, 50))
        new_path = images[i][:len(images[i]) - 3]
        new_path += 'png'
        print('Saving to: {}'.format(new_path))
        # Save over the existing images
        img.save(new_path)

    print('done')


def fix_annotations(annotation_path: str):
    directories = os.listdir(annotation_path)
    for d in directories:
        anno = d.split('-')[1].capitalize()
        if '_' in anno:
            anno.replace('_', ' ')
            strings = anno.split(' ')
            for s in range(len(strings)):
                strings[s] = strings[s].capitalize()
            anno = ' '.join(strings)
        all_anno = glob.glob(annotation_path + '/' + d + '/*')
        for ann in all_anno:
            with open(ann, 'w') as anno_file:
                anno_file.write(anno)
    print('done')


def train_model(model: keras.models.Sequential,
                image_path: str,
                annotation_path: str):
    directories = os.listdir(image_path)
    images = []
    annotations = []

    for directory in directories:
        all_img_path = image_path + '/' + directory + '/*.png'
        all_images = glob.glob(all_img_path)
        images.append(all_images)
        all_annotation_path = annotation_path + '/' + directory + '/*'
        all_annotations = glob.glob(all_annotation_path)
        annotations.append(all_annotations)

    # Place images in one flat list
    images = flatten(images)
    annotations = flatten(annotations)
    for i in range(len(annotations)):
        annotations[i] = open(annotations[i], 'r').readline()

    le = LabelEncoder()

    X = np.array([np.expand_dims(img_to_array(load_img(i)), axis=0) for i in images])
    X = np.reshape(X, (len(X), 50, 50, 3))
    print(f'Using input with shape: {X.shape}')
    print(f'Loaded {len(X)} images')

    y = np.array(annotations)
    y = le.fit_transform(y)
    y = keras.utils.to_categorical(y, 120)
    print(f'Using labels with shape: {y.shape}')

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test)

    model.fit(X_train, y_train,
              validation_data=(X_valid, y_valid), epochs=1000, batch_size=32)

    loss, acc = model.evaluate(X_test, y_test)

    json = model.to_json()
    with open('woof.json', 'w') as json_file:
        json_file.write(json)
    model.save_weights('woof.h5')

    print(f'Loss: {loss}, Accuracy: {acc}')


def do_the_thing_bro(image: list)->str:
    json = open('woof.json', 'r')
    loaded_json = json.read()
    json.close()
    the_model = keras.models.model_from_json(loaded_json)
    the_model.load_weights('woof.h5')

    the_model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['accuracy'])

    return the_model.predict_classes(np.array(image))


if __name__ == '__main__':
    opt = sys.argv[1]
    if not opt.startswith('--'):
        raise Doofus('Idiot, your flags are wrong')
    elif opt not in ['--fix_images', '--fix_annotations', '--go', '--train']:
        raise Doofus('Dude, your flags don\'t even exist!')

    if opt == '--train':
        initialize_model(build_model((50, 50, 3)),
                         './Images', './Annotation')

    if opt == '--fix_images':
        fix_images('./Images')

    if opt == '--fix_annotations':
        fix_annotations('./Annotation')

    if opt == '--go':
        if len(sys.argv) < 3:
            raise('Error bro, we need more than that')
        do_the_thing_bro(sys.argv[2])
