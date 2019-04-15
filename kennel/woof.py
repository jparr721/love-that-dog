#!/usr/bin/env python3

import keras
import glob
import os
import sys
import random
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from sklearn.model_selection import train_test_split
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Doofus(Exception):
    pass


def build_model(shape: tuple)->keras.models.Sequential:
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
            shape[0] * 2,
            (3, 3),
            padding='same',
            activation='relu'
        ),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(
            2500,
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

    dimx = 50
    dimy = 50
    for i in range(len(images)):
        print('Fixing image: {}'.format(images[i]))
        img = Image.open(images[i])
        img.resize((dimx, dimy), Image.ANTIALIAS)
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
        all_img_path = image_path + '/' + directory + '/*.jpg'
        all_images = glob.glob(all_img_path)
        images.append(all_images)
        all_annotation_path = annotation_path + '/' + directory + '/*'
        all_annotations = glob.glob(all_annotation_path)
        annotations.append(all_annotations)

    # Take random sample of images
    images = [random.sample(sublist, len(sublist) // 3) for sublist in images]

    # Open all of the images
    images = flatten(images)
    annotations = flatten(annotations)

    # Ram destroyer 9000
    X = np.array(flatten([list(Image.open(i, 'r').getdata()) for i in images]))
    y = np.array(annotations)
    print(X)

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
    the_img = np.array(img)

    return the_model.predict_classes(the_img)


if __name__ == '__main__':
    opt = sys.argv[1]
    if not opt.startswith('--'):
        raise Doofus('Idiot, your flags are wrong')

    if opt == '--train':
        initialize_model(build_model((400, 500, 3)),
                         './Images', './Annotation')

    if opt == '--fix_images':
        fix_images('./Images')

    if opt == '--fix_annotations':
        fix_annotations('./Annotation')

    if opt == '--go':
        if len(sys.argv) < 3:
            raise('Error bro, we need more than that')
        do_the_thing_bro(sys.argv[2])
