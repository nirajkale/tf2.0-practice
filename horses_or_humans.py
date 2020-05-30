from plots import *
from mycallbacks import *

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import models, layers, optimizers, regularizers
from tensorflow.data.experimental import AUTOTUNE

IMG_HEIGHT, IMG_WIDTH = 300,300

ds_train, ds_info_train = tfds.load('horses_or_humans', split = tfds.Split.TRAIN, with_info=True)
ds_val, ds_info_val = tfds.load('horses_or_humans', split = tfds.Split.TEST, with_info=True)

def augment(features):
    image = features['image']
    label = features['label']
    image = tf.image.random_brightness(image, max_delta= 0.5)
    image = tf.image.random_contrast(image, lower=0.5, upper=3)
    # image = tf.image.random_hue(image,max_delta=0.1)
    image = tf.image.flip_left_right(image)
    image = tf.image.resize_with_crop_or_pad(image, target_height= IMG_HEIGHT, target_width= IMG_WIDTH)
    return image, label

def preprocess(features):
    image = features['image']
    label = features['label']
    image = tf.image.resize_with_crop_or_pad(image, target_height= IMG_HEIGHT, target_width= IMG_WIDTH)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

training_pipeline = ds_train.shuffle(1024).map(augment).batch(32).prefetch(AUTOTUNE)
val_pipeline = ds_val.shuffle(1024).map(preprocess).batch(32).prefetch(AUTOTUNE)

model = models.Sequential([
    # 300x300x3
    layers.Conv2D(filters=16, kernel_size=5, activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)), # 298x298x16
    layers.MaxPool2D(pool_size=(2,2), strides=2), #149x149x16
    layers.Dropout(rate=0.1),
    layers.Conv2D(filters=32, kernel_size=3, activation='relu'), #147x147x32
    layers.MaxPool2D(pool_size=(2,2), strides=2), #73x73x32
    layers.Dropout(rate=0.1),
    layers.Conv2D(filters=64, kernel_size=3, activation='relu'), #147x147x32
    layers.MaxPool2D(pool_size=(2,2), strides=2), #73x73x32
    layers.Dropout(rate=0.1),
    # layers.Conv2D(filters=64, kernel_size=3, activation='relu'), #71x71x64
    # layers.MaxPool2D(pool_size=(2,2), strides=2), #35x35x64
    # layers.Dropout(rate=0.2),
    layers.Flatten(),
    layers.Dense(units=512, activation='relu'),
    layers.Dense(units=1, activation='sigmoid')
])

print(model.summary())
model.compile(optimizer= optimizers.RMSprop(), loss='binary_crossentropy', metrics=['acc'])

my_callbacks = [
    callbacks.EarlyStopping(monitor='val_acc', min_delta=0.01, patience=7),
    # StopOnThreshold(0.93, 0.93)
]

history = model.fit(training_pipeline, batch_size=32,\
    epochs= 5,\
    callbacks= my_callbacks,\
    validation_data= val_pipeline).history

plot_history(history)
print('train:', model.evaluate(training_pipeline))
print('val:', model.evaluate(val_pipeline))