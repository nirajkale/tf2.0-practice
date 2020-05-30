import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import models, layers, callbacks, optimizers, regularizers
from tensorflow.data.experimental import AUTOTUNE

from plots import *
from mycallbacks import StopOnThreshold
import matplotlib.pyplot as plt

ds_train, ds_info_train = tfds.load('tf_flowers', split='train[:80%]', with_info=True, data_dir='data')
ds_val, ds_info_val = tfds.load('tf_flowers', split='train[-20%:]', with_info=True, data_dir='data')
num_classes=5
IMG_HEIGHT, IMG_WIDTH = 250,250

def augment(features):
    image = features['image']
    label = features['label']
    image = tf.image.random_brightness(image, max_delta=0.5)
    image = tf.image.random_contrast(image, lower=0.8, upper=3)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize_with_crop_or_pad(image, IMG_HEIGHT+35, IMG_WIDTH+35)
    image = tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, 3]) # Random crop back to 28x28
    image = tf.image.convert_image_dtype(image, tf.float32)
    label = tf.one_hot(label, num_classes)
    return image, label

def preprocess(features):
    image = features['image']
    label = features['label']
    image = tf.image.resize_with_crop_or_pad(image, IMG_HEIGHT, IMG_WIDTH)
    image = tf.image.convert_image_dtype(image, tf.float32)
    label = tf.one_hot(label, num_classes)
    return image, label

training_pipeline = ds_train.shuffle(1024).map(preprocess).batch(32).prefetch(AUTOTUNE)
val_pipeline = ds_val.shuffle(1024).map(preprocess).batch(32).prefetch(AUTOTUNE)

# sample_data, sample_label = next(iter(training_pipeline))

model = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=5, activation='relu',input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPool2D(pool_size=(2,2), strides=2),
    layers.Dropout(rate=0.2),
    layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
    layers.MaxPool2D(pool_size=(2,2), strides=2),
    layers.Dropout(rate=0.2),
    layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
    layers.MaxPool2D(pool_size=(2,2), strides=2),
    layers.Dropout(rate=0.2),
    # layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
    # layers.MaxPool2D(pool_size=(2,2), strides=2),
    # layers.Dropout(rate=0.25),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(units=num_classes, activation='softmax')
])

print(model.summary())
model.compile(optimizer= 'rmsprop', loss='categorical_crossentropy', metrics=['acc'])

my_callbacks = [
    callbacks.EarlyStopping(monitor='val_acc', min_delta=0.01, patience=9),
    StopOnThreshold(0.9, 0.9)
]

history = model.fit(training_pipeline, batch_size=32,\
    epochs= 30,\
    callbacks= my_callbacks,\
    validation_data= val_pipeline).history

plot_history(history)
print('train:', model.evaluate(training_pipeline))
print('val:', model.evaluate(val_pipeline))