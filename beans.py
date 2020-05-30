import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from tensorflow.data.experimental import AUTOTUNE
from tensorflow.keras import models, optimizers, layers, regularizers,callbacks
import matplotlib.pyplot as plt
import time
from mycallbacks import StopOnThreshold

print('TF:', tf.__version__)

IMAGE_HEIGHT, IMAGE_WIDTH = 500, 500
data_dir = os.path.join(os.path.dirname(__file__), 'data')
ds_train, train_info= tfds.load('beans', split=tfds.Split.TRAIN, with_info=True, data_dir=data_dir)
ds_val, val_info= tfds.load('beans', split=tfds.Split.VALIDATION, with_info=True, data_dir=data_dir)

def augment(features):
    image = features['image']
    label = features['label']
    image = tf.image.random_brightness(image, max_delta=0.5)
    image = tf.image.random_contrast(image, 0.5, 4)
    image = tf.image.random_hue(image, max_delta=0.1)
    image = tf.image.flip_left_right(image)
    image = tf.image.resize_with_crop_or_pad(image, IMAGE_HEIGHT, IMAGE_WIDTH)
    image = tf.image.convert_image_dtype(image, tf.float32)
    label = tf.one_hot(label, depth=3)
    return image, label

def preprocess(features):
    image = features['image']
    label = features['label']
    image = tf.image.resize_with_crop_or_pad(image, IMAGE_HEIGHT, IMAGE_WIDTH)
    image = tf.image.convert_image_dtype(image, tf.float32)
    label = tf.one_hot(label, depth=3)
    return image, label

training_pipeline = ds_train.shuffle(1024).map(augment).batch(32).prefetch(AUTOTUNE)
val_pipeline = ds_val.shuffle(1024).map(preprocess).batch(32).prefetch(AUTOTUNE)

model = models.Sequential(layers=[
    layers.Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH,3)), #496x496x16
    layers.MaxPool2D(pool_size=(2,2), strides=2), #248 x248x 16
    layers.Conv2D(filters=32, kernel_size=3, activation='relu'),# 246x246x32
    layers.MaxPool2D(pool_size=(2,2), strides=2), # 122 x122x 32
    layers.Conv2D(filters=64, kernel_size=3, activation='relu'),# 120x120x64
    layers.MaxPool2D(pool_size=(2,2), strides=2), # 59 x59x 64
    layers.Conv2D(filters=128, kernel_size=3, activation='relu'),# 57x57x64
    layers.MaxPool2D(pool_size=(2,2), strides=2), # 27 x27x 128
    layers.Conv2D(filters=128, kernel_size=3, activation='relu'),# 57x57x64
    layers.MaxPool2D(pool_size=(2,2), strides=2), # 27 x27x 128
    layers.Conv2D(filters=128, kernel_size=3, activation='relu'),# 57x57x64
    layers.MaxPool2D(pool_size=(2,2), strides=2), # 27 x27x 128
    layers.Flatten(),
    layers.Dropout(rate=0.2),
    layers.Dense(512, activation='relu', kernel_regularizer= regularizers.l2(l=0.006)),
    layers.Dense(32, activation='relu', kernel_regularizer= regularizers.l2(l=0.004)),
    layers.Dense(3, activation='softmax'),
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())

my_callbacks = [
    callbacks.EarlyStopping(monitor='val_acc', min_delta=0.01, patience=7),
    StopOnThreshold(0.87, 0.87)
]
start = time.time()
history = model.fit(training_pipeline, \
    epochs=25, batch_size=32,\
    callbacks = my_callbacks,\
    validation_data= val_pipeline).history
end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("Time taken:  {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

def plot_history(history, attribute):
    fig = plt.figure()
    plt.plot(history[attribute], c='r')
    plt.plot(history['val_'+attribute], c='b')
    plt.title(attribute+' History')
    plt.show()

plot_history(history, 'loss')
plot_history(history, 'acc')
print(model.evaluate(val_pipeline))
model.save('beans.h5')