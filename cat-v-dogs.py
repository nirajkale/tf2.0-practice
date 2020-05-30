import tensorflow as tf
from tensorflow.data.experimental import AUTOTUNE
import tensorflow_datasets as tfds
from tensorflow.keras import layers, optimizers, regularizers, models
import matplotlib.pyplot as plt
print('TF Version:', tf.__version__)
import time

ds_train, info_train = tfds.load('cats_vs_dogs', split= 'train[:80%]', with_info= True)
ds_val, info_val = tfds.load('cats_vs_dogs', split= 'train[:-20%]', with_info= True)

IMAGE_WIDTH = 250
IMAGE_HEIGHT = 250

def preprocess(features):
    image = features['image']
    label = features['label']
    image = tf.image.random_brightness(image, max_delta=0.5)
    image = tf.image.random_contrast(image, lower=0.8, upper=3)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize_with_crop_or_pad(image, IMAGE_HEIGHT, IMAGE_WIDTH)
    image = tf.image.convert_image_dtype(image, dtype= tf.float32)
    return (image, label)

def preprocess_val(features):
    image = features['image']
    label = features['label']
    image = tf.image.resize_with_crop_or_pad(image, IMAGE_HEIGHT, IMAGE_WIDTH)
    image = tf.image.convert_image_dtype(image, dtype= tf.float32)
    return (image, label)

def prepare_model():
    model_in = layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH,3), dtype= tf.float32) #(250, 250, 3)
    a = layers.Conv2D(filters=16, kernel_size= 3, activation='relu')(model_in) #(248, 248, 16)
    a = layers.MaxPool2D(pool_size=(2,2), strides=2)(a) #(124, 124, 16)
    a = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(a) #(122, 122, 32)
    a = layers.MaxPool2D(pool_size=(2,2), strides=2)(a) #(61, 61, 32)
    a = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(a)  # (59, 59, 64)
    a = layers.MaxPool2D(pool_size=(2, 2), strides=2)(a)  # (29, 29, 64)
    a = layers.Flatten()(a)
    a = layers.Dense(64, activation='relu')(a)
    model_out = layers.Dense(1, activation='sigmoid')(a)
    model = models.Model(model_in, model_out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    print(model.summary())
    return model

def plot_history(history, attribute):
    fig = plt.figure()
    plt.plot(history[attribute], c='r')
    plt.plot(history['val_'+attribute], c='b')
    plt.title(attribute+' History')
    plt.show()

start = time.time()
training_pipeline = ds_train.shuffle(1024).map(preprocess).batch(32).prefetch(AUTOTUNE)
val_pipeline = ds_train.shuffle(1024).map(preprocess).batch(32).prefetch(AUTOTUNE)
model = prepare_model()

history = model.fit(training_pipeline, batch_size= 32, epochs=15, verbose=1, validation_data= val_pipeline).history
end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
# plot_history(history, 'loss')
# plot_history(history, 'acc')
print('done')