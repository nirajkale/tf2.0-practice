import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import models, layers, optimizers, regularizers, callbacks
from tensorflow.data.experimental import AUTOTUNE
from plots import *
from mycallbacks import *

ds_train, ds_info_train = tfds.load('citrus_leaves', split= 'train[:80%]', with_info = True)
ds_val, ds_info_val = tfds.load('citrus_leaves', split= 'train[:-20%]', with_info = True)
num_classes=4
IMG_HEIGHT, IMG_WIDTH = 256, 256

def preprocess(features):
    image = features['image']
    label = features['label']
    image = tf.image.resize_with_crop_or_pad(image, target_height= IMG_HEIGHT, target_width= IMG_WIDTH)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    label = tf.one_hot(label, num_classes)
    return image, label

training_pipeline = ds_train.shuffle(1024).map( preprocess).batch(32).prefetch(AUTOTUNE)
val_pipeline = ds_val.shuffle(1024).map( preprocess).batch(32).prefetch(AUTOTUNE)

#256x256x3
model = models.Sequential([ 
    layers.Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH,3)), #254x254x16
    layers.MaxPool2D(pool_size=(2,2), strides=2), #127x127x16
    layers.Dropout(0.2),
    layers.Conv2D(filters=32, kernel_size=3, activation='relu'), #125x125x32
    layers.MaxPool2D(pool_size=(2,2), strides=2),#62x62x32
    layers.Dropout(0.2),
    layers.Conv2D(filters=64, kernel_size=3, activation='relu'),#60x60x64
    layers.MaxPool2D(pool_size=(2,2), strides=2), #30x30x64
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
print(model.summary())
model.compile(optimizer= optimizers.RMSprop(), loss='categorical_crossentropy', metrics=['acc'])

my_callbacks = [
    callbacks.EarlyStopping(monitor='val_acc', min_delta=0.01, patience=4),
    StopOnThreshold(0.95, 0.95)
]

history = model.fit(training_pipeline, batch_size=32,\
    epochs= 20,\
    callbacks= my_callbacks,\
    validation_data= val_pipeline).history

plot_history(history)
print('train:', model.evaluate(training_pipeline))
print('val:', model.evaluate(val_pipeline))

