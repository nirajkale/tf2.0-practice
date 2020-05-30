from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
from plots import *
from mycallbacks import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, optimizers, regularizers, layers, callbacks
print('TF:', tf.__version__)


data_dir = os.path.join(os.path.dirname(__file__), 'data')

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

datagen_augment = ImageDataGenerator(featurewise_center= True,\
    featurewise_std_normalization= True,\
        width_shift_range=0.1,\
        shear_range=0.1,\
        zoom_range= 0.1)

datagen_non_augment = ImageDataGenerator(featurewise_center= True,\
    featurewise_std_normalization= True)
datagen_augment.fit(x_train)
datagen_non_augment.fit(x_train)

train_gen = datagen_augment.flow(x_train, y_train, batch_size= 32)
val_gen = datagen_non_augment.flow(x_test, y_test, batch_size= 32)
steps_per_epoch_train = int(len(x_train)/32)
steps_per_epoch_val = int(len(x_train)/32)

model = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(32,32,3)), # 30x30x16
    layers.MaxPool2D(pool_size=(2,2), strides=2), # 15x15x26
    layers.Dropout(rate = 0.2),
    layers.Conv2D(filters=64, kernel_size=3, activation='relu'), # 13x13x32
    layers.MaxPool2D(pool_size=(2,2), strides=2), # 6x6x32
    layers.Dropout(rate = 0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
print( model.summary())
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

my_callbacks = [
    callbacks.EarlyStopping(monitor='val_acc', min_delta=0.01, patience=4),
    StopOnThreshold(0.87, 0.87)
]

history = model.fit_generator(train_gen, 
    # steps_per_epoch= steps_per_epoch_train,\
    # validation_steps= steps_per_epoch_val,
    epochs= 25,\
    validation_data= val_gen,
    callbacks= my_callbacks
    ).history

plot_history(history)








