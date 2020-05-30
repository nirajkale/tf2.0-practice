import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import models, layers, optimizers, regularizers, callbacks
from tensorflow.data.experimental import AUTOTUNE
from plots import *
from mycallbacks import *

num_classes=3
feature_count = 4
batch_size = 10
ds_train, ds_info_train = tfds.load('iris', split= 'train[:85%]', with_info = True)
ds_val, ds_info_val = tfds.load('iris', split= 'train[-15%:]', with_info = True)

def preprocess(features):
    data = features['features']
    label = features['label']
    label = tf.one_hot(label, depth= num_classes)
    return data, label

training_pipeline = ds_train.shuffle(1024).map(preprocess).batch(batch_size).prefetch(AUTOTUNE)
val_pipeline = ds_val.shuffle(1024).map(preprocess).batch(batch_size).prefetch(AUTOTUNE)

model = models.Sequential([
    layers.Dense(9, activation='sigmoid', input_shape=(feature_count,), kernel_regularizer= regularizers.l2(l=0.007)),
    layers.Dropout(0.15),
    layers.Dense(num_classes, activation='softmax'),
])
print(model.summary())
model.compile(optimizer= 'rmsprop', loss='categorical_crossentropy', metrics=['acc'])

my_callbacks = [
    # callbacks.EarlyStopping(monitor='val_acc', min_delta=0.01, patience=25),
    StopOnThreshold(0.95, 0.95)
]

history = model.fit(training_pipeline, batch_size=batch_size,\
    epochs= 100,\
    callbacks= my_callbacks,\
    validation_data= val_pipeline).history

plot_history(history)
print('train:', model.evaluate(training_pipeline))
print('val:', model.evaluate(val_pipeline))