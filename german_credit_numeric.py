import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import models, layers, optimizers, regularizers, callbacks
from tensorflow.data.experimental import AUTOTUNE
from plots import *
from mycallbacks import *
import numpy as np

feature_count = 24
batch_size = 10

# ds_all, ds_info = tfds.load('german_credit_numeric', split= tfds.Split.TRAIN, with_info = True)
ds_train, ds_info_train = tfds.load('german_credit_numeric', split= 'train[:80%]', with_info = True)
ds_val, ds_info_val = tfds.load('german_credit_numeric', split= 'train[-20%:]', with_info = True)
rows = np.array([list(item['features'].numpy()) for item in iter(ds_train)])
ds_mean = tf.convert_to_tensor(np.mean(rows, axis=0), tf.float32)
ds_std = tf.convert_to_tensor(np.std(rows, axis=0), tf.float32)

def preprocess(features):
    data = features['features']
    data = tf.cast(data, tf.float32)
    data = (data- ds_mean)/ ds_std
    label = features['label']
    # label = tf.one_hot(label, depth= num_classes)
    return data, label

training_pipeline = ds_train.shuffle(1024).map(preprocess).batch(batch_size).prefetch(AUTOTUNE)
val_pipeline = ds_val.shuffle(1024).map(preprocess).batch(batch_size).prefetch(AUTOTUNE)

model = models.Sequential([
    layers.Dense(24, activation='sigmoid', input_shape=(feature_count,), kernel_regularizer= regularizers.l2(l=0.005)),
    layers.Dropout(0.2),
    layers.Dense(6, activation='relu', kernel_regularizer= regularizers.l2(l=0.001)),
    layers.Dense(1, activation='sigmoid'),
])

print(model.summary())
model.compile(optimizer= optimizers.RMSprop(learning_rate=0.0001), loss='binary_crossentropy', metrics=['acc'])

my_callbacks = [
    # callbacks.EarlyStopping(monitor='val_acc', min_delta=0.01, patience=25),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_delta=0.005),
    StopOnThreshold(0.95, 0.95)
]

history = model.fit(training_pipeline, batch_size=batch_size,\
    epochs= 250,\
    callbacks= my_callbacks,\
    validation_data= val_pipeline).history

plot_history(history)
print('train:', model.evaluate(training_pipeline))
print('val:', model.evaluate(val_pipeline))

