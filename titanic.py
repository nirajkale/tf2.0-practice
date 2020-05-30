import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import models, layers, optimizers, regularizers, callbacks
from tensorflow.data.experimental import AUTOTUNE
from plots import *
from mycallbacks import *
import numpy as np

num_classes=3
feature_count = 4
batch_size = 10

ds_all, ds_info = tfds.load('titanic', split= tfds.Split.TRAIN, with_info = True)
ds_train, ds_info_train = tfds.load('titanic', split= 'train[:80%]', with_info = True)
ds_val, ds_info_val = tfds.load('titanic', split= 'train[-20%:]', with_info = True)

def generate_string_lookups(ds):
    sample = next(iter(ds))
    feature_dict = dict([(feature, []) for feature in sample['features'] if sample['features'][feature].dtype== tf.string])
    for item in iter(ds):
        for feature, value in item['features'].items():
            if feature in feature_dict:
                feature_dict[feature].append(value.numpy())
    for feature,values in feature_dict.items():
        feature_dict[feature] = tf.constant(np.unique(values))
    return feature_dict
            
feature_dict = generate_string_lookups(ds_all)
boat_lookup  = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(feature_dict['boat'], tf.range(0, len(feature_dict['boat']))), -1)
cabin_lookup  = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(feature_dict['cabin'], tf.range(0, len(feature_dict['cabin']))), -1)

def preprocess(features):
    data = features['features']
    age = data['age']
    body = data['body']
    boat = boat_lookup.lookup(data['boat'])
    cabin = cabin_lookup.lookup(data['cabin'])
    embarked = data['embarked']
    fare = data['fare']
    parch = data['parch']
    pclass = data['pclass']
    sex = data['sex']
    sibsp = data['sibsp']
    data = tf.convert_to_tensor([age, body, boat, cabin, embarked, fare, parch, pclass, sex, sibsp], tf.float32)
    label = features['survived']
    return data, label

training_pipeline = ds_train.shuffle(1024).map(preprocess).batch(batch_size).prefetch(AUTOTUNE)
# val_pipeline = ds_val.shuffle(1024).map(preprocess).batch(batch_size).prefetch(AUTOTUNE)

print(ds_info_val)