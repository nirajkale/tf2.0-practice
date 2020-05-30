import re
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import models, optimizers, callbacks, layers
import numpy as np

from plots import *
from mycallbacks import *

def clean_text(text):
    text = text.encode('ascii',errors='ignore').decode()
    text = ' '.join(re.findall('[a-zA-Z]+',text))
    text = re.sub('\s{2,}',' ', text)
    return text

def fetch_data():
    with open('data/sarcasm.json','r') as f:
        data = json.load(f)
    x = []
    y= []
    for record in data:
        text = record['headline']
        # text = clean_text(text)
        x.append( text)
        y.append(int(record['is_sarcastic']))
    return x,y

VOCAB_SIZE = 1000
MAXLEN = 120
oov_tok = "<OOV>"

texts, labels = fetch_data()
tokenizer = Tokenizer(num_words= VOCAB_SIZE, oov_token= oov_tok)
tokenizer.fit_on_texts(texts)
x = tokenizer.texts_to_sequences(texts)
x = pad_sequences(x, maxlen= MAXLEN, padding='post', truncating='post')
edge = int(len(x)*0.84)
labels = np.array(labels)
x_train, x_test = x[:edge], x[edge:]
y_train, y_test = labels[:edge], labels[edge:]

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

model = models.Sequential([
    layers.Embedding(input_dim= VOCAB_SIZE, output_dim=16),
    layers.Bidirectional( layers.LSTM(units=32, recurrent_dropout= 0.2)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
print(model.summary())

model.compile( optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

my_callbacks = [
    callbacks.EarlyStopping(monitor='val_acc', min_delta=0.01, patience=3),
    StopOnThreshold(0.84, 0.83)
]

history = model.fit(x_train, y_train,\
    epochs= 5, batch_size=32, \
    callbacks= my_callbacks,\
    validation_data= (x_test, y_test)).history

plot_history(history)
print('train:', model.evaluate(x_train, y_train))
print('val:', model.evaluate(x_test, y_test))

model.save('sarcasm.h5')