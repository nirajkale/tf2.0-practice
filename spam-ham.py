import string
import re
import tensorflow as tf
from tensorflow.keras import callbacks, models, optimizers, layers
from plots import *
from mycallbacks import *
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def read_dataset():
    texts = []
    labels = []
    with open(r'data/SMSSpamCollection','r') as f:
        for line in f.readlines():
            label, text = line.split('\t', maxsplit=2)
            text = text.encode('ascii',errors='ignore').decode()
            text = ' '.join(re.findall('[a-zA-Z]+',text))
            text = re.sub('\s{2,}',' ', text)
            if label.strip()=='spam':
                labels.append(1)
            else:
                labels.append(0)
            texts.append(text)
    return texts, labels

VOCAB_SIZE = 5000
MAXLEN = 100
texts, labels = read_dataset()
tokenizer= Tokenizer(num_words=VOCAB_SIZE, lower=True)
tokenizer.fit_on_texts(texts)
x = tokenizer.texts_to_sequences(texts)
x = pad_sequences(x, maxlen= MAXLEN, padding='post', truncating='post')
edge = int(len(x)*0.84)
labels = np.array(labels)
x_train, x_test = x[:edge], x[edge:]
y_train, y_test = labels[:edge], labels[edge:]

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

model = models.Sequential([
    layers.Embedding(input_dim= VOCAB_SIZE, output_dim=16, input_shape=(MAXLEN,)),
    layers.Bidirectional(layers.LSTM(32, recurrent_dropout=0.2)),
    layers.Dense(units=32, activation='sigmoid'),
    layers.Dense(units=1, activation='sigmoid')
])

print(model.summary())
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

my_callbacks = [
    callbacks.EarlyStopping(monitor='val_acc', min_delta=0.01, patience=4),
    StopOnThreshold(0.95, 0.95)
]

history = model.fit(x_train, y_train,\
    epochs= 25, batch_size=32, \
    callbacks= my_callbacks,\
    validation_data= (x_test, y_test)).history

plot_history(history)
print('train:', model.evaluate(x_train, y_train))
print('val:', model.evaluate(x_test, y_test))