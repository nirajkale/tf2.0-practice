import pandas as pd
from tensorflow.keras import models, callbacks, layers, optimizers, regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import numpy as np

#settings
VOCAB_SIZE = 5000
MAXLEN = 50
EMBEDDING_DIM = 12

def shuffle_arrays_in_unison(arrays):
    fixed_length = arrays[0].shape[0]
    for arr in arrays[1:]:
        if arr.shape[0] != fixed_length:
            raise Exception('All the arrays need to have same length')
    shuffled_indices = np.random.permutation(fixed_length)
    for i in range(len(arrays)):
        arrays[i] = arrays[i][shuffled_indices]
    return arrays

def read_data():
    fname = r'C:\drive\datasets\text classification\sentiments.xlsx'
    df = pd.read_excel(fname, sheet_name='Sheet1')
    data = df['data'].tolist()
    labels = [1 if 'pos' in lbl else 0 for lbl in df['labels'].tolist()]
    return data, labels

texts, labels = read_data()
tokenizer = Tokenizer(num_words= VOCAB_SIZE)
tokenizer.fit_on_texts(texts)
x = tokenizer.texts_to_sequences(texts)
x = pad_sequences(x, maxlen= MAXLEN, padding='post', truncating='post')
labels = np.array(labels)

labels, x = shuffle_arrays_in_unison([labels, x])
edge = int(len(x)*0.8)
x_train, x_val = x[:edge], x[edge:]
y_train, y_val = labels[:edge], labels[edge:]

model = models.Sequential([
    layers.Embedding(input_dim= VOCAB_SIZE, output_dim= EMBEDDING_DIM, input_length= MAXLEN),
    layers.Bidirectional(layers.LSTM(32, recurrent_dropout=0.2)),
    # layers.GlobalAveragePooling1D(),
    # layers.Dense(8, activation='relu', kernel_regularizer= regularizers.l2(l=0.005)),
    layers.Dense(1, activation='sigmoid')
])

print(model.summary())
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
my_callbacks = [
    callbacks.EarlyStopping(monitor='val_acc', min_delta=0.01, patience=3)
]

history = model.fit(x_train, y_train,\
                    epochs= 5, batch_size=32,\
                    callbacks= my_callbacks,\
                    validation_data=(x_val, y_val)).history

print('training acc:', model.evaluate(x_train, y_train))
print('val acc:', model.evaluate(x_val, y_val))