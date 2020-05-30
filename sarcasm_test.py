
import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
from tensorflow.keras import models
from os import path
import numpy as np
import random

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

# DO NOT CHANGE THIS CODE OR THE TESTS MAY NOT WORK
vocab_size = 1000
max_length = 120
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000
sentences = []
labels = []
x,y = fetch_data()
tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)
tokenizer.fit_on_texts(x)
x_seq = tokenizer.texts_to_sequences(x)
x_seq = pad_sequences(x_seq, maxlen= max_length, padding='post', truncating='post')

random_sel = random.sample(range(200, len(x_seq)), 2000)
x_test = x_seq[random_sel]
y_test = np.array(y)[random_sel]

model = models.load_model('sarcasm.h5')
print('testing model..')
print(model.evaluate(x_test, y_test))
