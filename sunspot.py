import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, callbacks, optimizers, layers
import matplotlib.pyplot as plt

def read_csv():
    time_step = []
    sunspots = []
    with open('sunspots.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            sunspots.append(float(row[2]))
            time_step.append(row[1])
    return sunspots

def create_windowed_dataset(series, window_size, batch_size):
    # series = tf.convert_to_tensor(series, dtype=tf.float32) #[[1], [2], [3], ..]
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series) #[[1], [2], [3], ..]
    ds = ds.window(size= window_size+1, shift=1, drop_remainder=True) #[ [array[1], array[2], array[3], array[4]], ..]
    ds = ds.flat_map(lambda w: w.batch(window_size+1)) #[ [[1], [2], [3], [4]], [[5], [6], [7], [8]]]
    ds = ds.shuffle(1024)
    ds = ds.map(lambda w: (w[:-1], w[1:])) #create data, targets
    return ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

sunspots = np.array(read_csv())
split_time = 3000
window_size = 30
batch_size = 32

series_train = sunspots[:split_time]
series_val = sunspots[split_time:]

training_pipeline = create_windowed_dataset(series_train, window_size, batch_size)
val_pipeline = create_windowed_dataset(series_val, window_size, batch_size)

model = models.Sequential([
    #32, 30, 1
    layers.Bidirectional(layers.LSTM(units=32, recurrent_dropout=0.25, return_sequences=True), input_shape=(window_size,1)),
    # layers.LSTM(units=32, recurrent_dropout=0.25, return_sequences=True, input_shape=(window_size,1)),
    layers.Bidirectional(layers.LSTM(units=16, recurrent_dropout=0.25, return_sequences=True)),
    layers.Dense(10, activation='relu'),
    layers.Dense(1, activation='relu'),
    layers.Lambda(lambda x: x * 400)
])

print(model.summary())
model.compile(optimizer='adam', loss='mae', metrics=['mae'])
my_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_mae', min_delta=0.5, patience=10)
]
history = model.fit(training_pipeline,\
                    epochs=60, batch_size= batch_size,\
                    callbacks= my_callbacks,\
                    validation_data= val_pipeline).history

def plot_history(history):
    fig = plt.figure()
    plt.plot(history['mae'], c='r')
    plt.plot(history['val_mae'], c='b')
    plt.title('Loss History')
    plt.show()

plot_history(history)

print('validation:')
print(model.evaluate(val_pipeline))

def model_forecast(model, series, window_size):
   ds = tf.data.Dataset.from_tensor_slices(series)
   ds = ds.window(window_size, shift=1, drop_remainder=True)
   ds = ds.flat_map(lambda w: w.batch(window_size))
   ds = ds.batch(32).prefetch(1)
   forecast = model.predict(ds)
   return forecast


window_size = window_size
rnn_forecast = model_forecast(model, sunspots[..., np.newaxis], window_size)
rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]

result = tf.keras.metrics.mean_absolute_error(series_val, rnn_forecast).numpy()

# WE EXPECT AN MAE OF 15 or less for the maximum score
score = 20 - result
if score > 5:
   score = 5
print('-'*30)
print(result)
print(score)