from tensorflow.keras import callbacks

early_stop = callbacks.EarlyStopping(monitor='val_acc', min_delta=0.01, patience=7)

class StopOnThreshold(callbacks.Callback):

    def __init__(self, training_acc_threshold, val_acc_threshold):
        self.training_acc_threshold= training_acc_threshold
        self.val_acc_threshold = val_acc_threshold

    def on_epoch_end(self, epoch, logs={}):
        if logs['val_acc'] >= self.val_acc_threshold and logs['acc'] >= self.training_acc_threshold:
            print('\nreached training & validation acc threshold')
            self.model.stop_training = True