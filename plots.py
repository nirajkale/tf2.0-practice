import matplotlib.pyplot as plt

def plot_history(history):
    fig = plt.figure()
    plt.plot(history['loss'], c='r')
    plt.plot(history['val_loss'], c='b')
    plt.title('Loss History')
    fig = plt.figure()
    plt.show()
    plt.plot(history['acc'], c='r')
    plt.plot(history['val_acc'], c='b')
    plt.title('Acc History')
    plt.show()