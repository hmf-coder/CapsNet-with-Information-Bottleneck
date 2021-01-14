from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.utils import to_categorical

def occluded_cifar():
    # the data, shuffled and split between train and test sets
    mnist = keras.datasets.mnist
    cifar10 = keras.datasets.cifar10
    (mx_train, my_train), (mx_test, my_test) = mnist.load_data()
    (cx_train, cy_train), (cx_test, cy_train) = cifar10.load_data()

    row = 22  #the size of resize mnist
    n_sample = cx_train.shape[0]  
    mx_len = cx_train.shape[1]
    mx_wid = cx_train.shape[2]

    mx_train = mx_train[:50000].astype('float32') / 255.
    mx_test = mx_test[:50000].astype('float32') / 255.
    cx_train = cx_train.astype('float32') / 255.
    cx_test = cx_test.astype('float32') / 255.
    
    mx_rescale = np.zeros((n_sample, mx_len, mx_wid))  
    x = np.zeros((n_sample, row, row))
    for i in range(len(mx_train)):
        x[i] = cv2.resize(mx_train[i], (row, row))   #squash mnist
        mx_rescale[i] = cv2.copyMakeBorder(x[i], mx_len-row, 0, mx_wid-row, 0, cv2.BORDER_CONSTANT)    #extent mnist with constant 0(black)
    mx_resize = np.stack((mx_rescale, mx_rescale, mx_rescale), axis=3)      #from gray to RGB
    
    occluded_cifar = mx_resize + cx_train
    occluded_cifar[occluded_cifar>1.] = 1.    #insure all values are small than 1.
    
    return occluded_cifar, my_train, cy_train  
    #occluded_cifar is occluded samples, my_train is the main task labels(cifar10), cy_train is the nuisance task labels(MNIST).

def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)

def load_smallmnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    # print(x_train.shape)
    # print(y_train.shape)

    # ratio = 0.8
    # from sklearn.model_selection import train_test_split
    # x_train, x_, y_train, y_ = train_test_split(x_train, y_train, stratify=y_train, train_size=ratio)

    return (x_train, y_train), (x_test, y_test)

def load_fashionmnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)

def load_cifar10():
    from keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.reshape(-1, 32, 32, 3).astype('float32') / 255.
    x_test = x_test.reshape(-1, 32, 32, 3).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))

    # ratio = 0.6
    # from sklearn.model_selection import train_test_split
    # x_train, x_, y_train, y_ = train_test_split(x_train, y_train, stratify=y_train, train_size=ratio)

    return (x_train, y_train), (x_test, y_test)

def load_cifar100():
    from keras.datasets import cifar100
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    x_train = x_train.reshape(-1, 32, 32, 3).astype('float32') / 255.
    x_test = x_test.reshape(-1, 32, 32, 3).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)
