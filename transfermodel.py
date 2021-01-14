from keras.applications import DenseNet121
from keras import layers
from keras.models import Model
import tensorflow as tf 
from keras import models
from keras.layers import Dense, Reshape, BatchNormalization, Deconvolution2D

from keras.utils.vis_utils import plot_model
# from keras.datasets import cifar10
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()

conv_base = DenseNet121(weights=None, include_top=False, input_shape=(32,32,3))

plot_model(conv_base, to_file='model1.png', show_shapes=True)

decoder = models.Sequential(name='decoder')
decoder.add(layers.Dense(8*8*256, use_bias=False, input_dim=16*n_class))
decoder.add(layers.BatchNormalization())
decoder.add(layers.LeakyReLU())
 decoder.add(layers.Reshape((8, 8, 256)))
decoder.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
decoder.add(layers.BatchNormalization())
decoder.add(layers.LeakyReLU())
decoder.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
decoder.add(layers.BatchNormalization())
decoder.add(layers.LeakyReLU())
decoder.add(layers.Conv2DTranspose(32, (7, 7), strides=(2, 2), padding='same', use_bias=False))
decoder.add(layers.BatchNormalization())
decoder.add(layers.LeakyReLU())
decoder.add(layers.Conv2DTranspose(3, (7, 7), strides=(1, 1), padding='same', use_bias=False, name='out_recon'))

decoder.summary()
