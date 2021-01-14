from keras.applications import VGG16, ResNet50, DenseNet121
from keras import layers
from capsulelayers import squash
import numpy as np
import tensorflow as tf  

def VGG16_block5(inputs, dim_capsule):    
    conv_base = VGG16(weights='imagenet', include_top=False)
    output = conv_base(inputs)
    conv_base.trainable = True
    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == 'block1_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    #output = layers.Flatten()(output)
    #output = layers.Dense(2048)(output)
    outputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)

def ResNet50_base(inputs, dim_capsule):   
    conv_base = ResNet50(weights='imagenet', include_top=False)
    output = conv_base(inputs)
    conv_base.trainable = True
    #set_trainable = False
    #for layer in conv_base.layers:
    #    if layer.name == 'add_15':
    #        set_trainable = True
    #    if set_trainable:
    #        layer.trainable = True
    #    else:
    #        layer.trainable = False
    #output = layers.Flatten()(output)
    #output = layers.Dense(2048)(output)
    outputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)

def DenseNet121_base(inputs, dim_capsule):
    conv_base = DenseNet121(weights='imagenet', include_top=False)
    output = conv_base(inputs)

    tf.random_shuffle(output)
    
    # output = layers.Reshape(target_shape=[1024])(output)
    # output = output[shuffle_list]

    # output = tf.transpose(output)
    # output = tf.random_shuffle(output, seed=1)
    # output = tf.transpose(output)

    outputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)