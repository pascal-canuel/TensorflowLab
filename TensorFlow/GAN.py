
# https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import sys
import cv2
import time
import glob
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop

'''
The generator show fake images to the discriminator. 
The discriminator says why it is fake.
The generator try to make better images until the discriminator is fool and thinks it is real images.
'''

# Dataset
# load dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# map class names with labels
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
              
train_images = train_images / 255.0
test_images = test_images / 255.0

# display sample 
# display the first 25 images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Discriminator network
depth = 64
dropout = 0.4
input_shape = (28, 28, 3)

discriminator = Sequential([
   Conv2D(depth*1, 5, strides=2, input_shape=input_shape, padding='same', activation=LeakyReLU(alpha=0.2)),
   Dropout(dropout),

   Conv2D(depth*2, 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)),
   Dropout(dropout),

   Conv2D(depth*4, 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)),
   Dropout(dropout),

   Conv2D(depth*8, 5, strides=1, padding='same', activation=LeakyReLU(alpha=0.2)),
   Dropout(dropout),

   Flatten(),
   Dense(1, activation = 'sigmoid'),
])

discriminator.summary()

# Generative network

'''
used 
'''
