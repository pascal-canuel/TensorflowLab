
# https://www.tensorflow.org/tutorials/keras/basic_classification

import tensorflow as tf
from tensorflow import keras
#from keras.models import *
#from keras.layers import *
import numpy as np
import os
import sys
import cv2
import time
import matplotlib.pyplot as plt

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# load dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# map class names with labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# show dataset format
print(train_images.shape)
print(len(train_labels))

print(test_images.shape)
print(len(test_labels))

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

# preprocess the data by scaling values to a range of 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# building the layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), # linearize the data 
    keras.layers.Dense(128, activation=tf.nn.relu), # implement operation: outputs = activation(inputs * kernel + bias) / relu: Computes rectified linear: max(features, 0)
    keras.layers.Dense(10, activation=tf.nn.softmax) # softmax: Computes softmax activations
    ]) # returns an array of 10 probability scores that sum to 1

# compile the model
model.compile(optimizer=tf.train.AdamOptimizer(), # ** measures how accurate the model is during training **
              loss='sparse_categorical_crossentropy', # ** how the model is updated based on the data it sees and its loss function **
              metrics=['accuracy']) # monitor the training and testing steps

# train the model
model.fit(train_images, train_labels, epochs=5)

# evaluation with the data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

predictions = model.predict(test_images)
print('Prediction:', np.argmax(predictions[0]))
print('Real label:', test_labels[0])

# test the model

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

# Grab an image from the test dataset and add it too a list
img = (np.expand_dims(test_images[0],0))
predictions_single = model.predict(img)
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()



