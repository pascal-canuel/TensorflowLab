
# https://www.tensorflow.org/tutorials/keras/basic_text_classification

from __future__ import absolute_import, division, print_function
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

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# download dataset
# each word (integer) correspond to a specific word into a dictionary
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print(train_data[0])

# nb of words for a given review
len(train_data[0]), len(train_data[1])

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# decode a review 
print(decode_review(train_data[0]))

# convert data to the same length
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

# length changed
print(len(train_data[0]), len(train_data[1]))
print(train_data[0])

# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

# hidden unit
'''
***
If a model has more hidden units (a higher-dimensional representation space), and/or more layers, 
then the network can learn more complex representations. However, it makes the network more computationally 
expensive and may lead to learning unwanted patterns—patterns that improve performance on training data but not on the test data. 
This is called overfitting, and we'll explore it later.
***
'''

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy', # its a binary classification measuring the distance between the ground-truth and the predictions
              metrics=['accuracy'])

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

results = model.evaluate(test_data, test_labels)
print(results)

history_dict = history.history
history_dict.keys()
dict_keys(['val_acc', 'acc', 'val_loss', 'loss'])

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

'''
***
Notice the training loss decreases with each epoch and the training accuracy increases with each epoch. 
This is expected when using a gradient descent optimization—it should minimize the desired quantity on every iteration.
This isn't the case for the validation loss and accuracy—they seem to peak after about twenty epochs. 
This is an example of overfitting: the model performs better on the training data than it does on data it has never seen before. 
After this point, the model over-optimizes and learns representations specific to the training data that do not generalize to test data.
***
'''

# overfitting