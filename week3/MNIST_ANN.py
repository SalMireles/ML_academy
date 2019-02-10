# -*- coding: utf-8 -*-

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from keras.utils import np_utils



"""### Initialize the necessary variables"""

batch_size = 128 # Revelance: After each batch back propogation happens - this is the beauty of neural nets
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28 # Image is input as 28x28

"""### Reading the image data from Keras"""

(x_train, y_train), (x_test, y_test) = mnist.load_data()

"""# 03 : Analyze Data

Prepare the Features and Target variables.
To analyze what is the shape of the feature set.
"""

plt.subplot(221)
plt.imshow(x_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(x_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(x_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(x_train[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()

print(x_train.shape)

"""Visualizing a random digit using the Matplotlib Library

# 04 : Feature Engineering

MNIST data is divided as follows:
- Train Data - First 60000 rows
- Test Data - Last 10000 rows
"""

# Reshaping the image to the size of the image i.e, 28 x 28
# converting into 1D
print(x_train.shape[1])
print(x_train.shape[2])

num_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')


#converting from 3D to 2D x_train.shape = (6000,28,28)

# Normalizing the pixel data

x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

num_classes

"""# 05 : Model Selection

### Create the Baseline Model

Create the baseline model using seqeuential and dense from Keras module.
"""

x_train.shape

from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

from keras.layers.core import Dense, Dropout, Activation

model = Sequential()
model.add(Dense(512, input_shape=(784,))) #It doesn't know the shape of the data comming in
model.add(Activation('relu'))

model.add(Dense(512))  # Fist hidden layer
model.add(Activation('relu'))

model.add(Dense(10)) # output 10 nodes #Second hidden layer
model.add(Activation('softmax')) # gives you the probability of each class (in this case 10)

#Different types of activation functions:
#threshold
#softmax
#relu
#sigmoid

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.summary()

#Params: previous layer connects to next layer
#First one is 513*784
#The 513 is due to bias weight (or previously known as coefficient)

x_train.shape

y_train.shape

# build the model
# Fit the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200, verbose=2)

# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=2)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

print('Test accuracy:', scores[1])

scores

"""<center><h1>The End</h1></center>"""
