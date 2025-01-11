### CNN:
# Basic structure:
# Convolution -> Pooling -> Convolution -> Pooling -> Fully Connected Layer-> Output

# Imports
import tensorflow as tf 
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callBacks import TensorBoard
import time

# Name model
NAME = "Cats-vs-Dog-cnn-64x2-{}".format(int(time.time()))

# Define Tensorboard
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

# Data
import pickle
pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

# Data Preprocessing
# X = X/255.0 # Normalizing
X = tf.keras.utils.normalize(X, axis=1) # Normalize using Keras?


### Build Model
model = Sequential()

# Conv + pool 1
model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Conv + pool 2
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Other layers
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

# Compile/finalize 
model.compile(loss='binary_crossentropy',
			  optimizer='adam',
			  metrics=['accuracy'])

# Pass callbacks into fit
model.fit(X, y, batch_size=32, epochs=10, validation_split=0.3, callbacks=[tensorboard])

## Save model
model.save('cats_dogs.model')

