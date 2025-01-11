### CNN:
# Basic structure:
# Convolution -> Pooling -> Convolution -> Pooling -> Fully Connected Layer-> Output

# Imports
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# Data
import pickle
pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

IMG_SIZE = 50

# Data Preprocessing
# X = X/255.0 # Normalizing
X = tf.keras.utils.normalize(X, axis=1) # Normalize using Keras?

cat_dog_labels = {
	0: "Cat",
	1: "Dog"
}

# loading model

model = tf.keras.models.load_model('cats_dogs.model')

image_num = 2

out_of_sample_probabilty = model.predict(X[image_num].reshape(-1, IMG_SIZE, IMG_SIZE, 1))
out_of_sample_prediction = np.argmax(out_of_sample_probabilty)

plt.imshow(X[image_num])
plt.show()

print(f"NN's Answer: {cat_dog_labels[out_of_sample_prediction]}")
print(f'Correct Answer: {cat_dog_labels[y[image_num]]}')

num_of_cats = 0
num_of_dogs = 0

for Y in y:
	if Y == 0:
		num_of_cats += 1
	else:
		num_of_dogs += 1

print(f'Number of Cats: {num_of_cats}')
print(f'Number of Dogs: {num_of_dogs}')

run_all = False

if run_all == True:
	for x in X:
		out_of_sample_probabilty = model.predict(x.reshape(-1, IMG_SIZE, IMG_SIZE, 1))
		out_of_sample_prediction = np.argmax(out_of_sample_probabilty)

		print(cat_dog_labels[out_of_sample_prediction])

