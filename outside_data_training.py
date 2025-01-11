import numpy as np 
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random

DATADIR = "PetImages"
Categories = ["Dog", "Cat"]
IMG_SIZE = 50

training_data = []

def create_training_data():
	for category in Categories:

		path = os.path.join(DATADIR, category) # create path to dogs and cats in proper format
		class_num = Categories.index(category)

		for img in tqdm(os.listdir(path)):
			try:
				img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) # convert to array
				new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) # resize array
				training_data.append([new_array, class_num]) # add this to training data
			except Exception as e:
				pass

create_training_data()
print(len(training_data))

### Shuffle Data
random.shuffle(training_data)
# check shuffle
for sample in training_data[:10]:
	print(sample[1])

### Make dataset and save it
X = []
y = []

for features, label in training_data:
	X.append(features)
	y.append(label)

print(len(X))
print(len(y))

#print(X)
X = np.array(X)
print(X.shape)
X = X.reshape(25000, IMG_SIZE, IMG_SIZE, 1)

plt.imshow(X[0])
y = np.array(y)
X = X.reshape(25000, IMG_SIZE, IMG_SIZE, 1)

print(len(X))
print(len(y))

# Save it
import pickle
pickle_out = open("X.pickle", "wb") # Open new file
pickle.dump(X, pickle_out) # Save to file
pickle_out.close() # close file

pickle_out = open("y.pickle", "wb") # Open new file
pickle.dump(y, pickle_out) # Save to file
pickle_out.close() # close file

### Loading data:
pickle_in = open("X.pickle", "rb") # Open file (rb = read binary)
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb") # Open file (rb = read binary)
y = pickle.load(pickle_in)

pickle_in.close()