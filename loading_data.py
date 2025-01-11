import pickle
import matplotlib.pyplot as plt

### Loading data:
pickle_in = open("X.pickle", "rb") # Open file (rb = read binary)
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb") # Open file (rb = read binary)
y = pickle.load(pickle_in)

image_num = 1000
print(X[image_num])
print(y[image_num])
plt.imshow(X[image_num])
plt.show()

print(len(X))
print(len(y))

