import glob

import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from skimage.io import imread
from sklearn.model_selection import train_test_split

train_images = []
train_labels=[]
test_images=[]
test_labels=[]

for i in glob.glob("flower_photos/sunflowers/*.jpg"):
    image_convert = imread(i, as_gray=True)
    image_convert = np.resize(image_convert, (80, 80,2))
    # image_convert = image_convert.reshape(image_convert.shape[1] * image_convert.shape[0]*image_convert.shape[2], ).T
    train_images.append(image_convert)
    train_labels.append(0)


for i in glob.glob("flower_photos/roses/*.jpg"):
    image_convert = imread(i, as_gray=True)
    image_convert = np.resize(image_convert, (80, 80,2))
    #image_convert = image_convert.reshape(image_convert.shape[1] * image_convert.shape[0], ).T
    train_images.append(image_convert)
    train_labels.append(1)

for i in glob.glob("flower_photos/test/*.jpg"):
    image_convert = imread(i, as_gray=True)
    image_convert = np.resize(image_convert, (80, 80,2))
    #image_convert = image_convert.reshape(image_convert.shape[1] * image_convert.shape[0], ).T
    test_images.append(image_convert)
    test_labels.append(2)
train_images = np.asarray(train_images)
train_labels = np.asarray(train_labels)
test_images = np.asarray(test_images)
test_labels = np.asarray(test_labels)

print("train_images shape: ", train_images.shape)
print("train_labels shape: ", train_labels.shape)
print("test_images shape: ", test_images.shape)
print("test_labels shape: ", test_labels.shape)


# Converting image pixel values to 0 - 1
train_images = train_images.reshape(train_images.shape[3] * train_images.shape[1] * train_images.shape[2], train_images.shape[0]).T
train_labels = train_labels.reshape(train_labels.shape[0], )
X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels)
print(X_train)
print(y_train)
print(X_test)
print(y_test)


# Training the SVM model on the Training set
classifier = SVC(C=1.7,kernel='poly',gamma='auto')
history = classifier.fit(X_train, y_train)

prediction = classifier.predict(X_test)

accuracy = classifier.score(X_test, y_test)
print('Accuracy: ', accuracy)


