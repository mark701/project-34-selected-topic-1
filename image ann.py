import glob

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.io import imread
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from numpy import mean,absolute

train_images = []
train_labels=[]
test_images=[]
test_labels=[]

for i in glob.glob("flower_photos/sunflowers/*.jpg"):
    image_convert = imread(i,as_gray=True)
    image_convert = np.resize(image_convert, (320, 280))
    # image_convert = image_convert.reshape(image_convert.shape[1] * image_convert.shape[0]*image_convert.shape[2], ).T
    train_images.append(image_convert)
    train_labels.append(0)

for i in glob.glob("flower_photos/roses/*.jpg"):
    image_convert = imread(i,as_gray=True)
    image_convert = np.resize(image_convert, (320, 280))
    # image_convert = image_convert.reshape(image_convert.shape[1] * image_convert.shape[0]*image_convert.shape[2], ).T
    train_images.append(image_convert)
    train_labels.append(1)

for i in glob.glob("flower_photos/test/*.jpg"):
    image_convert = imread(i,as_gray=True)
    image_convert = np.resize(image_convert, (320, 280))
    # image_convert = image_convert.reshape(image_convert.shape[1] * image_convert.shape[0]*image_convert.shape[2], ).T
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
train_images = train_images / 255
test_images = test_images / 255


X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.4, random_state=True)

# Converting labels to one-hot encoded vectors
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)



model = tf.keras.Sequential([

    # Flatten Layer that converts images to 1D array
    tf.keras.layers.Flatten(),

    # Hidden Layer with 512 units and relu activation
    tf.keras.layers.Dense(units=80, activation='relu'),

    # Output Layer with 10 units for 10 classes and softmax activation
    tf.keras.layers.Dense(units=2, activation='softmax')])

model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

history = model.fit(x = train_images,y = train_labels,epochs = 60)



# Showing plot for loss
plt.plot(history.history['loss'])
plt.xlabel('epochs')
plt.legend(['loss'])
plt.show()

# Showing plot for accuracy
plt.plot(history.history['accuracy'], color='orange')
plt.xlabel('epochs')
plt.legend(['accuracy'])
plt.show()


predicted_probabilities = model.predict(test_images)
predicted_classes = tf.argmax(predicted_probabilities, axis=-1).numpy()



index = 4

# Showing image
plt.imshow(test_images[index])

# Printing Probabilities
print("Probabilities predicted for image at index", index)
print(predicted_probabilities[index])

print()

# Printing Predicted Class
print("Probabilities class for image at index", index)
print(predicted_classes[index])




predicted_probabilities02 = model.predict(X_test)
predicted_classes = tf.argmax(predicted_probabilities, axis=-1).numpy()

index = 4
# Showing image
plt.imshow(test_images[index])
# Printing Probabilities
print("Probabilities predicted for image at index", index)
print(predicted_probabilities[index])
# Printing Predicted Class
print("Probabilities class for image at index", index)
print(predicted_classes[index])