#import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

#load mnist data set
(train_images,train_labels), (test_images, test_labels) = datasets.mnist.load_data()

#preprocessing: normalize the pixel values to be between 0 and 1
train_images=train_images/255.0
test_images=test_images/255.0

#reshape the images (28,28,1) as they are greyscale
train_images=train_images.reshape((train_images.shape[0], 28,28,1))
test_images=test_images.reshape((test_images.shape[0], 28,28,1))

# convert the labels into one hot encoded format
train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)

# build a cnn model
model=models.Sequential()

#first convolutional layer
model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))

#second convolutional layer
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

# Third Convolutional layer
model.add(layers.Conv2D(64,(3,3), activation='relu'))

# Flatten the 3D oytput to 1D and add a Dense layer
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

# Output layer with 10 neurons (for 10 digit classes)
model.add(layers.Dense(10, activation='softmax'))

#Compile the model 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                                                                          
# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))

#evaluate the model on test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc * 100:2f}%")

# make prediction on test images
predictions=model.predict(test_images)
print(f"Prediction for the first test image: {np.argmax(predictions[0])}")

plt.imshow(test_images[0].reshape(28,28), cmap='gray')
plt.title(f"Predicted Label: {predictions[0].argmax()}")
plt.show()