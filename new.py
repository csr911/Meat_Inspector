import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import os
from distutils.dir_util import copy_tree, remove_tree

from random import randint

from sklearn.model_selection import train_test_split

from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input  # DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG
from tensorflow.keras.layers import SeparableConv2D, BatchNormalization, GlobalAveragePooling2D

print("TensorFlow Version:", tf.__version__)


WORK_DIR = './enhanced/'

CLASSES = ['fresh',
           'half_fresh',
           'spoiled',
           ]

IMG_SIZE = 176
IMAGE_SIZE = [176, 176]
DIM = (IMG_SIZE, IMG_SIZE)

# Performing Image Augmentation to have more data samples


ZOOM = [.99, 1.01]
BRIGHT_RANGE = [0.8, 1.2]
HORZ_FLIP = True
FILL_MODE = "constant"
DATA_FORMAT = "channels_last"

work_dr = IDG(preprocessing_function=preprocess_input, brightness_range=BRIGHT_RANGE, zoom_range=ZOOM,
              data_format=DATA_FORMAT, fill_mode=FILL_MODE, horizontal_flip=HORZ_FLIP)

train_data_gen = work_dr.flow_from_directory(directory=WORK_DIR, target_size=DIM, batch_size=6500, shuffle=False)


def show_images(generator, y_pred=None):
    """
    Input: An image generator,predicted labels (optional)
    Output: Displays a grid of 9 images with lables
    """

    # get image labels
    labels = dict(zip([0, 1, 2, 3], CLASSES))

    # get a batch of images
    x, y = generator.next()

    # display a grid of 9 images
    plt.figure(figsize=(10, 10))
    if y_pred is None:
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            idx = randint(0, 6400)
            plt.imshow(x[idx])
            plt.axis("off")
            plt.title("Class:{}".format(labels[np.argmax(y[idx])]))

    else:
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(x[i])
            plt.axis("off")
            plt.title("Actual:{} \nPredicted:{}".format(labels[np.argmax(y[i])], labels[y_pred[i]]))


# Display Train Images
#show_images(train_data_gen)

# Retrieving the data from the ImageDataGenerator iterator

train_data, train_labels = train_data_gen.next()
# Getting to know the dimensions of our dataset


print(train_data.shape, train_labels.shape)

# Splitting the data into train, test, and validation sets


train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size=0.2,
                                                                    random_state=42)
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2,
                                                                  random_state=42)

densenet_model = DenseNet121(input_shape=(176, 176, 3), include_top=False, weights="imagenet")

for layer in densenet_model.layers:
    layer.trainable = False
custom_densenet_model = Sequential([
    densenet_model,
    Dropout(0.5),
    Flatten(),
    BatchNormalization(),
    Dense(521, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(521, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(3, activation='softmax')
], name="densenet_cnn_model")


# Defining a custom callback function to stop training our model when accuracy goes above 99%

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('acc') > 0.99:
            print("\nReached accuracy threshold! Terminating training.")
            self.model.stop_training = True


my_callback = MyCallback()

# ReduceLROnPlateau to stabilize the training process of the model
rop_callback = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5, verbose=1, min_lr=1e-3)
METRICS = [tf.keras.metrics.CategoricalAccuracy(name='acc'),
           tf.keras.metrics.AUC(name='auc'),
           ]

CALLBACKS = [my_callback, rop_callback]

custom_densenet_model.compile(optimizer='rmsprop',
                              loss=tf.losses.CategoricalCrossentropy(),
                              metrics=METRICS)

custom_densenet_model.summary()

# Fit the training data to the model and validate it using the validation data
EPOCHS = 5

history = custom_densenet_model.fit(train_data, train_labels, validation_data=(val_data, val_labels),
                                    callbacks=CALLBACKS, epochs=EPOCHS)

fig, ax = plt.subplots(1, 3, figsize=(30, 5))
ax = ax.ravel()
# Accuracy curve
ax[0].plot(history.history['acc'])
ax[0].plot(history.history['val_acc'])
ax[0].set_title('Model Accuracy')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy')
ax[0].legend(['Train', 'Validation'])

# Loss curve
ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])
ax[1].set_title('Model Loss')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss')
ax[1].legend(['Train', 'Validation'])

# AUC curve
ax[2].plot(history.history['auc'])
ax[2].plot(history.history['val_auc'])
ax[2].set_title('Model AUC')
ax[2].set_xlabel('Epochs')
ax[2].set_ylabel('AUC')
ax[2].legend(['Train', 'Validation'])

plt.tight_layout()
plt.show()

test_scores = custom_densenet_model.evaluate(test_data, test_labels)

print("Testing Accuracy: %.2f%%" % (test_scores[1] * 100))

# Predicting the test data

pred_labels = custom_densenet_model.predict(test_data)
# Saving the model for future use

custom_densenet_model_dir =  "model.h5"
custom_densenet_model.save(custom_densenet_model_dir)
