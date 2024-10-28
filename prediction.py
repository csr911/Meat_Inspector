import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG
from random import randint

# Define constants
WORK_DIR = './enhanced/'
CLASSES = ['fresh', 'half_fresh', 'spoiled']
IMG_SIZE = 176
DIM = (IMG_SIZE, IMG_SIZE)

# Load the trained model
custom_densenet_model_dir = "model.h5"
model = tf.keras.models.load_model(custom_densenet_model_dir)

# Create an ImageDataGenerator for test data (without augmentation)
test_data_gen = IDG(preprocessing_function=tf.keras.applications.densenet.preprocess_input)

# Load the test data
test_data_gen = test_data_gen.flow_from_directory(directory=WORK_DIR, target_size=DIM, batch_size=1, class_mode='categorical', shuffle=False)

# Make predictions on multiple random samples from the test data
def predict_random_samples(generator, num_samples=10):
    for _ in range(num_samples):
        # Get a random index
        random_index = randint(0, generator.samples - 1)

        # Retrieve the image and label
        img, actual_label = generator[random_index]

        # Make a prediction
        predicted_label = model.predict(img)
        predicted_class = np.argmax(predicted_label)

        # Display the image, actual label, and predicted label
        plt.figure(figsize=(5, 5))
        plt.imshow(img[0])  # Show the first image in the batch
        plt.axis('off')
        plt.title(f"Actual: {CLASSES[np.argmax(actual_label[0])]} \nPredicted: {CLASSES[predicted_class]}")
        plt.show()

# Run the prediction function for 10 random samples
predict_random_samples(test_data_gen, num_samples=10)