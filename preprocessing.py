import os
import cv2
import numpy as np


# -------------------- IMAGE ENHANCEMENT FUNCTIONS -------------------- #

# Function to enhance color features (CLAHE)
def enhance_color_features(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    L = clahe.apply(L)
    lab = cv2.merge((L, A, B))
    enhanced_image = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    return enhanced_image


# Function to apply Gaussian Blur
def apply_gaussian_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


# Function to apply Histogram Equalization
def apply_histogram_equalization(image):
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


# Complete enhancement pipeline
def enhance_image(image_path):
    image = cv2.imread(image_path)

    # Apply the enhancement techniques sequentially
    clahe_image = enhance_color_features(image)
    gb_image = apply_gaussian_blur(clahe_image)
    he_image = apply_histogram_equalization(gb_image)

    return clahe_image, gb_image, he_image


# Function to create directory if not exists
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Function to save enhanced images in subfolders
def save_enhanced_images(clahe_img, gb_img, he_img, base_save_path, image_name):
    # Paths for different enhancement techniques
    clahe_save_path = os.path.join(base_save_path, 'clahe')
    gb_save_path = os.path.join(base_save_path, 'gaussian_blur')
    he_save_path = os.path.join(base_save_path, 'histogram_equalization')

    # Create directories if they don't exist
    create_directory(clahe_save_path)
    create_directory(gb_save_path)
    create_directory(he_save_path)

    # Save the images
    cv2.imwrite(os.path.join(clahe_save_path, image_name), clahe_img)
    cv2.imwrite(os.path.join(gb_save_path, image_name), gb_img)
    cv2.imwrite(os.path.join(he_save_path, image_name), he_img)


# Function to process images in a folder for each class
def process_images_in_folder(class_name, folder_path, enhanced_folder_path):
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        if os.path.isfile(img_path):
            clahe_img, gb_img, he_img = enhance_image(img_path)
            save_enhanced_images(clahe_img, gb_img, he_img, os.path.join(enhanced_folder_path, class_name), img_file)


# Main function to process images from all classes and save enhanced versions
def process_all_images():
    data_dir = 'data'  # Folder containing fresh, half_fresh, spoiled
    enhanced_dir = 'enhanced'  # Folder to save enhanced images

    # Class folders
    classes = ['fresh', 'half_fresh', 'spoiled']

    for class_name in classes:
        class_folder = os.path.join(data_dir, class_name)
        enhanced_class_folder = os.path.join(enhanced_dir, class_name)

        process_images_in_folder(class_name, class_folder, enhanced_class_folder)


# Run the processing
process_all_images()