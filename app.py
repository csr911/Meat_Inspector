from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)

# Load pre-trained model
MODEL_PATH = 'model.h5'  # Path to your saved model
model = load_model(MODEL_PATH)

# Set upload folder
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Function to check if file type is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess image for prediction
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(176, 176))  # Adjust to match the input shape of your model
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale the image
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess the image and make a prediction
        img = preprocess_image(filepath)
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)

        # Map predicted class to label
        labels = ['Fresh', 'Half Fresh', 'Spoiled']
        result = labels[predicted_class[0]]

        return render_template('index.html', uploaded_image=filepath, prediction_text=result)

    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)