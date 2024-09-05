<<<<<<< HEAD
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from PIL import Image
import requests
from io import BytesIO

# Load the trained model
model = tf.keras.models.load_model('recycle_model.keras')  # Use your saved model file

class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']  # Replace with your actual class labels

# Function to preprocess and classify an image
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)  # Ensure the array is of type float32
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

# Function to classify the image
def classify_image(img_path_or_url):
    if img_path_or_url.startswith('http'):
        # If it's a URL, download the image with proper headers
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(img_path_or_url, headers=headers)
            response.raise_for_status()  # Check for request errors
            if 'image' not in response.headers['Content-Type']:
                raise ValueError("URL does not point to an image")
            img = Image.open(BytesIO(response.content)).convert('RGB')  # Ensure image is in RGB mode
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve image from URL: {e}")
    else:
        # If it's a local file path
        try:
            img = Image.open(img_path_or_url).convert('RGB')  # Ensure image is in RGB mode
        except Exception as e:
            raise RuntimeError(f"Failed to open image from path: {e}")
    
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    
    return class_labels[predicted_class[0]]

# Main program to get user input and classify the image
if __name__ == "__main__":
    img_path_or_url = input("Please enter the path of the image you want to classify (or a URL): ").strip()
    
    if not img_path_or_url:
        print("No image source provided.")
        exit(1)
    
    try:
        result = classify_image(img_path_or_url)
        print(f"The image is classified as: {result}")
    except Exception as e:
        print(f"An error occurred: {e}")
=======
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from PIL import Image
import requests
from io import BytesIO

# Load the trained model
model = tf.keras.models.load_model('recycle_model.keras')  # Use your saved model file

class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']  # Replace with your actual class labels

# Function to preprocess and classify an image
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)  # Ensure the array is of type float32
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

# Function to classify the image
def classify_image(img_path_or_url):
    if img_path_or_url.startswith('http'):
        # If it's a URL, download the image with proper headers
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(img_path_or_url, headers=headers)
            response.raise_for_status()  # Check for request errors
            if 'image' not in response.headers['Content-Type']:
                raise ValueError("URL does not point to an image")
            img = Image.open(BytesIO(response.content)).convert('RGB')  # Ensure image is in RGB mode
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve image from URL: {e}")
    else:
        # If it's a local file path
        try:
            img = Image.open(img_path_or_url).convert('RGB')  # Ensure image is in RGB mode
        except Exception as e:
            raise RuntimeError(f"Failed to open image from path: {e}")
    
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    
    return class_labels[predicted_class[0]]

# Main program to get user input and classify the image
if __name__ == "__main__":
    img_path_or_url = input("Please enter the path of the image you want to classify (or a URL): ").strip()
    
    if not img_path_or_url:
        print("No image source provided.")
        exit(1)
    
    try:
        result = classify_image(img_path_or_url)
        print(f"The image is classified as: {result}")
    except Exception as e:
        print(f"An error occurred: {e}")
>>>>>>> 6743dd3 (first commit)
