from flask import Flask, jsonify, request
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from PIL import Image
import requests
from io import BytesIO
import groq
from groq import Groq

client = Groq(api_key="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")

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('recycle_model.keras')  # Use your saved model file

# Class labels for classification
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']  # Replace with your actual class labels

# Function to preprocess and classify image
def preprocess_and_classify(img):
    try:
        # Resize and preprocess the image
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image

        # Make a prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)

        # Map the predicted class index to the class label
        return class_labels[predicted_class[0]]
    except Exception as e:
        return f"An error occurred: {e}"

# Function to classify image from URL
def classify_image_from_url(img_url):
    try:
        # Download the image from the URL
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return preprocess_and_classify(img)
    except Exception as e:
        return f"An error occurred: {e}"

# Function to classify image from uploaded file
def classify_image_from_file(file):
    try:
        img = Image.open(file).convert('RGB')
        return preprocess_and_classify(img)
    except Exception as e:
        return f"An error occurred: {e}"

# Define the Flask route for the classification API
@app.route("/classify", methods=["POST"])
def classify():
    # Check if image is provided in the form of URL or file
    if 'img_url' in request.json:
        img_url = request.json['img_url']
        try:
            result = classify_image_from_url(img_url)
            completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": """
You are a helpful answering assistant about sustainable Development goals, specially about creating something from trash.
Your task is to detect the trash and reply about how to recycle the trash by creating it into some useful or beautiful things. please respond to the user politely and concisely.
Answer in plain text (concisely, maximum 15 sentences) and explain to the user how to create something creative from the detected trash step by step. 
At the end, please give a random quote about Sustainable Development goals (no need "Quotes:").
Example:
You've detected metal trash. Here's a creative way to recycle it:

**Project: Metal Wind Chime**

Materials needed:
- Old metal cans (aluminum or tin)
- Copper wire
- Pliers
- Hammer
- Drill
- Metal ring
- Small objects to create sound (e.g., keys, washers)

Step-by-Step Instructions:

1. Collect and clean the metal cans.
2. Cut the cans into different shapes and sizes.
3. Use copper wire to connect the cans to a metal ring.
4. Use pliers to shape the wire into desired forms.
5. Hammer the wire to secure it.
6. Drill holes in the cans for hanging.
7. Attach small objects to create sound.
8. Hang the wind chime in a breezy area.

This project not only reduces waste but also creates a beautiful and melodious piece of art. (customizable)

"quotes" - name of the figure
"""
                }
                , {
                    "role": "user",
                    "content": "My item is "+ result
                    }
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,)
            
            return jsonify({"classification": result, "output":completion.choices[0].message.content})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "No image URL or file provided"}), 400

    """elif 'file' in request.files:
        file = request.files['file']
        try:
            result = classify_image_from_file(file)
            return jsonify({"classification": result})
        except Exception as e:
            return jsonify({"error": str(e)}), 500"""
    
# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
