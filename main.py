from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
import cv2 as cv
import numpy as np
from keras.models import load_model
from PIL import Image

# --- Basic Flask App Setup ---
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing for your frontend

# --- Configuration ---
# Define the folder to store user-uploaded images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# **FIX 1: Ensure the upload folder exists when the app starts**
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# --- Model and Label Loading ---
# Define the class labels your model can predict
# This must match the order the model was trained on
label_name = ['Apple scab','Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 'Cherry Powdery mildew',
'Cherry healthy','Corn Cercospora leaf spot Gray leaf spot', 'Corn Common rust', 'Corn Northern Leaf Blight','Corn healthy', 
'Grape Black rot', 'Grape Esca', 'Grape Leaf blight', 'Grape healthy','Peach Bacterial spot','Peach healthy', 'Pepper bell Bacterial spot', 
'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight', 'Potato healthy', 'Strawberry Leaf scorch', 'Strawberry healthy',
'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot',
'Tomato Spider mites', 'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy']

# **CRITICAL: Load the model. This will fail if the file is not found.**
# Make sure 'model/Leaf Deases(96,88).h5' is correctly placed in your project directory.
try:
    model = load_model('model/Leaf Deases(96,88).h5')
except FileNotFoundError:
    print("ERROR: Model file not found. Make sure 'model/Leaf Deases(96,88).h5' exists.")
    model = None # Set model to None so the app can still start to show the error

# --- API Routes ---

@app.route('/')
def root():
    """A simple root URL to check if the API is running."""
    return "ML API is live and running."

@app.route('/predict', methods=['POST'])
def predict():
    """The main prediction endpoint."""
    if model is None:
        return jsonify({'error': 'Model is not loaded. Check server logs.'}), 500

    # Get the file from the POST request
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    try:
        # Create a unique filename and save the uploaded file
        filename = f"{uuid.uuid4().hex}.jpg"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # --- Image Preprocessing ---
        # Load the saved image using PIL
        img = Image.open(file_path).convert('RGB')
        img_np = np.array(img)
        
        # **FIX 2: Resize the image to the exact dimensions the model was trained on.**
        # The model name suggests it expects 96x96 or 88x88 images. Using (96, 96) as an example.
        # Using the wrong size will lead to incorrect predictions.
        resized_img = cv.resize(img_np, (96, 96))
        
        # Add a batch dimension and normalize if necessary (assuming model expects pixels 0-1)
        normalized_img = np.expand_dims(resized_img, axis=0) / 255.0

        # --- Prediction ---
        predictions = model.predict(normalized_img)
        confidence = float(np.max(predictions)) * 100
        label_index = np.argmax(predictions)
        label = label_name[label_index]

        # You can add logic for low confidence if needed
        # if confidence < 80:
        #     return jsonify({'error': 'Low confidence score. Please try another image.'}), 400

        # **FIX 3: Create a dynamic image URL that works in production.**
        # This uses the request's host URL instead of hardcoding 'localhost'.
        image_url = f"{request.host_url}{app.config['UPLOAD_FOLDER']}/{filename}"

        # --- Construct the JSON Response ---
        response = {
            'predicted_disease': label,
            'confidence_score': round(confidence, 2), # Round to 2 decimal places
            'is_healthy': 'healthy' in label.lower(),
            'risk_level': 'Low' if confidence < 85 else 'Medium' if confidence < 95 else 'High',
            'recommendations': [
                "Apply fungicide for rot or blight." if "rot" in label.lower() or "blight" in label.lower() else "Monitor plants regularly.",
                "Ensure proper spacing for air circulation.",
                "Avoid watering the leaves directly; water at the base."
            ],
            'image_url': image_url
        }
        return jsonify(response)

    except Exception as e:
        # Return a generic error message and log the specific error on the server
        print(f"An error occurred during prediction: {e}")
        return jsonify({'error': 'An internal error occurred. Please check server logs.'}), 500

# --- Route to Serve Uploaded Images ---
@app.route('/static/uploads/<filename>')
def serve_image(filename):
    """This route allows the frontend to access the uploaded images."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- Start the App ---
if __name__ == '__main__':
    # Use port 5050 and enable debug mode for local development
    app.run(debug=True, port=5050)
