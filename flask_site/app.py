import os
import json
from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import tensorflow as tf
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model from the .pb file
model = tf.saved_model.load("Agri-guard\saved_model")

# Load class labels from a JSON file
with open("Agri-guard\class_indices.json", "r") as f:
    class_labels = json.load(f)

# Convert JSON keys to integers for easier indexing
class_labels = {int(k): v for k, v in class_labels.items()}

def process_image(image_path):
    image = Image.open(image_path).resize((224, 224))  # Adjust size based on your model's requirements
    input_data = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)
    input_tensor = tf.convert_to_tensor(input_data)

    # Run inference using the loaded model
    infer = model.signatures['serving_default']
    output_data = infer(input_tensor)
    logits = list(output_data.values())[0].numpy()[0]  # Get the first (and only) batch

    return logits

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            session['uploaded_image'] = file.filename
            session['chat_history'] = []  # Initialize chat history for the session

            # Process the image and get the class label
            logits = process_image(file_path)
            predicted_index = np.argmax(logits)  # Get index of highest softmax value
            predicted_label = class_labels[predicted_index]  # Map to class label
            session['model_output'] = predicted_label  # Store class label in session

            return redirect(url_for('results'))
    return render_template('index.html')

@app.route('/results', methods=['GET', 'POST'])
def results():
    uploaded_image = session.get('uploaded_image', None)
    model_output = session.get('model_output', None)
    chat_history = session.get('chat_history', [])

    if request.method == 'POST':
        user_input = request.form.get('user_input')
        if user_input:
            response = f"Bot response for: '{user_input}'"
            chat_history.append(f"User: {user_input}")
            chat_history.append(response)
            session['chat_history'] = chat_history  # Update chat history in session

    return render_template('results.html', 
                           uploaded_image=uploaded_image, 
                           model_output=model_output, 
                           chat_history=chat_history)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
