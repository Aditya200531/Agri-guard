import os
from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import tensorflow as tf
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="Agri-guard\model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels in index order
class_labels = [
    "Corn (maize) Cercospora leaf spot Gray leaf spot",
    "Corn (maize) Common rust",
    "Corn (maize) healthy",
    "Corn (maize) Northern Leaf Blight",
    "Pepper, bell Bacterial spot",
    "Pepper, bell healthy",
    "Potato Early blight",
    "Potato healthy",
    "Potato Late blight",
    "Tomato Bacterial spot",
    "Tomato Early blight",
    "Tomato healthy",
    "Tomato Late blight",
    "Tomato Leaf Mold",
    "Tomato Septoria leaf spot",
    "Tomato Spider mites Two-spotted spider mite",
    "Tomato Target Spot",
    "Tomato Tomato mosaic virus",
    "Tomato Tomato Yellow Leaf Curl Virus"
]

def process_image(image_path):
    image = Image.open(image_path).resize((input_details[0]['shape'][1], input_details[0]['shape'][2]))
    input_data = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]  # Get first row from softmax
    return output_data

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
            model_output = process_image(file_path)
            predicted_index = np.argmax(model_output)  # Get index of highest softmax value
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
