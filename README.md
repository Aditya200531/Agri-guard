# Agri-guard

Agri-guard is a machine-learning-based application designed to assist with agricultural tasks. It utilizes a TensorFlow Lite model to identify or analyze agricultural inputs, which could include crop health assessment or similar functions. This project integrates a web interface for easy interaction.

## Features

- **Machine Learning Inference**: Uses TensorFlow Lite models to provide predictions.
- **Web Interface**: User-friendly web application for easy interaction and visualization.
- **Image Upload and Processing**: Allows users to upload images for analysis.
- **Real-time Results**: Provides results in an easy-to-understand format.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Aditya200531/Agri-guard
   cd Agri-guard
   ```
2. **Install dependencies: Ensure you have Python installed, then run**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application: Start the Flask app by running**:
   ```bash
   python app.py
   ```

## Project Structure

* `app.py`: Main application file to run the web server and handle inference.
* `model.tflite`: Pre-trained TensorFlow Lite model for prediction.
* `class_indices.json`: Mapping of class indices to class names for interpreting results.
* `templates/`: Contains HTML templates for the web interface.
* `static/`: Houses static files like CSS and uploaded images.

## Usage

1. Run the application as described above.
2. Upload an image for analysis on the main page.
3. View the prediction results and interpret the findings as shown.

## Requirements

Please refer to `requirements.txt` for a list of dependencies.

