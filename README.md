# Agri-Guard

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow Lite](https://img.shields.io/badge/TensorFlow%20Lite-2.x-orange)
![Flask](https://img.shields.io/badge/Flask-2.x-green)

**Agri-guard** is a machine-learning-based application designed to support agricultural activities. Leveraging a TensorFlow Lite model, this project provides a web interface for farmers or agricultural experts to assess crop health and analyze agricultural inputs efficiently. Additionally, it includes a chatbot that answers agricultural questions by referencing relevant articles and knowledge sources.

## Table of Contents
- [Features](#features)
- [Chatbot](#chatbot)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)


## Features

- **Machine Learning Inference**: Utilizes TensorFlow Lite models for crop health assessment and agricultural input analysis.
- **Web Interface**: User-friendly interface for interaction and visualization.
- **Chatbot Support**: Integrates a chatbot for answering agricultural questions.
- **CNN-based Analysis**: Implements Convolutional Neural Networks (CNN) for accurate crop health predictions.


## Chatbot

The **Agri-guard chatbot** is designed to answer questions related to agriculture. It uses a set of pre-compiled articles and knowledge sources in the `wiki_articles` directory to provide informative responses. The chatbot is available through the web interface, making it easy for users to get quick, relevant answers to agricultural queries.

### Chatbot Features

- **Knowledge Base**: Utilizes a repository of agricultural articles and documents to answer questions.
- **Natural Language Processing (NLP)**: Processes user queries and provides accurate, contextually relevant responses.
- **Integration**: Accessible through the main Agri-guard web interface.

### Running the Chatbot

The chatbot is integrated into the web app and starts automatically with the Flask server. Simply type your question in the designated input box, and the chatbot will respond with helpful information.

## Project Structure

- `app.py`: Main application file to run the web server and handle inference.
- `model.tflite` and `2.tflite`: Pre-trained TensorFlow Lite models for prediction.
- `class_indices.json`: Mapping of class indices to class names for interpreting results.
- `templates/`: Contains HTML templates for the web interface.
  - `index.html`: Main page for uploading images.
  - `results.html`: Results page displaying predictions.
- `static/`: Houses static files like CSS and uploaded images.
  - `css/styles.css`: Styles for the web interface.
  - `uploads/`: Directory for storing uploaded images.
- `saved_model/`: Contains the saved model files.
- `requirements.txt`: Lists dependencies required by the project.
- `.git/`: Git version control files.

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

## Usage

1. **Run the Web Interface**  
   Navigate to the `flask_site` directory and start the Flask server:
   ```bash
   cd flask_site
   python app.py
   ```
2. **Access the Application**
   Open a browser and go to http://127.0.0.1:5000 to start using Agri-guard.
   

