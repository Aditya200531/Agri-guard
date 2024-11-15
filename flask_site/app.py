import os
import json
import uuid
import time
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model from the .pb file
model = tf.saved_model.load("Agri-guard/saved_model")

# Load class labels from a JSON file
with open("Agri-guard/class_indices.json", "r") as f:
    class_labels = json.load(f)
class_labels = {int(k): v for k, v in class_labels.items()}

# Set up Groq Chatbot
GROQ_API_KEY = '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'
GOOGLE_API_KEY = 'AIzaSyBkoorRTaH08H3RFIft4ug6bT1ABexXswI'
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions: {input}
    """
)


def vector_embedding():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    loader = PyPDFDirectoryLoader("model_stuff/wiki_articles")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs[:20])

    vectors = FAISS.from_documents(final_documents, embeddings)
    return vectors

vectors = vector_embedding()
retriever = vectors.as_retriever()
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

def process_image(image_path):
    image = Image.open(image_path).resize((224, 224))
    input_data = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)
    input_tensor = tf.convert_to_tensor(input_data)

    infer = model.signatures['serving_default']
    output_data = infer(input_tensor)
    logits = list(output_data.values())[0].numpy()[0]

    return logits

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            # Create a unique filename and save the image
            unique_filename = str(uuid.uuid4()) + "_" + file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)

            # Log for debugging
            print(f"Image saved at: {file_path}")

            # Store the image and clear the chat history
            session['uploaded_image'] = unique_filename
            session['chat_history'] = []

            # Process the image and get the predicted label
            logits = process_image(file_path)
            predicted_index = np.argmax(logits)
            predicted_label = class_labels[predicted_index]
            session['model_output'] = predicted_label

            return redirect(url_for('results'))
    return render_template('index.html')

@app.route('/results')
def results():
    uploaded_image = session.get('uploaded_image', None)
    model_output = session.get('model_output', None)
    chat_history = session.get('chat_history', [])
    return render_template('results.html',
                           uploaded_image=uploaded_image,
                           model_output=model_output,
                           chat_history=chat_history)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('user_input')
    chat_history = session.get('chat_history', [])

    if user_input:
        response = retrieval_chain.invoke({'input': user_input})
        bot_response = response['answer']

        # Split response into lines and remove duplicates
        formatted_response = bot_response.split('\n')
        unique_lines = set(line.strip() for line in formatted_response if line.strip())  # Remove duplicates and empty lines
        formatted_response = [f"• {line}" for line in unique_lines]

        chat_history.append(f"User: {user_input}")
        chat_history.append({'type': 'bot', 'response': formatted_response})
        session['chat_history'] = chat_history

        return jsonify({'user': user_input, 'bot': formatted_response})
    return jsonify({'error': 'Invalid input'})



if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
