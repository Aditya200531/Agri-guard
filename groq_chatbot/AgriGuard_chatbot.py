import os
import time
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

groq_api_key = 'GROQ_API_KEY'
google_api_key = 'GOOGLE_API_KEY'

# Check for missing API keys
if not groq_api_key or not google_api_key:
    raise ValueError("Missing GROQ_API_KEY or GOOGLE_API_KEY. Please set them in your .env file.")

# Set the Google API key in the environment
os.environ["GOOGLE_API_KEY"] = google_api_key

# Initialize the language model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt template
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
    print("Initializing vector embedding...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    loader = PyPDFDirectoryLoader("model_stuff/wiki_articles")  # Data Ingestion
    docs = loader.load()  # Document Loading

    if not docs:
        print("Error: No documents found in the specified directory.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
    final_documents = text_splitter.split_documents(docs[:20])  # Splitting

    if not final_documents:
        print("Error: Document splitting resulted in empty chunks.")
        return None

    if not embeddings:
        print("Error: Embeddings initialization failed.")
        return None

    vectors = FAISS.from_documents(final_documents, embeddings)  # Create vector store
    print("Vector Store DB is ready.")
    return vectors

def main():
    vectors = vector_embedding()
    if not vectors:
        print("Error: Vector store creation failed.")
        return

    retriever = vectors.as_retriever()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # User input for the question
    user_question = input("Enter your question from documents: ")

    if user_question:
        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_question})
        print("Response time: ", time.process_time() - start)
        print("\nAnswer:\n", response['answer'])

        # Display the document similarity search results
        print("\nDocument Similarity Search Results:\n")
        for i, doc in enumerate(response.get("context", [])):
            print(doc.page_content)
            print("--------------------------------")

if __name__ == "__main__":
    main()