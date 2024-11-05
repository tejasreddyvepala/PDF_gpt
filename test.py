import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import openai
import numpy as np

# Initialize Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Set the OpenAI API Key
openai.api_key = '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'


# Streamlit UI
st.title("PDF Question & Answer Application")
st.write("Upload a PDF file, view FAQs, and ask questions about the content.")

# Function to load the document
def load_document(file):
    doc = fitz.open("pdf", file.read())
    text = [page.get_text() for page in doc]
    return text

# Function to create a FAISS index with the embeddings
def create_faiss_index(text):
    embeddings = [model.encode(paragraph) for paragraph in text]
    embeddings = np.array(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

# Function to retrieve relevant text
def retrieve_relevant_text(question, index, embeddings, text):
    question_embedding = model.encode(question)
    D, I = index.search(np.array([question_embedding]), k=5)
    relevant_text = " ".join([text[i] for i in I[0]])
    return relevant_text

# Function to generate an answer with the updated OpenAI ChatCompletion API
def generate_answer(question, relevant_text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Question: {question}\nContext: {relevant_text}\nAnswer:"}
        ],
        max_tokens=150
    )
    return response['choices'][0]['message']['content'].strip()

# Function to generate FAQs
def generate_faqs(text):
    faqs = []
    for paragraph in text:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Generate a frequently asked question based on provided text."},
                {"role": "user", "content": f"Generate a frequently asked question based on this text: {paragraph}"}
            ],
            max_tokens=50
        )
        faqs.append(response['choices'][0]['message']['content'].strip())
    return list(set(faqs))  # Remove duplicate questions

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    # Load document and generate FAQs
    with st.spinner("Processing document..."):
        text = load_document(uploaded_file)
        index, embeddings = create_faiss_index(text)
        faqs = generate_faqs(text)
    
    st.success("File processed successfully!")

    # Display FAQs
    st.subheader("Generated FAQs")
    for faq in faqs:
        st.write(f"- {faq}")

    # Step 2: Question Input
    question = st.text_input("Ask a question about the document:")
    if question:
        # Retrieve relevant text and generate an answer
        with st.spinner("Finding the answer..."):
            relevant_text = retrieve_relevant_text(question, index, embeddings, text)
            answer = generate_answer(question, relevant_text)
        
        st.subheader("Answer")
        st.write(answer)
