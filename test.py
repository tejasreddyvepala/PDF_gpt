import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import openai
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time

# Initialize Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Set the OpenAI API Key
openai.api_key = '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'

# Model usage counters
usage_counter = {"gpt-3.5-turbo": 0, "gpt-4": 0}
token_limit = {"gpt-3.5-turbo": 200000, "gpt-4": 400000}  # Example token per minute limits
max_tokens_per_call = 120  # To avoid larger-than-allowed requests

# Streamlit UI
st.title("PDF Question & Answer Application")
st.write("Upload PDF files, view FAQs, and ask questions about the content.")

# Function to load and segment documents
def load_and_segment_document(file, max_size=1 * 1024 * 1024):  # Max size in bytes
    doc = fitz.open("pdf", file.read())
    text = []
    for page in doc:
        text.append(page.get_text())
    
    # Segment if document size exceeds max_size
    doc_segments = []
    segment_text = ""
    for paragraph in text:
        segment_text += paragraph
        if len(segment_text.encode('utf-8')) >= max_size:
            doc_segments.append(segment_text)
            segment_text = ""
    if segment_text:  # Add any remaining text as a segment
        doc_segments.append(segment_text)
    return doc_segments

# Function to create a FAISS index with the embeddings
def create_faiss_index(text_segments):
    embeddings = [model.encode(segment) for segment in text_segments]
    embeddings = np.array(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

# Function to retrieve relevant text
def retrieve_relevant_text(question, index, embeddings, text_segments, max_tokens=1500):
    question_embedding = model.encode(question)
    D, I = index.search(np.array([question_embedding]), k=5)
    relevant_text = " ".join([text_segments[i] for i in I[0]])
    
    # Split relevant text into smaller chunks if it exceeds max_tokens
    words = relevant_text.split()
    chunks = [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]
    return chunks

# Function to select model based on rate limits
def select_model():
    # Choose the model with the least usage or one close to the limit
    if usage_counter["gpt-3.5-turbo"] < token_limit["gpt-3.5-turbo"]:
        return "gpt-3.5-turbo"
    elif usage_counter["gpt-4"] < token_limit["gpt-4"]:
        return "gpt-4"
    else:
        time.sleep(60)  # Wait for token limits to reset
        usage_counter["gpt-3.5-turbo"] = 0
        usage_counter["gpt-4"] = 0
        return "gpt-3.5-turbo"  # Default to gpt-3.5-turbo after waiting

# Helper function to generate responses with selected model
def generate_response(question, text_chunks):
    answers = []
    for chunk in text_chunks:
        model_name = select_model()  # Rotate model
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Question: {question}\nContext: {chunk[:max_tokens_per_call]} Answer:"}
            ],
            max_tokens=max_tokens_per_call
        )
        answers.append(response['choices'][0]['message']['content'].strip())
        usage_counter[model_name] += max_tokens_per_call  # Update token usage
    return " ".join(answers)

# Function to generate FAQs with selected model
def generate_faqs(text_segments):
    faqs = []
    for segment in text_segments:
        model_name = select_model()  # Rotate model
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Generate a frequently asked question based on provided text."},
                {"role": "user", "content": f"Generate a frequently asked question based on this text: {segment[:max_tokens_per_call]}"}
            ],
            max_tokens=50
        )
        faqs.append(response['choices'][0]['message']['content'].strip())
        usage_counter[model_name] += 50  # Track tokens used for FAQ generation
    return list(set(faqs))  # Remove duplicate questions

# Step 1: File Upload
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    text_segments_all_docs = []
    with ThreadPoolExecutor() as executor:
        # Load and segment documents in parallel
        segmented_texts = list(executor.map(load_and_segment_document, uploaded_files))
        
    # Flatten the list of segments from all documents
    text_segments_all_docs = [segment for doc_segments in segmented_texts for segment in doc_segments]

    # Create a FAISS index for all document segments
    with st.spinner("Creating index and processing embeddings..."):
        index, embeddings = create_faiss_index(text_segments_all_docs)
    
    # Generate FAQs
    with st.spinner("Generating FAQs..."):
        faqs = generate_faqs(text_segments_all_docs)
    
    st.success("Files processed successfully!")

    # Display FAQs
    st.subheader("Generated FAQs")
    for faq in faqs:
        st.write(f"- {faq}")

    # Step 2: Question Input
    question = st.text_input("Ask a question about the documents:")
    if question:
        # Retrieve relevant text in manageable chunks and generate an answer
        with st.spinner("Finding the answer..."):
            text_chunks = retrieve_relevant_text(question, index, embeddings, text_segments_all_docs)
            answer = generate_response(question, text_chunks)
        
        st.subheader("Answer")
        st.write(answer)
