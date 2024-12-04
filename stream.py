import streamlit as st
import os
import fitz  # PyMuPDF for PDF processing
from sentence_transformers import SentenceTransformer, util
import tempfile
import shutil

# Load model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create directory to store uploaded PDFs if it doesn't exist
upload_dir = 'uploaded_cvs'
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to process uploaded PDFs and save them to the dataset
def process_uploaded_files(uploaded_files):
    for uploaded_file in uploaded_files:
        # Save each uploaded file in the 'uploaded_cvs' directory
        pdf_path = os.path.join(upload_dir, uploaded_file.name)
        with open(pdf_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.write(f"Uploaded: {uploaded_file.name}")
    
    # Extract text from each PDF and store the text for later screening
    cv_texts = {}
    for uploaded_file in uploaded_files:
        pdf_path = os.path.join(upload_dir, uploaded_file.name)
        text = extract_text_from_pdf(pdf_path)
        cv_texts[uploaded_file.name] = text
    
    return cv_texts

# Function to perform CV screening based on user query
def screen_cvs(query, cv_texts):
    # Create embeddings for the user query
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Compute embeddings for each CV and find the top 3 most similar
    scores = {}
    for cv_name, cv_text in cv_texts.items():
        cv_embedding = model.encode(cv_text, convert_to_tensor=True)
        score = util.pytorch_cos_sim(query_embedding, cv_embedding)[0][0].item()
        scores[cv_name] = score
    
    # Sort CVs by similarity score and select top 3
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_3_cvs = sorted_scores[:3]
    
    return top_3_cvs

# Streamlit UI
st.title('CV Screening App')

# Step 1: Upload CVs (placed in the sidebar)
st.sidebar.header('Upload CVs to the Dataset')
uploaded_files = st.sidebar.file_uploader("Upload CVs (PDF format)", accept_multiple_files=True, type="pdf")

if uploaded_files:
    # Process uploaded files
    cv_texts = process_uploaded_files(uploaded_files)

    # Step 2: Input query for CV screening
    st.header('Enter Screening Query')
    query = st.text_area("Enter your query or job description to match CVs:")

    if query:
        # Step 3: Screen the uploaded CVs
        st.header('Top 3 Matching CVs')
        top_3_cvs = screen_cvs(query, cv_texts)

        # Display top 3 matching CVs and their similarity scores
        for i, (cv_name, score) in enumerate(top_3_cvs):
            st.subheader(f"{i+1}. {cv_name} (Score: {score:.4f})")
            st.text(cv_texts[cv_name][:1000])  # Display first 1000 characters of the CV content
            st.markdown(f"[Download CV](file://{os.path.abspath(os.path.join(upload_dir, cv_name))})")  # Link to download CV

else:
    st.sidebar.info('Please upload some CVs to proceed with screening.')

