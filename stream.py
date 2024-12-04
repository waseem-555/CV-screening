import os
import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download
from PIL import Image
import io

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with fitz.open(pdf_path) as pdf:
            for page in pdf:
                text += page.get_text()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

# Function to load PDFs from the directory and compute embeddings
def load_cvs_and_embeddings(cv_dir):
    cv_files = [os.path.join(cv_dir, file) for file in os.listdir(cv_dir) if file.endswith('.pdf')]
    cv_texts = []
    cv_filenames = []
    
    for file in cv_files:
        text = extract_text_from_pdf(file)
        if text.strip():  # Skip if text extraction fails
            cv_texts.append(text)
            cv_filenames.append(file)
        else:
            print(f"Warning: No text extracted from {file}. Skipping...")
    
    cv_embeddings = model.encode(cv_texts, show_progress_bar=True)
    return cv_filenames, cv_embeddings

# Function to find top matching CVs
def find_top_matches(user_query, cv_filenames, cv_embeddings, top_n=3):
    user_embedding = model.encode([user_query])
    similarities = cosine_similarity(user_embedding, cv_embeddings)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    results = []
    for idx in top_indices:
        results.append((cv_filenames[idx], similarities[idx]))
    
    return results

# Function to convert a PDF page to an image using PyMuPDF
def pdf_to_image(pdf_path, page_number=0):
    try:
        with fitz.open(pdf_path) as pdf:
            page = pdf.load_page(page_number)  # Get the specified page
            pix = page.get_pixmap()  # Convert the page to a pixmap (image)
            img = Image.open(io.BytesIO(pix.tobytes()))  # Convert pixmap to an image object
            return img
    except Exception as e:
        print(f"Error converting PDF {pdf_path} to image: {e}")
        return None

# Streamlit interface
def main():
    st.title("CV Screening Application")

    # Upload CVs from the folder or any other mechanism you prefer
    cv_dir = 'CVs'
    st.sidebar.header("Instructions")
    st.sidebar.write("Upload your PDFs in the 'cv_files' directory.")
    
    # Load CVs and embeddings
    st.sidebar.button("Load CVs")
    st.sidebar.text("Loading CVs, please wait...")
    cv_filenames, cv_embeddings = load_cvs_and_embeddings(cv_dir)
    st.sidebar.text(f"Loaded {len(cv_filenames)} CVs.")
    
    # Input box for user query
    user_query = st.text_input("Enter your search criteria for CV matching:")
    
    if user_query:
        st.write(f"**Searching for**: {user_query}")
        
        st.write("### Top 3 Matching CVs:")
        matches = find_top_matches(user_query, cv_filenames, cv_embeddings)
        
        # Display top 3 matches and their PDF previews
        for i, (filename, score) in enumerate(matches, start=1):
            st.write(f"{i}. {filename} (Similarity: {score:.4f})")
            
            # Show the preview of the first page of the matched CV
            pdf_image = pdf_to_image(filename)
            if pdf_image:
                st.image(pdf_image, caption=f"Preview of {os.path.basename(filename)}", use_column_width=True)
            else:
                st.write(f"Failed to load preview for {os.path.basename(filename)}.")

if __name__ == "__main__":
    main()
