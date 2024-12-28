
############################################################################################################
# this is new approach to multimodel rag                                                                   #            
############################################################################################################

import streamlit as st
import fitz
from PIL import Image
import io
from transformers import AutoProcessor, AutoModel
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import os

class MultimodalRAG:
    def __init__(self):
        # Initialize models
        self.text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.image_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.image_model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        
        # Storage for extracted content
        self.text_chunks = []
        self.text_embeddings = []
        self.images = []
        self.image_embeddings = []
        self.image_locations = []
        
    def chunk_text(self, text, chunk_size=1000, overlap=200):
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            chunk = text[start:end]
            
            if end < text_len:
                last_period = chunk.rfind('.')
                if last_period != -1:
                    end = start + last_period + 1
                    chunk = text[start:end]
            
            chunks.append(chunk)
            start = end - overlap
            
        return chunks
        
    def extract_content_from_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        
        # Progress bar for processing
        progress_bar = st.progress(0)
        total_pages = len(doc)
        
        for page_num, page in enumerate(doc):
            # Update progress
            progress_bar.progress((page_num + 1) / total_pages)
            
            # Extract and chunk text
            text = page.get_text()
            if text.strip():
                chunks = self.chunk_text(text)
                for chunk in chunks:
                    if len(chunk.strip()) > 0:
                        self.text_chunks.append(chunk)
                        embedding = self.text_model.encode(chunk)
                        self.text_embeddings.append(embedding)
            
            # Extract images
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    inputs = self.image_processor(images=image, return_tensors="pt")
                    image_features = self.image_model.get_image_features(**inputs)
                    
                    self.images.append(image)
                    self.image_embeddings.append(image_features.detach().numpy())
                    self.image_locations.append((page_num, img_index))
                except Exception as e:
                    st.warning(f"Error processing image {img_index} on page {page_num}: {str(e)}")
                    continue
        
        doc.close()
        progress_bar.empty()  # Remove progress bar when done
        
    def search(self, query, top_k=3):
        results = {
            'text': [],
            'images': [],
            'text_locations': [],
            'image_locations': []
        }
        
        if self.text_embeddings:
            text_query_embedding = self.text_model.encode(query)
            text_similarities = cosine_similarity(
                [text_query_embedding],
                self.text_embeddings
            )[0]
            
            top_text_indices = np.argsort(text_similarities)[-top_k:][::-1]
            results['text'] = [self.text_chunks[i] for i in top_text_indices]
            results['text_locations'] = top_text_indices
        
        if self.image_embeddings:
            image_query_inputs = self.image_processor(text=[query], return_tensors="pt", padding=True)
            image_query_features = self.image_model.get_text_features(**image_query_inputs)
            
            image_similarities = cosine_similarity(
                image_query_features.detach().numpy(),
                np.vstack(self.image_embeddings)
            )[0]
            
            top_image_indices = np.argsort(image_similarities)[-top_k:][::-1]
            results['images'] = [self.images[i] for i in top_image_indices]
            results['image_locations'] = [self.image_locations[i] for i in top_image_indices]
        
        return results

def main():
    st.set_page_config(page_title="Multimodal PDF RAG System", layout="wide")
    st.title("Multimodal PDF RAG System")
    
    # Initialize session state
    if 'rag' not in st.session_state:
        st.session_state.rag = MultimodalRAG()
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False

    # Sidebar for file upload and system reset
    with st.sidebar:
        st.header("Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if st.button("Reset System"):
            st.session_state.rag = MultimodalRAG()
            st.session_state.pdf_processed = False
            st.rerun()

    # Main content area
    if uploaded_file is not None and not st.session_state.pdf_processed:
        pdf_path = "temp.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        with st.spinner('Processing PDF...'):
            st.session_state.rag.extract_content_from_pdf(pdf_path)
            st.session_state.pdf_processed = True
        
        # Clean up temporary file
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        
        st.success(f"PDF processed successfully! Found {len(st.session_state.rag.text_chunks)} text chunks and {len(st.session_state.rag.images)} images.")

    # Query interface
    if st.session_state.pdf_processed:
        st.header("Ask Questions")
        query = st.text_input("Enter your question:")
        
        if query:
            with st.spinner('Searching...'):
                results = st.session_state.rag.search(query)

            # Display results
            col1, col2 = st.columns([3, 2])

            with col1:
                st.subheader("Text Results")
                if results['text']:
                    for i, text in enumerate(results['text'], 1):
                        with st.expander(f"Text Result {i}", expanded=(i==1)):
                            st.write(text)
                else:
                    st.info("No relevant text found.")

            with col2:
                st.subheader("Image Results")
                if results['images']:
                    for i, (img, loc) in enumerate(zip(results['images'], results['image_locations']), 1):
                        st.image(img, caption=f"Image {i} (Page {loc[0] + 1}, Image {loc[1] + 1})")
                else:
                    st.info("No relevant images found.")

    else:
        st.info("👆 Please upload a PDF file using the sidebar to begin.")

if __name__ == "__main__":
    main()










