# import fitz
# from PIL import Image
# import io
# from transformers import AutoProcessor, AutoModel
# from sentence_transformers import SentenceTransformer
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# import torch
# import os

# class MultimodalRAG:
#     def __init__(self):
#         # Initialize models
#         print("Initializing models...")
#         self.text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#         self.image_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
#         self.image_model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        
#         # Storage for extracted content
#         self.text_chunks = []
#         self.text_embeddings = []
#         self.images = []
#         self.image_embeddings = []
#         self.image_locations = []
        
#     def chunk_text(self, text, chunk_size=1000, overlap=200):
#         chunks = []
#         start = 0
#         text_len = len(text)
        
#         while start < text_len:
#             end = start + chunk_size
#             chunk = text[start:end]
            
#             if end < text_len:
#                 last_period = chunk.rfind('.')
#                 if last_period != -1:
#                     end = start + last_period + 1
#                     chunk = text[start:end]
            
#             chunks.append(chunk)
#             start = end - overlap
            
#         return chunks
        
#     def extract_content_from_pdf(self, pdf_path):
#         print(f"Processing PDF: {pdf_path}")
#         doc = fitz.open(pdf_path)
#         total_pages = len(doc)
        
#         for page_num, page in enumerate(doc):
#             print(f"Processing page {page_num + 1}/{total_pages}")
            
#             # Extract and chunk text
#             text = page.get_text()
#             if text.strip():
#                 chunks = self.chunk_text(text)
#                 for chunk in chunks:
#                     if len(chunk.strip()) > 0:
#                         self.text_chunks.append(chunk)
#                         embedding = self.text_model.encode(chunk)
#                         self.text_embeddings.append(embedding)
            
#             # Extract images
#             image_list = page.get_images(full=True)
#             for img_index, img in enumerate(image_list):
#                 try:
#                     xref = img[0]
#                     base_image = doc.extract_image(xref)
#                     image_bytes = base_image["image"]
                    
#                     image = Image.open(io.BytesIO(image_bytes))
                    
#                     inputs = self.image_processor(images=image, return_tensors="pt")
#                     image_features = self.image_model.get_image_features(**inputs)
                    
#                     self.images.append(image)
#                     self.image_embeddings.append(image_features.detach().numpy())
#                     self.image_locations.append((page_num, img_index))
#                 except Exception as e:
#                     print(f"Error processing image {img_index} on page {page_num}: {str(e)}")
#                     continue
        
#         doc.close()
#         print(f"PDF processing complete. Found {len(self.text_chunks)} text chunks and {len(self.images)} images.")
    
#     def search(self, query, top_k=3):
#         results = {
#             'text': [],
#             'images': [],
#             'text_locations': [],
#             'image_locations': []
#         }
        
#         if self.text_embeddings:
#             print("Searching text...")
#             text_query_embedding = self.text_model.encode(query)
#             text_similarities = cosine_similarity(
#                 [text_query_embedding],
#                 self.text_embeddings
#             )[0]
            
#             top_text_indices = np.argsort(text_similarities)[-top_k:][::-1]
#             results['text'] = [self.text_chunks[i] for i in top_text_indices]
#             results['text_locations'] = top_text_indices
        
#         if self.image_embeddings:
#             print("Searching images...")
#             image_query_inputs = self.image_processor(text=[query], return_tensors="pt", padding=True)
#             image_query_features = self.image_model.get_text_features(**image_query_inputs)
            
#             image_similarities = cosine_similarity(
#                 image_query_features.detach().numpy(),
#                 np.vstack(self.image_embeddings)
#             )[0]
            
#             top_image_indices = np.argsort(image_similarities)[-top_k:][::-1]
#             results['images'] = [self.images[i] for i in top_image_indices]
#             results['image_locations'] = [self.image_locations[i] for i in top_image_indices]
        
#         return results

# # Example usage
# def main():
#     # Initialize the RAG system
#     rag = MultimodalRAG()
    
#     # Process PDF
#     pdf_path = "/home/cair/Downloads/multimodel/temp.pdf"  # Replace with your PDF path
#     rag.extract_content_from_pdf(pdf_path)
    
#     # Interactive query loop
#     while True:
#         query = input("\nEnter your question (or 'quit' to exit): ")
#         if query.lower() == 'quit':
#             break
            
#         results = rag.search(query)
        
#         # Display text results
#         print("\nText Results:")
#         if results['text']:
#             for i, text in enumerate(results['text'], 1):
#                 print(f"\nResult {i}:")
#                 print("-" * 50)
#                 print(text)
#         else:
#             print("No relevant text found.")
            
#         # Display image locations (since we can't display images in console)
#         print("\nImage Results:")
#         if results['image_locations']:
#             for i, loc in enumerate(results['image_locations'], 1):
#                 print(f"Image {i} found on Page {loc[0] + 1}, Image {loc[1] + 1}")
#         else:
#             print("No relevant images found.")

# if __name__ == "__main__":
#     main()










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
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
import gc

class MultimodalRAG:
    def __init__(self):
        # Initialize models on GPU
        self.text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.image_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.image_model = AutoModel.from_pretrained("openai/clip-vit-base-patch32").to('cuda')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Storage for extracted content
        self.text_chunks = []
        self.text_embeddings = []
        self.images = []
        self.image_embeddings = []
        self.image_locations = []
        
        self.batch_size = 16  # Batch size for text embedding
    
    def clear_gpu_memory(self):
        """Clear GPU memory to prevent memory overflow."""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    def process_text_batch(self, text_batch):
        """Encode a batch of text on GPU."""
        embeddings = self.text_model.encode(text_batch, convert_to_tensor=True, device=self.device)
        return embeddings.cpu().numpy()  # Move to CPU for storage
    
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
        st.write(f"Processing PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        # Create progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()
        current_text_batch = []
        
        try:
            for page_num, page in enumerate(doc):
                # Update progress
                progress = (page_num + 1) / total_pages
                progress_bar.progress(progress)
                progress_text.text(f"Processing page {page_num + 1}/{total_pages}")
                
                # Extract and chunk text
                text = page.get_text()
                if text.strip():
                    chunks = self.chunk_text(text)
                    current_text_batch.extend(chunks)
                    
                    # Process text batch if it reaches batch size
                    if len(current_text_batch) >= self.batch_size:
                        self.text_chunks.extend(current_text_batch)
                        embeddings = self.process_text_batch(current_text_batch)
                        self.text_embeddings.extend(embeddings)
                        current_text_batch = []
                        self.clear_gpu_memory()
                
                # Extract images
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        image = Image.open(io.BytesIO(image_bytes))
                        
                        # Process image with GPU
                        inputs = self.image_processor(images=image, return_tensors="pt")
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        
                        with torch.no_grad():
                            image_features = self.image_model.get_image_features(**inputs)
                            image_features = image_features.cpu().numpy()
                        
                        self.images.append(image)
                        self.image_embeddings.append(image_features)
                        self.image_locations.append((page_num, img_index))
                        self.clear_gpu_memory()
                        
                    except Exception as e:
                        st.warning(f"Error processing image {img_index} on page {page_num}: {str(e)}")
                        continue
                
                self.clear_gpu_memory()
            
            # Process any remaining text
            if current_text_batch:
                self.text_chunks.extend(current_text_batch)
                embeddings = self.process_text_batch(current_text_batch)
                self.text_embeddings.extend(embeddings)
            
        finally:
            progress_bar.empty()
            progress_text.empty()
            doc.close()
            self.clear_gpu_memory()
        
        st.success(f"PDF processing complete. Found {len(self.text_chunks)} text chunks and {len(self.images)} images.")

        
        st.success(f"PDF processing complete. Found {len(self.text_chunks)} text chunks and {len(self.images)} images.")
    
    def search(self, query, top_k=3):
        results = {
            'text': [],
            'images': [],
            'text_locations': [],
            'image_locations': []
        }
        
        if self.text_embeddings:
            st.write("Searching text...")
            with torch.no_grad():
                text_query_embedding = self.text_model.encode(query, convert_to_tensor=True)
                text_query_embedding = text_query_embedding.cpu().numpy()
            
            # Process similarities in batches to save memory
            batch_size = 1000
            text_similarities = []
            for i in range(0, len(self.text_embeddings), batch_size):
                batch = np.array(self.text_embeddings[i:i + batch_size])
                batch_similarities = cosine_similarity([text_query_embedding], batch)[0]
                text_similarities.extend(batch_similarities)
            
            top_text_indices = np.argsort(text_similarities)[-top_k:][::-1]
            results['text'] = [self.text_chunks[i] for i in top_text_indices]
            results['text_locations'] = top_text_indices
            self.clear_gpu_memory()
        
        if self.image_embeddings:
            st.write("Searching images...")
            image_query_inputs = self.image_processor(text=[query], return_tensors="pt", padding=True)
            image_query_inputs = {k: v.to(self.device) for k, v in image_query_inputs.items()}
            
            with torch.no_grad():
                image_query_features = self.image_model.get_text_features(**image_query_inputs)
                image_query_features = image_query_features.cpu().numpy()
            
            # Process similarities in batches
            batch_size = 1000
            image_similarities = []
            for i in range(0, len(self.image_embeddings), batch_size):
                batch = np.vstack(self.image_embeddings[i:i + batch_size])
                batch_similarities = cosine_similarity(image_query_features, batch)[0]
                image_similarities.extend(batch_similarities)
            
            top_image_indices = np.argsort(image_similarities)[-top_k:][::-1]
            results['images'] = [self.images[i] for i in top_image_indices]
            results['image_locations'] = [self.image_locations[i] for i in top_image_indices]
            self.clear_gpu_memory()
        
        return results

def main():
    st.set_page_config(page_title="Multimodal PDF RAG System", layout="wide")
    st.title("Multimodal PDF RAG System")
    
    # Initialize session state
    if 'rag' not in st.session_state:
        st.session_state.rag = MultimodalRAG()
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False

    # Display GPU status
    if torch.cuda.is_available():
        st.sidebar.success(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.sidebar.warning("GPU not available, using CPU")

    # Sidebar for file upload and system reset
    with st.sidebar:
        st.header("Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        batch_size = st.slider("Batch Size", min_value=1, max_value=64, value=32, 
                             help="Adjust batch size based on your GPU memory")
        
        if st.button("Reset System"):
            st.session_state.rag = MultimodalRAG()
            st.session_state.rag.batch_size = batch_size
            st.session_state.pdf_processed = False
            st.rerun()

    # Main content area
    if uploaded_file is not None and not st.session_state.pdf_processed:
        pdf_path = "temp.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        try:
            with st.spinner('Processing PDF...'):
                st.session_state.rag.batch_size = batch_size
                st.session_state.rag.extract_content_from_pdf(pdf_path)
                st.session_state.pdf_processed = True
            
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
        finally:
            # Clean up temporary file
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

    # Query interface
    if st.session_state.pdf_processed:
        st.header("Ask Questions")
        query = st.text_input("Enter your question:")
        
        if query:
            try:
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
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")

    else:
        st.info("👆 Please upload a PDF file using the sidebar to begin.")

if __name__ == "__main__":
    main()