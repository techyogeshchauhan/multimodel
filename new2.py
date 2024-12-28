# # import streamlit as st
# # import fitz
# # from PIL import Image
# # import io
# # import pytesseract
# # from transformers import AutoProcessor, AutoModel
# # from sentence_transformers import SentenceTransformer
# # import numpy as np
# # from sklearn.metrics.pairwise import cosine_similarity
# # import torch
# # import os
# # import cv2

# # class MultimodalRAG:

# #     def __init__(self):
# #         self.text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# #         self.image_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
# #         self.image_model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        
# #         self.text_chunks = []
# #         self.text_embeddings = []
# #         self.images = []
# #         self.image_embeddings = []
# #         self.image_locations = []

# #     def enhance_image(self, image):
# #         """Enhanced image preprocessing for better OCR results"""
# #         # Convert PIL Image to OpenCV format
# #         img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
# #         # Convert to grayscale
# #         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
# #         # Apply deskewing
# #         coords = np.column_stack(np.where(gray > 0))
# #         angle = cv2.minAreaRect(coords)[-1]
# #         if angle < -45:
# #             angle = 90 + angle
# #         (h, w) = gray.shape[:2]
# #         center = (w // 2, h // 2)
# #         M = cv2.getRotationMatrix2D(center, angle, 1.0)
# #         rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
# #         # Apply adaptive thresholding with optimized parameters
# #         thresh = cv2.adaptiveThreshold(
# #             rotated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15
# #         )
        
# #         # Apply morphological operations to remove noise
# #         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# #         morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
# #         # Denoise with optimized parameters
# #         denoised = cv2.fastNlMeansDenoising(morph, None, 10, 7, 21)
        
# #         # Increase contrast
# #         enhanced = cv2.convertScaleAbs(denoised, alpha=1.2, beta=0)
        
# #         return Image.fromarray(enhanced)

# #     def extract_text_from_image(self, image):
# #         """Enhanced OCR with better image preprocessing"""
# #         enhanced_image = self.enhance_image(image)
        
# #         # Configure OCR parameters for better accuracy
# #         custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!? " -l eng'
        
# #         # Perform OCR with confidence check
# #         text = pytesseract.image_to_string(enhanced_image, config=custom_config)
        
# #         # Clean up the extracted text
# #         cleaned_text = ' '.join(text.split())
# #         return cleaned_text.strip()

# #     def chunk_text(self, text, chunk_size=500, overlap=100):
# #         """Optimized text chunking for better retrieval"""
# #         chunks = []
# #         start = 0
# #         text_len = len(text)
        
# #         while start < text_len:
# #             end = start + chunk_size
# #             chunk = text[start:end]
            
# #             if end < text_len:
# #                 # Try to break at sentence boundary
# #                 last_period = chunk.rfind('.')
# #                 last_question = chunk.rfind('?')
# #                 last_exclamation = chunk.rfind('!')
                
# #                 break_point = max(last_period, last_question, last_exclamation)
# #                 if break_point != -1:
# #                     end = start + break_point + 1
# #                     chunk = text[start:end]
            
# #             if len(chunk.strip()) > 50:  # Only keep chunks with substantial content
# #                 chunks.append(chunk)
# #             start = end - overlap
            
# #         return chunks

# #     def extract_content_from_pdf(self, pdf_path):
# #         """Enhanced PDF content extraction"""
# #         doc = fitz.open(pdf_path)
        
# #         progress_bar = st.progress(0)
# #         status_text = st.empty()
# #         total_pages = len(doc)
        
# #         for page_num, page in enumerate(doc):
# #             progress = (page_num + 1) / total_pages
# #             progress_bar.progress(progress)
# #             status_text.text(f"Processing page {page_num + 1} of {total_pages}")
            
# #             # Get high-resolution page image
# #             pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
# #             img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
# #             # Extract text using enhanced OCR
# #             ocr_text = self.extract_text_from_image(img)
            
# #             # Get native PDF text as backup
# #             pdf_text = page.get_text()
            
# #             # Combine texts, giving preference to PDF text if available
# #             combined_text = pdf_text if len(pdf_text.strip()) > len(ocr_text.strip()) else ocr_text
            
# #             if combined_text.strip():
# #                 chunks = self.chunk_text(combined_text)
# #                 for chunk in chunks:
# #                     if len(chunk.strip()) > 0:
# #                         self.text_chunks.append(chunk)
# #                         embedding = self.text_model.encode(chunk)
# #                         self.text_embeddings.append(embedding)
            
# #             # Process the page image for visual search
# #             try:
# #                 inputs = self.image_processor(images=img, return_tensors="pt")
# #                 image_features = self.image_model.get_image_features(**inputs)
                
# #                 self.images.append(img)
# #                 self.image_embeddings.append(image_features.detach().numpy())
# #                 self.image_locations.append((page_num, 0))
# #             except Exception as e:
# #                 st.warning(f"Error processing page {page_num} image: {str(e)}")
        
# #         doc.close()
# #         progress_bar.empty()
# #         status_text.empty()

# #     def search(self, query, top_k=3):
# #         """Improved search with better ranking"""
# #         results = {
# #             'text': [],
# #             'images': [],
# #             'text_locations': [],
# #             'image_locations': []
# #         }
        
# #         if self.text_embeddings:
# #             text_query_embedding = self.text_model.encode(query)
# #             text_similarities = cosine_similarity(
# #                 [text_query_embedding],
# #                 self.text_embeddings
# #             )[0]
            
# #             # Filter out low-confidence matches
# #             confidence_threshold = 0.3
# #             valid_indices = np.where(text_similarities > confidence_threshold)[0]
# #             if len(valid_indices) > 0:
# #                 top_text_indices = valid_indices[np.argsort(text_similarities[valid_indices])[-top_k:][::-1]]
# #                 results['text'] = [self.text_chunks[i] for i in top_text_indices]
# #                 results['text_locations'] = top_text_indices
        
# #         if self.image_embeddings:
# #             image_query_inputs = self.image_processor(text=[query], return_tensors="pt", padding=True)
# #             image_query_features = self.image_model.get_text_features(**image_query_inputs)
            
# #             image_similarities = cosine_similarity(
# #                 image_query_features.detach().numpy(),
# #                 np.vstack(self.image_embeddings)
# #             )[0]
            
# #             # Filter out low-confidence image matches
# #             image_threshold = 0.2
# #             valid_image_indices = np.where(image_similarities > image_threshold)[0]
# #             if len(valid_image_indices) > 0:
# #                 top_image_indices = valid_image_indices[np.argsort(image_similarities[valid_image_indices])[-top_k:][::-1]]
# #                 results['images'] = [self.images[i] for i in top_image_indices]
# #                 results['image_locations'] = [self.image_locations[i] for i in top_image_indices]
        
# #         return results

# # def main():
# #     st.set_page_config(
# #         page_title="PDF RAG System",
# #         page_icon="📚",
# #         layout="wide",
# #         initial_sidebar_state="expanded"
# #     )
    
# #     st.markdown("""
# #         <style>
# #         .stAlert {
# #             background-color: #f0f2f6;
# #             padding: 1rem;
# #             border-radius: 0.5rem;
# #         }
# #         .stProgress > div > div > div {
# #             background-color: #00a0a0;
# #         }
# #         </style>
# #     """, unsafe_allow_html=True)
    
# #     st.title("📚 PDF Question Answering System")
    
# #     if 'rag' not in st.session_state:
# #         st.session_state.rag = MultimodalRAG()
# #     if 'pdf_processed' not in st.session_state:
# #         st.session_state.pdf_processed = False

# #     with st.sidebar:
# #         st.header("📄 Upload PDF")
# #         uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
# #         if st.button("🔄 Reset"):
# #             st.session_state.rag = MultimodalRAG()
# #             st.session_state.pdf_processed = False
# #             st.rerun()

# #     if uploaded_file is not None and not st.session_state.pdf_processed:
# #         pdf_path = "temp.pdf"
# #         with open(pdf_path, "wb") as f:
# #             f.write(uploaded_file.getvalue())

# #         with st.spinner('Processing PDF...'):
# #             st.session_state.rag.extract_content_from_pdf(pdf_path)
# #             st.session_state.pdf_processed = True
        
# #         if os.path.exists(pdf_path):
# #             os.remove(pdf_path)
        
# #         st.success("✅ PDF processed!")

# #     if st.session_state.pdf_processed:
# #         st.header("🔍 Ask Questions")
# #         query = st.text_input("Enter your question:")
        
# #         if query:
# #             with st.spinner('🔎 Searching...'):
# #                 results = st.session_state.rag.search(query)

# #             col1, col2 = st.columns([3, 2])

# #             with col1:
# #                 st.subheader("📝 Text Results")
# #                 if results['text']:
# #                     for i, text in enumerate(results['text'], 1):
# #                         with st.expander(f"Text Result {i}", expanded=(i==1)):
# #                             st.markdown(f"```\n{text}\n```")
# #                 else:
# #                     st.info("No relevant text found.")

# #             with col2:
# #                 st.subheader("🖼️ Image Results")
# #                 if results['images']:
# #                     for i, (img, loc) in enumerate(zip(results['images'], results['image_locations']), 1):
# #                         st.image(img, caption=f"Image {i} (Page {loc[0] + 1})")
# #                 else:
# #                     st.info("No relevant images found.")

# #     else:
# #         st.info("👆 Upload a PDF to begin.")

# # if __name__ == "__main__":
# #     main()



# # this is new approach to the problem of creating PDF files   from    images and text files

# import streamlit as st
# import fitz
# from PIL import Image
# import io
# import pytesseract
# from transformers import AutoProcessor, AutoModel
# from sentence_transformers import SentenceTransformer
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# import torch
# import os
# import cv2
# import re

# class MultimodalRAG:
#     def __init__(self):
#         self.text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#         self.image_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
#         self.image_model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        
#         self.text_chunks = []
#         self.text_embeddings = []
#         self.images = []
#         self.image_embeddings = []
#         self.image_locations = []
#         self.chapter_info = {}

#     def enhance_image(self, image):
#         img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         blurred = cv2.GaussianBlur(gray, (3, 3), 0)
#         thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#         opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
#         enhanced = cv2.convertScaleAbs(opening, alpha=1.5, beta=0)
#         return Image.fromarray(enhanced)

#     def detect_chapter_info(self, text):
#         patterns = [
#             r'chapter\s+(\d+|[IVXivx]+)[:\s]+(.+?)(?=\n|$)',
#             r'(\d+|[IVXivx]+)\.\s+(.+?)(?=\n|$)',
#             r'CHAPTER\s+(\d+|[IVXivx]+)[:\s]+(.+?)(?=\n|$)'
#         ]
        
#         for pattern in patterns:
#             matches = re.finditer(pattern, text, re.IGNORECASE)
#             for match in matches:
#                 return {
#                     'number': match.group(1),
#                     'title': match.group(2).strip()
#                 }
#         return None

#     def extract_text_from_image(self, image):
#         enhanced_image = self.enhance_image(image)
#         configs = [
#             '--oem 3 --psm 6 -l eng',
#             '--oem 3 --psm 3 -l eng',
#             '--oem 3 --psm 4 -l eng'
#         ]
        
#         best_text = ""
#         max_confidence = 0
        
#         for config in configs:
#             try:
#                 text = pytesseract.image_to_string(enhanced_image, config=config)
#                 data = pytesseract.image_to_data(enhanced_image, output_type=pytesseract.Output.DICT)
#                 conf_scores = [int(x) for x in data['conf'] if x != '-1']
                
#                 if conf_scores:
#                     avg_confidence = sum(conf_scores) / len(conf_scores)
#                     if avg_confidence > max_confidence:
#                         max_confidence = avg_confidence
#                         best_text = text
#             except Exception:
#                 continue
        
#         return best_text.strip()

#     def extract_content_from_pdf(self, pdf_path):
#         doc = fitz.open(pdf_path)
        
#         progress_bar = st.progress(0)
#         status_text = st.empty()
        
#         for page_num, page in enumerate(doc):
#             progress = (page_num + 1) / len(doc)
#             progress_bar.progress(progress)
#             status_text.text(f"Processing page {page_num + 1}/{len(doc)}")
            
#             # Extract text using multiple methods
#             pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
#             img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
#             ocr_text = self.extract_text_from_image(img)
#             pdf_text = page.get_text()
            
#             # Combine texts
#             combined_text = f"{pdf_text}\n{ocr_text}"
            
#             # Process text
#             if combined_text.strip():
#                 chapter_info = self.detect_chapter_info(combined_text)
#                 if chapter_info:
#                     chunk = f"CHAPTER {chapter_info['number']}: {chapter_info['title']}"
#                     self.text_chunks.append(chunk)
#                     self.chapter_info[len(self.text_chunks)-1] = chapter_info
#                     embedding = self.text_model.encode(chunk)
#                     self.text_embeddings.append(embedding)
                
#                 # Process regular text
#                 words = combined_text.split()
#                 for i in range(0, len(words), 100):
#                     chunk = ' '.join(words[i:i+100])
#                     if len(chunk.strip()) > 0:
#                         self.text_chunks.append(chunk)
#                         embedding = self.text_model.encode(chunk)
#                         self.text_embeddings.append(embedding)
            
#             # Process images
#             try:
#                 inputs = self.image_processor(images=img, return_tensors="pt")
#                 image_features = self.image_model.get_image_features(**inputs)
#                 self.images.append(img)
#                 self.image_embeddings.append(image_features.detach().numpy())
#                 self.image_locations.append((page_num, 0))
#             except Exception as e:
#                 st.warning(f"Error processing page {page_num} image: {str(e)}")
        
#         doc.close()
#         progress_bar.empty()
#         status_text.empty()

#     def search(self, query, top_k=3):
#         results = {
#             'text': [],
#             'images': [],
#             'text_locations': [],
#             'image_locations': []
#         }
        
#         # Check for chapter queries
#         chapter_match = re.search(r'chapter\s+(\d+|[IVXivx]+)', query, re.IGNORECASE)
#         if chapter_match:
#             chapter_num = chapter_match.group(1)
#             for i, chunk in enumerate(self.text_chunks):
#                 if i in self.chapter_info and str(self.chapter_info[i]['number']) == chapter_num:
#                     results['text'] = self.text_chunks[i:i+3]
#                     results['text_locations'] = list(range(i, min(i+3, len(self.text_chunks))))
#                     break
        
#         # Regular search
#         if not results['text']:
#             if self.text_embeddings:
#                 query_embedding = self.text_model.encode(query)
#                 similarities = cosine_similarity([query_embedding], self.text_embeddings)[0]
#                 top_indices = np.argsort(similarities)[-top_k:][::-1]
#                 results['text'] = [self.text_chunks[i] for i in top_indices]
#                 results['text_locations'] = top_indices
        
#         # Image search
#         if self.image_embeddings:
#             image_query = self.image_processor(text=[query], return_tensors="pt", padding=True)
#             image_features = self.image_model.get_text_features(**image_query)
#             similarities = cosine_similarity(
#                 image_features.detach().numpy(),
#                 np.vstack(self.image_embeddings)
#             )[0]
#             top_indices = np.argsort(similarities)[-top_k:][::-1]
#             results['images'] = [self.images[i] for i in top_indices]
#             results['image_locations'] = [self.image_locations[i] for i in top_indices]
        
#         return results

# def main():
#     st.set_page_config(page_title="PDF RAG System", layout="wide")
    
#     st.title("📚 RAG    PDF Question Answering System")
    
#     if 'rag' not in st.session_state:
#         st.session_state.rag = MultimodalRAG()
#     if 'pdf_processed' not in st.session_state:
#         st.session_state.pdf_processed = False

#     with st.sidebar:
#         st.header("📄 Upload PDF")
#         uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
#         if st.button("🔄 Reset"):
#             st.session_state.rag = MultimodalRAG()
#             st.session_state.pdf_processed = False
#             st.rerun()

#     if uploaded_file is not None and not st.session_state.pdf_processed:
#         pdf_path = "temp.pdf"
#         with open(pdf_path, "wb") as f:
#             f.write(uploaded_file.getvalue())

#         with st.spinner('Processing PDF...'):
#             st.session_state.rag.extract_content_from_pdf(pdf_path)
#             st.session_state.pdf_processed = True
        
#         if os.path.exists(pdf_path):
#             os.remove(pdf_path)
        
#         st.success("✅ PDF processed!")

#     if st.session_state.pdf_processed:
#         st.header("🔍 Ask Questions")
#         query = st.text_input("Enter your question (e.g., 'What is Chapter 8 about?'):")
        
#         if query:
#             with st.spinner('🔎 Searching...'):
#                 results = st.session_state.rag.search(query)

#             col1, col2 = st.columns([3, 2])

#             with col1:
#                 st.subheader("📝 Text Results")
#                 if results['text']:
#                     for i, text in enumerate(results['text'], 1):
#                         with st.expander(f"Text Result {i}", expanded=(i==1)):
#                             st.markdown(f"```\n{text}\n```")
#                 else:
#                     st.info("No relevant text found.")

#             with col2:
#                 st.subheader("🖼️ Image Results")
#                 if results['images']:
#                     for i, (img, loc) in enumerate(zip(results['images'], results['image_locations']), 1):
#                         st.image(img, caption=f"Image {i} (Page {loc[0] + 1})")
#                 else:
#                     st.info("No relevant images found.")

#     else:
#         st.info("👆 Upload a PDF to begin.")

# if __name__ == "__main__":
#     main()






# indent error
# import streamlit as st
# import fitz
# from PIL import Image
# import io
# from transformers import AutoProcessor, AutoModel, AutoTokenizer, AutoModelForCausalLM
# from sentence_transformers import SentenceTransformer
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# import torch
# import os

# class MultimodalRAG:
#     def __init__(self):
#         self.text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#         self.image_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
#         self.image_model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
#         self.chat_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
#         self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
#         # self.chat_model = AutoModelForCausalLM.from_pretrained("gpt2")
#         # self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
#         self.text_chunks = []
#         self.text_embeddings = []
#         self.images = []
#         self.image_embeddings = []
#         self.image_locations = []
#         self.page_text_map = {}  # Maps page numbers to text content
        
#     def chunk_text(self, text, chunk_size=500, overlap=100):
#         words = text.split()
#         chunks = []
#         start = 0
        
#         while start < len(words):
#             chunk = ' '.join(words[start:start + chunk_size])
#             chunks.append(chunk)
#             start += chunk_size - overlap
            
#         return chunks

#     def process_text(self, text, page_num):
#         if not text.strip():
#             return
            
#         if page_num not in self.page_text_map:
#             self.page_text_map[page_num] = []
            
#         chunks = self.chunk_text(text)
#         for chunk in chunks:
#             if len(chunk.strip()) > 50:
#                 self.text_chunks.append(chunk)
#                 self.page_text_map[page_num].append(chunk)
#                 embedding = self.text_model.encode(chunk)
#                 self.text_embeddings.append(embedding)
        
#     def extract_content_from_pdf(self, pdf_path):
#         doc = fitz.open(pdf_path)
        
#         progress_bar = st.progress(0)
#         total_pages = len(doc)
        
#         for page_num, page in enumerate(doc):
#             progress_bar.progress((page_num + 1) / total_pages)
            
#             blocks = page.get_text("blocks")
#             for block in blocks:
#                 text = block[4]
#                 self.process_text(text, page_num)
            
#             image_list = page.get_images(full=True)
#             for img_index, img in enumerate(image_list):
#                 try:
#                     xref = img[0]
#                     base_image = doc.extract_image(xref)
#                     image_bytes = base_image["image"]
                    
#                     image = Image.open(io.BytesIO(image_bytes))
#                     if image.mode == 'RGBA':
#                         image = image.convert('RGB')
                    
#                     inputs = self.image_processor(images=image, return_tensors="pt")
#                     image_features = self.image_model.get_image_features(**inputs)
                    
#                     self.images.append(image)
#                     self.image_embeddings.append(image_features.detach().numpy())
#                     self.image_locations.append((page_num, img_index))
#                 except Exception as e:
#                     st.warning(f"Error processing image {img_index} on page {page_num}: {str(e)}")
#                     continue
        
#         doc.close()
#         progress_bar.empty()
        
#         if self.text_embeddings:
#             self.text_embeddings = np.vstack(self.text_embeddings)
    
#     def search(self, query, top_k=3):
#         results = {
#             'text': [],
#             'images': [],
#             'text_locations': [],
#             'image_locations': [],
#             'context': []
#         }

        
        
#         if len(self.text_chunks) > 0:
#             text_query_embedding = self.text_model.encode(query)
#             text_similarities = cosine_similarity(
#                 [text_query_embedding],
#                 self.text_embeddings
#             )[0]
            
#             top_text_indices = np.argsort(text_similarities)[-top_k:][::-1]
#             results['text'] = [self.text_chunks[i] for i in top_text_indices]
#             results['text_locations'] = top_text_indices
            
#             # Get surrounding context for each text chunk
#             for idx in top_text_indices:
#                 page_num = None
#                 for p, chunks in self.page_text_map.items():
#                     if self.text_chunks[idx] in chunks:
#                         page_num = p
#                         break
                        
#                 if page_num is not None:
#                     context = "\n".join(self.page_text_map[page_num])
#                     results['context'].append((context, page_num))
        
#         if len(self.images) > 0:
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
        
# def generate_response(self, query, context):
#     prompt = f"""Based on the following context, answer the question.
    
# Context: {context}

# Question: {query}

# Answer: """
    
#     inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
#     outputs = self.chat_model.generate(
#         inputs.input_ids,
#         max_new_tokens=512,  # Instead of max_length
#         num_return_sequences=1,
#         temperature=0.7,
#         pad_token_id=self.tokenizer.pad_token_id,
#         eos_token_id=self.tokenizer.eos_token_id
#     )
#     response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response.split("Answer: ")[-1].strip()

# def main():
#     st.set_page_config(page_title="Multimodal PDF Chat System", layout="wide")
#     st.title("Multimodal PDF Chat System")
    
#     if 'rag' not in st.session_state:
#         st.session_state.rag = MultimodalRAG()
#     if 'pdf_processed' not in st.session_state:
#         st.session_state.pdf_processed = False
#     if 'chat_history' not in st.session_state:
#         st.session_state.chat_history = []

#     with st.sidebar:
#         st.header("Upload PDF")
#         uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
#         if st.button("Reset System"):
#             st.session_state.rag = MultimodalRAG()
#             st.session_state.pdf_processed = False
#             st.session_state.chat_history = []
#             st.rerun()

#     if uploaded_file is not None and not st.session_state.pdf_processed:
#         pdf_path = "temp.pdf"
#         with open(pdf_path, "wb") as f:
#             f.write(uploaded_file.getvalue())

#         with st.spinner('Processing PDF...'):
#             st.session_state.rag.extract_content_from_pdf(pdf_path)
#             st.session_state.pdf_processed = True
        
#         if os.path.exists(pdf_path):
#             os.remove(pdf_path)
        
#         st.success(f"PDF processed successfully! Found {len(st.session_state.rag.text_chunks)} text chunks and {len(st.session_state.rag.images)} images.")

#     if st.session_state.pdf_processed:
#         st.header("Chat with your PDF")
        
#         # Display chat history
#         for message in st.session_state.chat_history:
#             with st.chat_message(message["role"]):
#                 st.write(message["content"])
#                 if "image" in message:
#                     st.image(message["image"])
        
#         # Chat input
#         query = st.chat_input("Ask a question about your PDF:")
        
#         if query:
#             # Display user message
#             with st.chat_message("user"):
#                 st.write(query)
#             st.session_state.chat_history.append({"role": "user", "content": query})
            
#             # Generate response
#             with st.spinner('Searching and generating response...'):
#                 results = st.session_state.rag.search(query)
                
#                 # Combine context from relevant text chunks
#                 context = ""
#                 if results['context']:
#                     context = "\n".join([ctx[0] for ctx in results['context']])
                
#                 response = st.session_state.rag.generate_response(query, context)
                
#                 # Display assistant response
#                 with st.chat_message("assistant"):
#                     st.write(response)
                    
#                     # Display relevant images if any
#                     if results['images']:
#                         for i, (img, loc) in enumerate(zip(results['images'], results['image_locations']), 1):
#                             st.image(img, caption=f"Related image from page {loc[0] + 1}")
                
#                 # Save response to chat history
#                 message = {
#                     "role": "assistant",
#                     "content": response
#                 }
#                 if results['images']:
#                     message["image"] = results['images'][0]
#                 st.session_state.chat_history.append(message)

#     else:
#         st.info("👆 Please upload a PDF file using the sidebar to begin.")

# if __name__ == "__main__":
#     main()








import streamlit as st
import fitz
from PIL import Image
import io
from transformers import AutoProcessor, AutoModel, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import os

class MultimodalRAG:
    def __init__(self):
        self.text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.image_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.image_model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        self.chat_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
        
        self.text_chunks = []
        self.text_embeddings = []
        self.images = []
        self.image_embeddings = []
        self.image_locations = []
        self.page_text_map = {}  # Maps page numbers to text content
        
    def chunk_text(self, text, chunk_size=500, overlap=100):
        words = text.split()
        chunks = []
        start = 0
        
        while start < len(words):
            chunk = ' '.join(words[start:start + chunk_size])
            chunks.append(chunk)
            start += chunk_size - overlap
            
        return chunks

    def process_text(self, text, page_num):
        if not text.strip():
            return
            
        if page_num not in self.page_text_map:
            self.page_text_map[page_num] = []
            
        chunks = self.chunk_text(text)
        for chunk in chunks:
            if len(chunk.strip()) > 50:
                self.text_chunks.append(chunk)
                self.page_text_map[page_num].append(chunk)
                embedding = self.text_model.encode(chunk)
                self.text_embeddings.append(embedding)
        
    def extract_content_from_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        
        progress_bar = st.progress(0)
        total_pages = len(doc)
        
        for page_num, page in enumerate(doc):
            progress_bar.progress((page_num + 1) / total_pages)
            
            blocks = page.get_text("blocks")
            for block in blocks:
                text = block[4]
                self.process_text(text, page_num)
            
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    image = Image.open(io.BytesIO(image_bytes))
                    if image.mode == 'RGBA':
                        image = image.convert('RGB')
                    
                    inputs = self.image_processor(images=image, return_tensors="pt")
                    image_features = self.image_model.get_image_features(**inputs)
                    
                    self.images.append(image)
                    self.image_embeddings.append(image_features.detach().numpy())
                    self.image_locations.append((page_num, img_index))
                except Exception as e:
                    st.warning(f"Error processing image {img_index} on page {page_num}: {str(e)}")
                    continue
        
        doc.close()
        progress_bar.empty()
        
        if self.text_embeddings:
            self.text_embeddings = np.vstack(self.text_embeddings)
    
    def search(self, query, top_k=3):
        results = {
            'text': [],
            'images': [],
            'text_locations': [],
            'image_locations': [],
            'context': []
        }
        
        if len(self.text_chunks) > 0:
            text_query_embedding = self.text_model.encode(query)
            text_similarities = cosine_similarity(
                [text_query_embedding],
                self.text_embeddings
            )[0]
            
            top_text_indices = np.argsort(text_similarities)[-top_k:][::-1]
            results['text'] = [self.text_chunks[i] for i in top_text_indices]
            results['text_locations'] = top_text_indices
            
            # Get surrounding context for each text chunk
            for idx in top_text_indices:
                page_num = None
                for p, chunks in self.page_text_map.items():
                    if self.text_chunks[idx] in chunks:
                        page_num = p
                        break
                        
                if page_num is not None:
                    context = "\n".join(self.page_text_map[page_num])
                    results['context'].append((context, page_num))
        
        if len(self.images) > 0:
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

    def generate_response(self, query, context):
        prompt = f"""Based on the following context, answer the question.
        
Context: {context}

Question: {query}

Answer: """
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        outputs = self.chat_model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Answer: ")[-1].strip()

def main():
    st.set_page_config(page_title="Multimodal PDF Chat System", layout="wide")
    st.title("Multimodal PDF Chat System")
    
    if 'rag' not in st.session_state:
        st.session_state.rag = MultimodalRAG()
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.header("Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if st.button("Reset System"):
            st.session_state.rag = MultimodalRAG()
            st.session_state.pdf_processed = False
            st.session_state.chat_history = []
            st.rerun()

    if uploaded_file is not None and not st.session_state.pdf_processed:
        pdf_path = "temp.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        with st.spinner('Processing PDF...'):
            st.session_state.rag.extract_content_from_pdf(pdf_path)
            st.session_state.pdf_processed = True
        
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        
        st.success(f"PDF processed successfully! Found {len(st.session_state.rag.text_chunks)} text chunks and {len(st.session_state.rag.images)} images.")

    if st.session_state.pdf_processed:
        st.header("Chat with your PDF")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "image" in message:
                    st.image(message["image"])
        
        # Chat input
        query = st.chat_input("Ask a question about your PDF:")
        
        if query:
            # Display user message
            with st.chat_message("user"):
                st.write(query)
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            # Generate response
            with st.spinner('Searching and generating response...'):
                results = st.session_state.rag.search(query)
                
                # Combine context from relevant text chunks
                context = ""
                if results['context']:
                    context = "\n".join([ctx[0] for ctx in results['context']])
                
                response = st.session_state.rag.generate_response(query, context)
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.write(response)
                    
                    # Display relevant images if any
                    if results['images']:
                        for i, (img, loc) in enumerate(zip(results['images'], results['image_locations']), 1):
                            st.image(img, caption=f"Related image from page {loc[0] + 1}")
                
                # Save response to chat history
                message = {
                    "role": "assistant",
                    "content": response
                }
                if results['images']:
                    message["image"] = results['images'][0]
                st.session_state.chat_history.append(message)

    else:
        st.info("👆 Please upload a PDF file using the sidebar to begin.")

if __name__ == "__main__":
    main()




    