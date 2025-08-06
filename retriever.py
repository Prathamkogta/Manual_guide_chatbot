import os
import pickle
import tempfile
import streamlit as st
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class Retriever:
    def __init__(self):
        self.vector_store = None
        self.images_data = []
        # Use Streamlit's cache directory or temp directory
        self.persist_dir = os.path.join(tempfile.gettempdir(), "chroma_data")
        # Ensure the directory exists
        os.makedirs(self.persist_dir, exist_ok=True)

    def create_vector_store(self, chunks, images_data=None):
        """Creates a vector store for text and saves the image data list."""
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", 
                google_api_key=os.environ.get('GOOGLE_API_KEY')
            )
            
            # Use the temp directory for persistence
            self.vector_store = Chroma.from_documents(
                chunks, 
                embeddings, 
                persist_directory=self.persist_dir
            )
            
            if images_data:
                self.images_data = images_data
                self._save_images_data()
                
        except Exception as e:
            st.error(f"Error creating vector store: {e}")
            # Fallback to in-memory store
            self._create_in_memory_vector_store(chunks, images_data)

    def _create_in_memory_vector_store(self, chunks, images_data=None):
        """Creates an in-memory vector store as fallback."""
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", 
                google_api_key=os.environ.get('GOOGLE_API_KEY')
            )
            
            # Create in-memory vector store (no persistence)
            self.vector_store = Chroma.from_documents(chunks, embeddings)
            
            if images_data:
                self.images_data = images_data
                
            st.warning("Using in-memory vector store. Data won't persist between sessions.")
            
        except Exception as e:
            st.error(f"Failed to create vector store: {e}")
            self.vector_store = None

    def _save_images_data(self):
        """Saves the combined list of image data to a pickle file."""
        try:
            images_file = os.path.join(self.persist_dir, "images_data.pkl")
            with open(images_file, "wb") as f:
                pickle.dump(self.images_data, f)
        except Exception as e:
            st.warning(f"Could not save images data: {e}")

    def _load_images_data(self):
        """Loads the combined list of image data from a pickle file."""
        try:
            images_file = os.path.join(self.persist_dir, "images_data.pkl")
            with open(images_file, "rb") as f:
                self.images_data = pickle.load(f)
        except FileNotFoundError:
            self.images_data = []
        except Exception as e:
            st.warning(f"Could not load images data: {e}")
            self.images_data = []

    def _load_vector_store(self):
        """Loads the vector store and image data from disk if they haven't been loaded yet."""
        if not self.vector_store:
            try:
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001", 
                    google_api_key=os.environ.get('GOOGLE_API_KEY')
                )
                
                # Try to load existing vector store
                if os.path.exists(self.persist_dir):
                    self.vector_store = Chroma(
                        persist_directory=self.persist_dir, 
                        embedding_function=embeddings
                    )
                    self._load_images_data()
                else:
                    st.warning("No existing vector store found. Please process documents first.")
                    
            except Exception as e:
                st.error(f"Error loading vector store: {e}")
                self.vector_store = None
            
    def retrieve_relevant_docs(self, query, k=5):
        """Retrieves relevant text documents from the vector store based on similarity."""
        if not self.vector_store:
            self._load_vector_store()
            
        if not self.vector_store:
            return []
            
        try:
            return self.vector_store.similarity_search_with_score(query, k=k)
        except Exception as e:
            st.error(f"Error retrieving documents: {e}")
            return []

    def get_relevant_images(self, query, text_context, max_images=1):
        """Finds the best-matching image from all sources using a scoring system."""
        if not self.vector_store:
            self._load_vector_store()
            
        query_lower = query.lower().split()
        
        # Use pages from the text context to boost image scores
        relevant_pages = set()
        if text_context:
            for doc, score in text_context:
                relevant_pages.add((doc.metadata.get('source'), doc.metadata.get('page')))

        def score_image(img):
            score = 0
            description = (img.get('description', '') or '').lower()
            label = (img.get('label', '') or '').lower()
            product_name = (img.get('product_name', '') or '').lower()
            
            for keyword in query_lower:
                if keyword in description: score += 2
                if keyword in label: score += 5
                if keyword in product_name: score += 10 # Higher score for matching product name
            
            # Boost score if the image is on a page relevant to the text context (for PDFs)
            if (img.get('source'), img.get('page')) in relevant_pages:
                score += 10
            
            # Boost score for images that come from Excel, as they are specific products
            if 'row_index' in img:
                score += 5
            
            return score

        scored_images = [(img, score_image(img)) for img in self.images_data]
        # Filter out images with a zero score
        scored_images = [si for si in scored_images if si[1] > 0]
        # Sort by score in descending order
        scored_images.sort(key=lambda x: x[1], reverse=True)
        
        # Return the top N images
        return [img for img, _ in scored_images[:max_images]]

    def get_excel_doc_by_image(self, all_documents, image):
        """Finds the text document that corresponds to a specific Excel image using its row index."""
        if 'row_index' not in image:
            return None
        
        img_row_index = image.get('row_index')
        img_source = image.get('source')

        for doc in all_documents:
            meta = doc.metadata
            if meta.get('source') == img_source and meta.get('row_index') == img_row_index:
                return doc
        return None
