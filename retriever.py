import os
import pickle
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class Retriever:
    def __init__(self):
        self.vector_store = None
        self.images_data = []
        # --- CHANGE START ---
        # Added a set of common stop words to ignore during search
        self.stop_words = set([
            "a", "about", "an", "are", "as", "at", "be", "by", "for", "from",
            "how", "in", "is", "it", "of", "on", "or", "that", "the", "this",
            "to", "was", "what", "when", "where", "who", "will", "with", "the",
            "tell", "me", "what's", "whats", "what is"
        ])
        # --- CHANGE END ---

    def create_vector_store(self, chunks, images_data=None):
        """Creates a vector store for text and saves the image data list."""
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.environ.get('GOOGLE_API_KEY'))
        self.vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_data")
        
        if images_data:
            self.images_data = images_data
            self._save_images_data()

    def _save_images_data(self):
        """Saves the combined list of image data to a pickle file."""
        with open("chroma_data/images_data.pkl", "wb") as f:
            pickle.dump(self.images_data, f)

    def _load_images_data(self):
        """Loads the combined list of image data from a pickle file."""
        try:
            with open("chroma_data/images_data.pkl", "rb") as f:
                self.images_data = pickle.load(f)
        except FileNotFoundError:
            self.images_data = []

    def _load_vector_store(self):
        """Loads the vector store and image data from disk if they haven't been loaded yet."""
        if not self.vector_store:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.environ.get('GOOGLE_API_KEY'))
            self.vector_store = Chroma(persist_directory="chroma_data", embedding_function=embeddings)
            self._load_images_data()
            
    def retrieve_relevant_docs(self, query, k=5):
        """Retrieves relevant text documents from the vector store based on similarity."""
        self._load_vector_store()
        return self.vector_store.similarity_search_with_score(query, k=k)

    def get_relevant_images(self, query, text_context, max_images=1):
        """
        Finds the best-matching image using advanced scoring with stop-word
        filtering and phrase matching.
        """
        self._load_vector_store()
        if not self.images_data:
            return []

        # --- CHANGE START ---
        # The new logic is more robust.
        MIN_RELEVANCE_SCORE = 10
        
        # Clean the query by removing stop words
        query_lower = query.lower()
        query_keywords = [word for word in query_lower.split() if word not in self.stop_words]
        clean_query_phrase = " ".join(query_keywords)

        if not query_keywords:
            return [] # Return empty if the query only contained stop words
        # --- CHANGE END ---
        
        relevant_pages = set()
        if text_context:
            for doc, score in text_context:
                relevant_pages.add((doc.metadata.get('source'), doc.metadata.get('page')))

        def score_image(img):
            score = 0
            description = (img.get('description', '') or '').lower()
            label = (img.get('label', '') or '').lower()
            product_name = (img.get('product_name', '') or '').lower()
            
            # --- CHANGE START: New Scoring Logic ---
            # 1. Big bonus for exact phrase match (highest priority)
            if clean_query_phrase in product_name or clean_query_phrase in label:
                score += 50

            # 2. Score based on number of matching keywords
            matched_keywords = 0
            for keyword in query_keywords:
                if keyword in description:
                    score += 2
                    matched_keywords += 1
                if keyword in label:
                    score += 5
                    matched_keywords += 1
                if keyword in product_name:
                    score += 10
                    matched_keywords += 1
            
            # 3. Bonus for matching multiple keywords
            if matched_keywords > 1:
                score += matched_keywords * 5
            # --- CHANGE END ---

            if (img.get('source'), img.get('page')) in relevant_pages:
                score += 10
            
            if 'row_index' in img:
                score += 5
            
            return score

        scored_images = [(img, score_image(img)) for img in self.images_data]
        
        scored_images = [si for si in scored_images if si[1] >= MIN_RELEVANCE_SCORE]

        scored_images.sort(key=lambda x: x[1], reverse=True)
        
        return [img for img, _ in scored_images[:max_images]]

    def get_excel_doc_by_image(self, all_documents, image):
        if 'row_index' not in image:
            return None
        
        img_row_index = image.get('row_index')
        img_source = image.get('source')

        for doc in all_documents:
            meta = doc.metadata
            if meta.get('source') == img_source and meta.get('row_index') == img_row_index:
                return doc
        return None
