import sys
import subprocess
import os

# Fix for SQLite3 version issue on Streamlit Cloud
try:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
    
import base64
import streamlit as st
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from retriever import Retriever
from generator import ResponseGenerator
from web_search import WebSearch

def display_image(image_data):
    """Displays an image from base64 data with a caption."""
    try:
        img_bytes = base64.b64decode(image_data['image_data'])
        caption = image_data.get('label') or image_data.get('description', 'Image')
        st.image(img_bytes, caption=caption, use_container_width=True)
        details = image_data.get('details')
        if details:
            st.markdown(f"**Details:** {details}")
    except Exception as e:
        st.error(f"Error displaying image: {e}")

def main():
    load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
    
    google_api_key = os.environ.get('GOOGLE_API_KEY')
    tavily_api_key = os.environ.get('TAVILY_API_KEY')

    st.set_page_config(page_title="Maintenance Manual Chatbot", layout="wide")
    st.title("⚙️ Maintenance Manual Chatbot")

    if not google_api_key:
        st.error("Error: GOOGLE_API_KEY not found. Please add it to your .env file.")
        return

    st.sidebar.title("Options")
    enable_web_search = st.sidebar.checkbox("Enable Web Search", value=False)
    
    if enable_web_search and not tavily_api_key:
        st.sidebar.error("TAVILY_API_KEY not found in .env file. Please add it to enable web search.")
        enable_web_search = False

    processor = DocumentProcessor()
    retriever = Retriever()
    generator = ResponseGenerator(api_key=google_api_key)
    web_search_tool = WebSearch() if enable_web_search and tavily_api_key else None

    if "data_loaded" not in st.session_state:
        with st.spinner("First-time setup: Processing all documents, please wait..."):
            st.session_state.all_documents, st.session_state.all_images_data = processor.load_documents()
            if not st.session_state.all_documents and not st.session_state.all_images_data:
                st.warning("No documents found in 'data' folder. Please add PDF or Excel files.")
                return
            
            chunks = processor.chunk_documents(st.session_state.all_documents)
            retriever.create_vector_store(chunks, st.session_state.all_images_data)
            st.session_state.data_loaded = True
            st.success("All documents processed and ready!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "images" in message:
                for img in message["images"]:
                    display_image(img)

    if prompt := st.chat_input("Ask about a product or maintenance procedure..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents, the web, and formulating responses..."):
                context = retriever.retrieve_relevant_docs(prompt)
                relevant_images = retriever.get_relevant_images(prompt, context)

                web_context = None
                if web_search_tool:
                    with st.spinner("Performing web search..."):
                        web_context = web_search_tool.search(prompt)

                response_data = generator.generate_response(prompt, context, relevant_images, web_context=web_context)
                contextual_response = response_data["contextual"]
                direct_response = response_data["direct"]

                st.markdown("### 📝 Answer from Your Documents & Web")
                st.markdown(contextual_response)
                st.markdown("---")
                st.markdown("### 💡 General Answer from Gemini")
                st.markdown(direct_response)

                if relevant_images:
                    st.write("**Relevant Image(s) from Documents:**")
                    for img in relevant_images:
                        display_image(img)
                
                # Combine for chat history
                full_response_for_history = (
                    f"### 📝 Answer from Your Documents & Web\n{contextual_response}\n\n"
                    f"---\n\n"
                    f"### 💡 General Answer from Gemini\n{direct_response}"
                )
                
                assistant_message = {"role": "assistant", "content": full_response_for_history}
                if relevant_images:
                    assistant_message["images"] = relevant_images
                st.session_state.messages.append(assistant_message)

if __name__ == "__main__":
    main()
