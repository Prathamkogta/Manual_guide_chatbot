import os
import google.generativeai as genai
import pandas as pd

class ResponseGenerator:
    def __init__(self, api_key, model="gemini-1.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def generate_direct_response(self, query):
            """Generates a standard, non-contextual response from the model."""
            # This new prompt asks for a much more detailed answer
            prompt = f"""
            You are a helpful and knowledgeable AI assistant. Your task is to provide a comprehensive and detailed answer to the user's question.
            - Elaborate on the main topic of the question.
            - Explain key concepts clearly and thoroughly.
            - Use examples or analogies if they help with the explanation.
            - Structure your answer with bullet points for better readability.
            - Do not refer to any external documents, just use your general knowledge.

            **User's Question:**
            {query}

            **Your Detailed Answer:**
            """
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                print(f"Error during direct response generation: {e}")
                return "I encountered an error while generating a standard response."

    def generate_response(self, query, context, images=None, product_info=None, web_context=None):
        """
        Generates a dictionary containing a contextual response (from documents and web)
        and a direct response (standard model answer).
        """
        # 1. Generate the contextual response
        context_text = self._build_context_text(context)
        image_context = self._build_image_context(images)
        product_context = self._build_product_context(product_info)
        web_search_context = self._build_web_context(web_context)

        contextual_prompt = f"""
        You are an expert technical assistant. Your task is to answer the user's question based ONLY on the provided context.
        Synthesize information from document excerpts, image descriptions, structured product data, and web search results.

        **Context from Documents:**
        {context_text}

        **Context from Relevant Images:**
        {image_context}
        
        **Structured Product Data from Database:**
        {product_context}

        **Context from Web Search:**
        {web_search_context}

        **Instructions:**
        1.  Carefully analyze the user's question and all provided context.
        2.  Formulate a comprehensive answer.
        3.  **Primary Answer from Documents:** First, provide a detailed answer using *only* the information from the "Context from Documents". Structure it with headings and bullet points for clarity. If the documents contain the answer, this should be the main part of your response.
        4.  **Web-Enhanced Information:** If the "Context from Web Search" is available and relevant, add a separate section at the end titled "Additional Information from the Web". Summarize the key points from the web context here.
        5.  **Handling No Information:**
            - If the documents do not contain the answer, state: "I could not find specific information on this in the provided documents."
            - If web search was enabled but yielded no relevant results, you can either omit the web section or state that no relevant information was found online.
            - If neither source provides an answer, state that you could not find information on the topic.
        6.  Refer to images if they are relevant to the answer.
        7.  Do not use any prior knowledge or information from outside the provided context.

        **User's Question:**
        {query}

        **Your Answer:**
        """
        
        contextual_response = "Error: Could not generate contextual response."
        try:
            response = self.model.generate_content(contextual_prompt)
            contextual_response = response.text
        except Exception as e:
            print(f"Error during contextual response generation: {e}")
            contextual_response = "I encountered an error while processing your request with the provided documents."

        # 2. Generate the direct, general response
        direct_response = self.generate_direct_response(query)

        # 3. Return both responses in a dictionary
        return {
            "contextual": contextual_response,
            "direct": direct_response
        }

    def _build_context_text(self, context):
        if not context: return "No relevant text context was found."
        context_parts = [
            f"--- START OF EXCERPT (Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}) ---\n{doc.page_content}\n--- END OF EXCERPT ---"
            for doc, score in context
        ]
        return "\n\n".join(context_parts)

    def _build_image_context(self, images):
        if not images: return "No relevant images were found."
        image_parts = []
        for i, image in enumerate(images):
            image_parts.append(f"""
            Image {i+1} Context (from {image.get('source', 'N/A')}, page {image.get('page', 'N/A')}):
            - Product Mentioned: {image.get('product_name') or 'N/A'}
            - Description: {image.get('description') or 'No description available.'}
            """)
        return "\n".join(image_parts)

    def _build_product_context(self, product_info):
        if not product_info:
            return "No specific product data was found in the database for this query."
        parts = [f"- **{key.replace('_', ' ').title()}:** {value}" for key, value in product_info.items() if value and pd.notna(value)]
        return "\n".join(parts) if parts else "Product data found, but it is empty."
    
    def _build_web_context(self, web_context):
        if not web_context:
            return "Web search was not enabled or no results were found."
        return web_context