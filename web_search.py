import os
from tavily import TavilyClient

class WebSearch:
    """
    A class to handle web searches using the Tavily API.
    """
    def __init__(self):
        """
        Initializes the TavilyClient with the API key from environment variables.
        """
        self.api_key = os.environ.get('TAVILY_API_KEY')
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY not found in environment variables.")
        self.client = TavilyClient(api_key=self.api_key)

    def search(self, query: str, max_results: int = 3) -> str:
        """
        Performs a web search for the given query.

        Args:
            query (str): The search query.
            max_results (int): The maximum number of search results to return.

        Returns:
            str: A formatted string of the search results, or an error message.
        """
        try:
            response = self.client.search(query=query, max_results=max_results)
            
            # Check if 'results' key exists and is not empty
            if 'results' in response and response['results']:
                # Format the results into a string
                results_str = "\n\n".join(
                    [f"**Source:** {res['url']}\n**Content:** {res['content']}" for res in response['results']]
                )
                return results_str
            else:
                return "No web results found for the query."
                
        except Exception as e:
            print(f"An error occurred during web search: {e}")
            return "There was an error performing the web search."