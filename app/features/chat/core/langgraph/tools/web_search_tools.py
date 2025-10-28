import os
from langchain_tavily import TavilySearch


os.environ["TAVILY_API_KEY"] 


# Initialize Tavily Search Tool
tavily_search_tool = TavilySearch(
    max_results=5,
    topic="general",
)