from crewai_tools import PDFSearchTool, tool
from langchain_community.tools.tavily_search import TavilySearchResults
import requests
import os

# Download and setup PDF
def setup_pdf():
    """Download the Attention is All You Need paper"""
    pdf_url = "https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf"
    response = requests.get(pdf_url)
    
    with open("attention_is_all_you_need.pdf", "wb") as file:
        file.write(response.content)
    
    return PDFSearchTool(pdf='attention_is_all_you_need.pdf',
        config=dict(
            llm=dict(
                provider="groq",
                config=dict(
                    model="llama3-8b-8192",
                ),
            ),
            embedder=dict(
                provider="huggingface",
                config=dict(
                    model="BAAI/bge-small-en-v1.5",
                ),
            ),
        )
    )

# Create a proper web search tool with a predictable name
@tool
def web_search_tool(query: str) -> str:
    """Search the web for information using Tavily search engine."""
    tavily_search = TavilySearchResults(k=3)
    results = tavily_search.run(query)
    
    # Format the results into a readable string
    if isinstance(results, list):
        formatted_results = []
        for i, result in enumerate(results, 1):
            if isinstance(result, dict):
                url = result.get('url', 'No URL')
                content = result.get('content', 'No content')
                formatted_results.append(f"Result {i}:\nURL: {url}\nContent: {content}\n")
        return "\n".join(formatted_results)
    else:
        return str(results)

@tool
def router_tool(question):
    """Router Function"""
    if 'self-attention' in question:
        return 'vectorstore'
    else:
        return 'web_search'
