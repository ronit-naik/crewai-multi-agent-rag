from langchain_openai import ChatOpenAI
import os

def setup_llm():
    """Setup and configure the LLM"""
    groq_api_key = os.environ.get("GROQ_API_KEY")
    
    # Check if API key exists
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set. Please set it in your environment.")

    llm = ChatOpenAI(
        openai_api_base="https://api.groq.com/openai/v1",
        openai_api_key=groq_api_key,
        model_name="llama3-8b-8192",
        temperature=0.1,
        max_tokens=1000,
    )
    
    return llm

def get_environment_variables():
    """Get all required environment variables"""
    return {
        'groq_api_key': os.environ.get("GROQ_API_KEY"),
        'tavily_api_key': os.environ.get("TAVILY_API_KEY"),
    }
