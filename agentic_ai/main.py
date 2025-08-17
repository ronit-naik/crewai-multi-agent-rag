"""
Main entry point for the RAG CrewAI system.
This file demonstrates how to use both the original monolithic approach and the new modular approach.
"""

# NEW MODULAR APPROACH (RECOMMENDED)
from .config import setup_llm
from .tools import setup_pdf
from .crew import create_rag_crew, run_rag_query

def main_modular():
    """Main function using the new modular structure"""
    
    # Setup LLM
    llm = setup_llm()
    
    # Setup PDF tool
    rag_tool = setup_pdf()
    
    # Create RAG crew
    rag_crew = create_rag_crew(llm, rag_tool)
    
    # Example queries
    questions = [
        "Tell me about self-attention mechanism in Transformers?",  # PDF question
        "Tell me about LLMs using web_search?",  # Web search question
    ]
    
    for question in questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")
        
        try:
            result = run_rag_query(rag_crew, question)
            print(f"Answer: {result}")
        except Exception as e:
            print(f"Error: {e}")

# ORIGINAL MONOLITHIC APPROACH (COMMENTED OUT FOR REFERENCE)
"""
from langchain_openai import ChatOpenAI
from crewai_tools import PDFSearchTool
from langchain_community.tools.tavily_search import TavilySearchResults
from crewai_tools import tool
from crewai import Crew
from crewai import Task
from crewai import Agent
import requests
import os

groq_api_key = os.environ.get("GROQ_API_KEY")
tavily_api_key = os.environ.get("TAVILY_API_KEY")

# Check if API key exists
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please set it in your environment.")

llm = ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=groq_api_key,  # Use the variable instead of direct access
    model_name="llama3-8b-8192",
    temperature=0.1,
    max_tokens=1000,
)

pdf_url = "https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf"
response = requests.get(pdf_url)

with open("attention_is_all_you_need.pdf", "wb") as file:
    file.write(response.content)
rag_tool = PDFSearchTool(pdf='attention_is_all_you_need.pdf',
    config=dict(
        llm=dict(
            provider="groq", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="llama3-8b-8192",
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="huggingface", # or openai, ollama, ...
            config=dict(
                model="BAAI/bge-small-en-v1.5",
                #task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )
)

# Create a proper web search tool with a predictable name
@tool
def web_search_tool(query: str) -> str:
    \"\"\"Search the web for information using Tavily search engine.\"\"\"
    tavily_search = TavilySearchResults(k=3)
    results = tavily_search.run(query)
    
    # Format the results into a readable string
    if isinstance(results, list):
        formatted_results = []
        for i, result in enumerate(results, 1):
            if isinstance(result, dict):
                url = result.get('url', 'No URL')
                content = result.get('content', 'No content')
                formatted_results.append(f"Result {i}:\\nURL: {url}\\nContent: {content}\\n")
        return "\\n".join(formatted_results)
    else:
        return str(results)

@tool
def router_tool(question):
  \"\"\"Router Function\"\"\"
  if 'self-attention' in question:
    return 'vectorstore'
  else:
    return 'web_search'
  

Router_Agent = Agent(
  role='Router',
  goal='Route user question to a vectorstore or web search',
  backstory=(
    "You are an expert at routing a user question to a vectorstore or web search."
    "Use the vectorstore for questions on concept related to Retrieval-Augmented Generation."
    "You do not need to be stringent with the keywords in the question related to these topics. Otherwise, use web-search."
  ),
  verbose=True,
  allow_delegation=False,
  llm=llm,
)
  
Retriever_Agent = Agent(
role="Retriever",
goal="Use the information retrieved from the vectorstore to answer the question",
backstory=(
    "You are an assistant for question-answering tasks."
    "Use the information present in the retrieved context to answer the question."
    "You have to provide a clear concise answer."
),
verbose=True,
allow_delegation=False,
llm=llm,
tools=[rag_tool, web_search_tool],
)

Grader_agent =  Agent(
  role='Answer Grader',
  goal='Filter out erroneous retrievals',
  backstory=(
    "You are a grader assessing relevance of a retrieved document to a user question."
    "If the document contains keywords related to the user question, grade it as relevant."
    "It does not need to be a stringent test.You have to make sure that the answer is relevant to the question."
  ),
  verbose=True,
  allow_delegation=False,
  llm=llm,
)
hallucination_grader = Agent(
    role="Hallucination Grader",
    goal="Filter out hallucination",
    backstory=(
        "You are a hallucination grader assessing whether an answer is grounded in / supported by a set of facts."
        "Make sure you meticulously review the answer and check if the response provided is in alignmnet with the question asked"
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)
answer_grader = Agent(
    role="Final Answer Provider",
    goal="Provide the final comprehensive answer based on retrieved information.",
    backstory=(
        "You are responsible for providing the final answer to the user's question."
        "You analyze the retrieved information and provide a comprehensive answer."
        "You work only with the information provided by previous agents and do not use any external tools."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

router_task = Task(
    description=("Analyse the keywords in the question {question}"
    "Based on the keywords decide whether it is eligible for a vectorstore search or a web search."
    "Return a single word 'vectorstore' if it is eligible for vectorstore search."
    "Return a single word 'websearch' if it is eligible for web search."
    "Do not provide any other premable or explaination."
    ),
    expected_output=("Give a binary choice 'websearch' or 'vectorstore' based on the question"
    "Do not provide any other premable or explaination."),
    agent=Router_Agent,
    tools=[router_tool],
)

retriever_task = Task(
    description=("Extract information for the question: {question}. "
    "The router has determined the search method. "
    "If the router output is 'websearch', use web_search_tool to search the web. "
    "If the router output is 'vectorstore', use rag_tool to search the PDF document. "
    "Provide detailed information based on the search results."
    ),
    expected_output=("Detailed information about the question based on the appropriate search method. "
    "Use the search results to provide a comprehensive answer."),
    agent=Retriever_Agent,
    context=[router_task],
)

grader_task = Task(
    description=("Based on the response from the retriever task for the quetion {question} evaluate whether the retrieved content is relevant to the question."
    ),
    expected_output=("Binary score 'yes' or 'no' score to indicate whether the document is relevant to the question"
    "You must answer 'yes' if the response from the 'retriever_task' is in alignment with the question asked."
    "You must answer 'no' if the response from the 'retriever_task' is not in alignment with the question asked."
    "Do not provide any preamble or explanations except for 'yes' or 'no'."),
    agent=Grader_agent,
    context=[retriever_task],
)
hallucination_task = Task(
    description=("Based on the response from the grader task for the quetion {question} evaluate whether the answer is grounded in / supported by a set of facts."),
    expected_output=("Binary score 'yes' or 'no' score to indicate whether the answer is sync with the question asked"
    "Respond 'yes' if the answer is in useful and contains fact about the question asked."
    "Respond 'no' if the answer is not useful and does not contains fact about the question asked."
    "Do not provide any preamble or explanations except for 'yes' or 'no'."),
    agent=hallucination_grader,
    context=[grader_task],
)
answer_task = Task(
    description=("Provide the final comprehensive answer to: {question}. "
    "Use the information from the retriever task to answer the question. "
    "If the hallucination grader says 'yes', provide a detailed answer based on the retrieved information. "
    "If the hallucination grader says 'no', acknowledge that the information may be insufficient but still provide the best possible answer based on available information."
    ),
    expected_output=("A comprehensive and well-structured answer to the user's question based on the retrieved information. "
    "Provide detailed explanations and context from the retrieved content."),
    context=[retriever_task, hallucination_task],
    agent=answer_grader,
)

rag_crew = Crew(
    agents=[Router_Agent, Retriever_Agent, Grader_agent, hallucination_grader, answer_grader],
    tasks=[router_task, retriever_task, grader_task, hallucination_task, answer_task],
    verbose=True,
)

# Example usage of the original approach
# inputs = {"question": "Tell me about LLMs using web_search?"}
inputs = {"question": "Tell me about self-attention mechanism in Transformers?"}
result = rag_crew.kickoff(inputs=inputs)
print(result)
"""

if __name__ == "__main__":
    # Use the new modular approach
    main_modular()
    
    # To use the original approach, uncomment the code above and comment out main_modular()
