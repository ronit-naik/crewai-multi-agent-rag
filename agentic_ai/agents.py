from crewai import Agent
from .tools import web_search_tool, router_tool

def create_agents(llm, rag_tool):
    """Create all agents for the RAG crew"""
    
    router_agent = Agent(
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
    
    retriever_agent = Agent(
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

    grader_agent = Agent(
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
    
    return {
        'router': router_agent,
        'retriever': retriever_agent,
        'grader': grader_agent,
        'hallucination_grader': hallucination_grader,
        'answer_grader': answer_grader
    }
