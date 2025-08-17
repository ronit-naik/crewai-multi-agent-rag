from crewai import Task
from .tools import router_tool

def create_tasks(agents):
    """Create all tasks for the RAG crew"""
    
    router_task = Task(
        description=(
            "Analyse the keywords in the question {question}"
            "Based on the keywords decide whether it is eligible for a vectorstore search or a web search."
            "Return a single word 'vectorstore' if it is eligible for vectorstore search."
            "Return a single word 'websearch' if it is eligible for web search."
            "Do not provide any other premable or explaination."
        ),
        expected_output=(
            "Give a binary choice 'websearch' or 'vectorstore' based on the question"
            "Do not provide any other premable or explaination."
        ),
        agent=agents['router'],
        tools=[router_tool],
    )

    retriever_task = Task(
        description=(
            "Extract information for the question: {question}. "
            "The router has determined the search method. "
            "If the router output is 'websearch', use web_search_tool to search the web. "
            "If the router output is 'vectorstore', use rag_tool to search the PDF document. "
            "Provide detailed information based on the search results."
        ),
        expected_output=(
            "Detailed information about the question based on the appropriate search method. "
            "Use the search results to provide a comprehensive answer."
        ),
        agent=agents['retriever'],
        context=[router_task],
    )

    grader_task = Task(
        description=(
            "Based on the response from the retriever task for the quetion {question} evaluate whether the retrieved content is relevant to the question."
        ),
        expected_output=(
            "Binary score 'yes' or 'no' score to indicate whether the document is relevant to the question"
            "You must answer 'yes' if the response from the 'retriever_task' is in alignment with the question asked."
            "You must answer 'no' if the response from the 'retriever_task' is not in alignment with the question asked."
            "Do not provide any preamble or explanations except for 'yes' or 'no'."
        ),
        agent=agents['grader'],
        context=[retriever_task],
    )
    
    hallucination_task = Task(
        description=(
            "Based on the response from the grader task for the quetion {question} evaluate whether the answer is grounded in / supported by a set of facts."
        ),
        expected_output=(
            "Binary score 'yes' or 'no' score to indicate whether the answer is sync with the question asked"
            "Respond 'yes' if the answer is in useful and contains fact about the question asked."
            "Respond 'no' if the answer is not useful and does not contains fact about the question asked."
            "Do not provide any preamble or explanations except for 'yes' or 'no'."
        ),
        agent=agents['hallucination_grader'],
        context=[grader_task],
    )
    
    answer_task = Task(
        description=(
            "Provide the final comprehensive answer to: {question}. "
            "Use the information from the retriever task to answer the question. "
            "If the hallucination grader says 'yes', provide a detailed answer based on the retrieved information. "
            "If the hallucination grader says 'no', acknowledge that the information may be insufficient but still provide the best possible answer based on available information."
        ),
        expected_output=(
            "A comprehensive and well-structured answer to the user's question based on the retrieved information. "
            "Provide detailed explanations and context from the retrieved content."
        ),
        context=[retriever_task, hallucination_task],
        agent=agents['answer_grader'],
    )
    
    return [router_task, retriever_task, grader_task, hallucination_task, answer_task]
