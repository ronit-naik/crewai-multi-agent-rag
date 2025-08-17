from crewai import Crew
from .agents import create_agents
from .tasks import create_tasks

def create_rag_crew(llm, rag_tool):
    """Create the RAG crew with all agents and tasks"""
    
    # Create agents
    agents = create_agents(llm, rag_tool)
    
    # Create tasks
    tasks = create_tasks(agents)
    
    # Create crew
    rag_crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        verbose=True,
    )
    
    return rag_crew

def run_rag_query(crew, question):
    """Run a query through the RAG crew"""
    inputs = {"question": question}
    result = crew.kickoff(inputs=inputs)
    return result
