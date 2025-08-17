# RAG CrewAI Multi-Agent System

A sophisticated multi-agent system built with CrewAI that can answer questions using either PDF documents or web search, with built-in quality control through grading and hallucination detection.

## 🏗️ Project Structure

```
crewai-multi-agent-rag/
├── .gitignore           # Git ignore rules
├── requirements.txt     # Python dependencies
├── README.md            # This documentation
└── agentic_ai/          # Main package
    ├── __init__.py      # Package initialization
    ├── config.py        # LLM configuration and environment setup
    ├── tools.py         # Custom tools (PDF search, web search, router)
    ├── agents.py        # Agent definitions and creation
    ├── tasks.py         # Task definitions and workflow
    ├── crew.py          # Crew orchestration and execution
    └── main.py          # Main entry point (both modular and legacy)
```

## 🚀 Features

- **Intelligent Routing**: Automatically routes questions to PDF search or web search
- **PDF Analysis**: Searches through uploaded PDF documents using RAG
- **Web Search**: Uses Tavily search for current information
- **Quality Control**: Multi-stage grading and hallucination detection
- **Modular Design**: Clean separation of concerns for maintainability

## 🔧 Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/crewai-multi-agent-rag.git
   cd crewai-multi-agent-rag
   ```

2. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Environment Variables**:
   ```bash
   export GROQ_API_KEY="your_groq_api_key_here"
   export TAVILY_API_KEY="your_tavily_api_key_here"  # Optional for web search
   ```

## 📖 Usage

### Modular Approach (Recommended)

```python
from agentic_ai.config import setup_llm
from agentic_ai.tools import setup_pdf
from agentic_ai.crew import create_rag_crew, run_rag_query

# Setup
llm = setup_llm()
rag_tool = setup_pdf()
crew = create_rag_crew(llm, rag_tool)

# Ask questions
result = run_rag_query(crew, "What are the main concepts discussed in this document?")
print(result)
```

### Direct Execution

```bash
cd /path/to/your/project
python agentic_ai/main.py
```

## 🤖 Agent Workflow

1. **Router Agent**: Analyzes the question and decides between PDF search or web search
2. **Retriever Agent**: Executes the appropriate search and retrieves information
3. **Grader Agent**: Evaluates if the retrieved content is relevant to the question
4. **Hallucination Grader**: Checks if the answer is grounded in facts
5. **Final Answer Provider**: Provides the comprehensive final answer

## 🛠️ Components

### Tools (`tools.py`)
- `web_search_tool`: Formatted web search using Tavily
- `router_tool`: Intelligent routing logic
- `setup_pdf()`: PDF download and RAG tool setup

### Agents (`agents.py`)
- Router Agent: Question routing
- Retriever Agent: Information retrieval
- Grader Agent: Relevance assessment
- Hallucination Grader: Fact verification
- Final Answer Provider: Answer synthesis

### Tasks (`tasks.py`)
- Router Task: Route questions to appropriate search method
- Retriever Task: Execute search and retrieve information
- Grader Task: Assess relevance of retrieved content
- Hallucination Task: Verify factual grounding
- Answer Task: Provide final comprehensive answer

### Configuration (`config.py`)
- LLM setup with Groq API
- Environment variable management

## 🔍 Example Questions

**PDF Questions** (searches through uploaded PDF documents):
- "What are the main concepts discussed in this document?"
- "Explain the key findings from the research paper"
- "What methodology was used in this study?"

**Web Search Questions**:
- "What are the latest developments in AI?"
- "Current trends in machine learning?"
- "Recent breakthroughs in natural language processing?"

## 🎯 Response Formatting

The web search tool formats results as:
```
Result 1:
URL: https://example.com
Content: [content snippet]

Result 2:
URL: https://example2.com
Content: [content snippet]
```

This structured format helps agents better understand and process the search results.

## 🔄 Migration from Legacy Code

The original monolithic code is preserved in `main.py` as commented reference. The new modular structure provides:

- **Better Maintainability**: Separated concerns
- **Easier Testing**: Individual components can be tested
- **Reusability**: Components can be reused in other projects
- **Scalability**: Easy to add new agents, tasks, or tools

## 🐛 Troubleshooting

1. **API Key Issues**: Ensure `GROQ_API_KEY` is set in your environment
2. **Import Errors**: Make sure you're running from the correct directory
3. **Tool Recognition**: The custom `web_search_tool` should be recognized properly
4. **PDF Processing**: Ensure your PDF documents are accessible and readable

## 📝 Notes

- The system maintains the exact same behavior as the original monolithic version
- All original functionality is preserved while improving code organization
- The modular structure makes it easier to extend and maintain the system
