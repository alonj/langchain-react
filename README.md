# langchain-react
ReACT agents implemented with LangChain and Pydantic AI.

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Set API keys:

```bash
export OPENAI_API_KEY=your-key
export TAVILY_API_KEY=your-key
export E2B_API_KEY=your-key
```

Run the LangChain agent:

```bash
python src/react_agent.py "What is the capital of France?"
```

Run the Pydantic AI agent with web search (Tavily) and code execution (E2B):

```bash
python src/pydantic_react_agent.py "What is the population of Japan?"
```

Both agents return a JSON payload containing the answer and the list of source URLs.
