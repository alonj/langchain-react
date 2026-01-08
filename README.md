# langchain-react
ReACT agent implemented with langchain

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Set API keys:

```bash
export OPENAI_API_KEY=your-key
export TAVILY_API_KEY=your-key
```

Run the agent:

```bash
python src/react_agent.py "What is the capital of France?"
```

The output is a JSON payload containing the answer and the list of source URLs.
