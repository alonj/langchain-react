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

Run the Pydantic AI agent with web search (Tavily) and code execution (E2B):
Example:
```bash
python src/pydantic_react_agent.py "What was the average population of Japan between 1972 and 1978?" --model gpt-5-mini --output test.json --type plan
```

### Command options

- `prompt` (positional): question passed to the agent.
- `--model`: model identifier understood by the configured provider (default: `openai:gpt-5-mini`).
- `--type`: agent behavior. Use `plan`, `answer`, or `plan_and_answer` (default: `plan_and_answer`).
- `--output`: target JSON file for the run (default: `output.json`). Set `--output ""` to print to stdout.

### Output format

The script writes a JSON document containing the original question, the generated plan and/or answer, any collected source URLs, and the intermediate trace of tool calls.
