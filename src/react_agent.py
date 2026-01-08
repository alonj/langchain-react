from __future__ import annotations

import json
import os
import sys
from collections import OrderedDict
from typing import Any, Iterable, List

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI

PROMPT_TEMPLATE = """You are a helpful assistant that can use tools to answer questions.

You have access to the following tools:
{tools}

Use the following format:

Question: {input}
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Begin!

Question: {input}
{agent_scratchpad}
"""


def _extract_urls(observation: Any) -> Iterable[str]:
    if isinstance(observation, str):
        try:
            observation = json.loads(observation)
        except json.JSONDecodeError:
            return []
    if isinstance(observation, dict):
        url = observation.get("url")
        return [url] if url else []
    if isinstance(observation, list):
        urls: List[str] = []
        for item in observation:
            if isinstance(item, dict) and item.get("url"):
                urls.append(item["url"])
        return urls
    return []


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    return list(OrderedDict.fromkeys(item for item in items if item))


def build_agent() -> AgentExecutor:
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    tools = [TavilySearchResults(k=5)]
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, return_intermediate_steps=True)


def run_agent(prompt: str) -> dict[str, Any]:
    agent_executor = build_agent()
    result = agent_executor.invoke({"input": prompt})
    sources: List[str] = []
    for _action, observation in result.get("intermediate_steps", []):
        sources.extend(_extract_urls(observation))
    return {
        "answer": result.get("output", ""),
        "sources": _dedupe_preserve_order(sources),
    }


def main() -> None:
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = sys.stdin.read().strip()
    if not prompt:
        raise SystemExit("Provide a prompt via argv or stdin.")
    output = run_agent(prompt)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
