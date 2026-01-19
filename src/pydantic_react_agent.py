from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from tavily import TavilyClient

import argparse

try:
    from e2b_code_interpreter import Sandbox
except ImportError as exc:  # pragma: no cover - helpful error when deps missing
    raise SystemExit(
        "Missing dependency 'e2b-code-interpreter'. Install it to use the E2B tool."
    ) from exc


parser = argparse.ArgumentParser(description="Run the Pydantic React Agent.")
parser.add_argument("prompt", type=str, help="The prompt/question to ask the agent.")
parser.add_argument("--model", type=str, default="openai:gpt-5-mini", help="The LLM model to use.")
parser.add_argument("--output", type=str, default="output.json", help="File to write output JSON to.")
args = parser.parse_args()

@dataclass
class AgentDeps:
    tavily: TavilyClient
    sources: list[str] = field(default_factory=list)


class AgentResponse(BaseModel):
    answer: str = Field(..., description="Final answer to the user question.")
    sources: list[str] = Field(default_factory=list, description="Source URLs used.")


agent = Agent(
    model=args.model,
    output_type=AgentResponse,
    system_prompt=(
        "You are a ReAct-style assistant. Use tools when helpful, and provide a concise "
        "final answer. Always include relevant source URLs when you use the web search tool."
    ),
)


@agent.tool
async def tavily_search(ctx: RunContext[AgentDeps], query: str) -> list[dict[str, Any]]:
    """Search the web with Tavily."""
    results = ctx.deps.tavily.search(query=query, max_results=5)
    results = results["results"]
    sources = [result.get("url") for result in results if result.get("url")]
    ctx.deps.sources.extend(sources)
    return results


@agent.tool
async def run_code(ctx: RunContext[AgentDeps], code: str) -> dict[str, Any]:
    """Execute Python code in an E2B sandbox."""
    with Sandbox.create() as sandbox:
        execution = sandbox.run_code(code)
    return {
        "stdout": getattr(execution, "stdout", ""),
        "stderr": getattr(execution, "stderr", ""),
        "result": getattr(execution, "text", None),
    }


def build_deps() -> AgentDeps:
    return AgentDeps(
        tavily=TavilyClient(api_key=os.getenv("TAVILY_API_KEY")),
    )


def run_agent(prompt: str) -> dict[str, Any]:
    deps = build_deps()
    result = agent.run_sync(prompt, deps=deps)
    output = result.output
    answer = output.answer
    sources = list(set(output.sources))
    return {
        "answer": answer,
        "sources": sources,
    }


def main() -> None:
    output = run_agent(args.prompt)
    output_json = json.dumps(output, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
