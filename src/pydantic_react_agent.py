from __future__ import annotations

import json
import os
import sys
import asyncio
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from tavily import TavilyClient
from e2b_code_interpreter import Sandbox

import argparse
import yaml



parser = argparse.ArgumentParser(description="Run the Pydantic React Agent.")
parser.add_argument("prompt", type=str, help="The prompt/question to ask the agent.")
parser.add_argument("--model", type=str, default="openai:gpt-5-mini", help="The LLM model to use.")
parser.add_argument("--output", type=str, default="output.json", help="File to write output JSON to.")
parser.add_argument("--type", type=str, choices=["plan", "answer", "plan_and_answer"], default="plan_and_answer", help="QA agent type.")
args = parser.parse_args()

cwd = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(cwd, "prompts.yaml"), "r") as f:
    prompts = yaml.safe_load(f)

@dataclass
class AgentDeps:
    tavily: TavilyClient
    e2b: type[Sandbox]
    sources: list[str] = field(default_factory=list)


class AgentResponse(BaseModel):
    answer: str = Field(..., description=prompts[args.type]["response_prompt"])
    sources: list[str] = Field(default_factory=list, description="Source URLs used.")


agent = Agent(
    model=args.model,
    output_type=AgentResponse,
    system_prompt=(prompts[args.type]["system_prompt"]),
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
    with ctx.deps.e2b.create(api_key=os.getenv("E2B_API_KEY")) as sandbox:
        execution = sandbox.run_code(code)
    return {
        "stdout": getattr(execution, "stdout", ""),
        "stderr": getattr(execution, "stderr", ""),
        "result": getattr(execution, "text", None),
    }


def build_deps() -> AgentDeps:
    return AgentDeps(
        tavily=TavilyClient(api_key=os.getenv("TAVILY_API_KEY")),
        e2b=Sandbox,
    )


async def run_agent(prompt: str) -> dict[str, Any]:
    deps = build_deps()
    nodes = []
    async with agent.iter(prompt, deps=deps) as agent_run:
        async for node in agent_run:
            nodes.append(node)
    output = agent_run.result.output
    answer = output.answer
    sources = list(set(output.sources))
    if args.type == "plan":
        return {
            "question": prompt,
            "plan": answer,
            "trace": [str(node) for node in nodes],
        }
    else:
        return {
            "question": prompt,
            "answer": answer,
            "sources": sources,
            "trace": [str(node) for node in nodes],
        }


async def main() -> None:
    output = await run_agent(args.prompt)

    output_json = json.dumps(output, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
    else:
        print(output_json)


if __name__ == "__main__":
    asyncio.run(main())