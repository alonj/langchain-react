from __future__ import annotations

import json
import os
from pydoc import text
import sys
import asyncio
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai import ModelRequest, ModelSettings
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.direct import model_request

from tavily import TavilyClient
from e2b_code_interpreter import Sandbox

import argparse
import yaml

parser = argparse.ArgumentParser(description="Run the Pydantic React Agent.")
parser.add_argument("prompt", type=str, help="The prompt/question to ask the agent.")
parser.add_argument("--model", type=str, default="openrouter:gpt-5-mini", help="The LLM model to use.")
parser.add_argument("--output", type=str, default="output.json", help="File to write output JSON to.")
parser.add_argument("--type", type=str, choices=["plan", "answer", "plan_and_answer"], default="plan_and_answer", help="QA agent type.")
args = parser.parse_args()

cwd = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(cwd, "agents.yaml"), "r") as f:
    prompts = yaml.safe_load(f)

@dataclass
class AgentDeps:
    tavily: TavilyClient
    e2b: type[Sandbox]
    sources: list[str] = field(default_factory=list)


class AgentResponse(BaseModel):
    answer: str = Field(..., description=prompts[args.type]["response_prompt"])
    sources: list[str] = Field(default_factory=list, description="Source URLs used.")


async def tavily_search(ctx: RunContext[AgentDeps], query: str) -> list[dict[str, Any]]:
    """Search the web with Tavily."""
    oai = OpenAIResponsesModel(model_name='gpt-4.1-nano', settings=ModelSettings(max_tokens=500))#, extra_body={'reasoning': {'effort': 'minimal'}}))
    prompter = lambda text, max_length: f"Summarize the following text in {max_length} characters. Maintain important and salient data. We're interested in information regarding the following query: \"{query}\"\n\nText to summarize:\n{text}"
    async def summarize_fun(text: str, max_length: int) -> str:
        prompt = prompter(text, max_length)
        model_response = await model_request(oai, [ModelRequest.user_text_prompt(prompt)], model_settings=ModelSettings(max_tokens=max_length))
        return model_response.parts[0].content
    
    results = ctx.deps.tavily.search(query=query, max_results=5, include_answer=False, include_raw_content='markdown')
    results = results["results"]
    sources = [result.get("url") for result in results if result.get("url")]
    for result in results:
        text = result.get("raw_content", "")
        if not text:
            continue
        summary = await summarize_fun(text, max_length=500)
        result["content"] = summary
        del result["raw_content"]
        del result["content"]
    ctx.deps.sources.extend(sources)
    return results


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


async def run_agent(agent: Agent, prompt: str) -> dict[str, Any]:
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
    tools_dict = {
        "tavily_search": tavily_search,
        "run_code": run_code,
    }
    tools_config = prompts[args.type].get("tools", [])
    selected_tools = [Tool(tools_dict[tool_name]) for tool_name in tools_config if tool_name in tools_dict]
    agent = Agent(
        model=args.model,
        output_type=AgentResponse,
        system_prompt=(prompts[args.type]["system_prompt"]),
        deps_type=AgentDeps,
        tools=selected_tools,
    )

    output = await run_agent(agent, args.prompt)

    output_json = json.dumps(output, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
    else:
        print(output_json)


if __name__ == "__main__":
    asyncio.run(main())