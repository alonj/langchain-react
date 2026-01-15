from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from tavily import TavilyClient

try:
    from e2b_code_interpreter import CodeInterpreter
except ImportError as exc:  # pragma: no cover - helpful error when deps missing
    raise SystemExit(
        "Missing dependency 'e2b-code-interpreter'. Install it to use the E2B tool."
    ) from exc


@dataclass
class AgentDeps:
    tavily: TavilyClient
    code_interpreter: CodeInterpreter
    sources: list[str] = field(default_factory=list)


class AgentResponse(BaseModel):
    answer: str = Field(..., description="Final answer to the user question.")
    sources: list[str] = Field(default_factory=list, description="Source URLs used.")


agent = Agent(
    model=os.getenv("OPENAI_MODEL", "openai:gpt-4o-mini"),
    result_type=AgentResponse,
    system_prompt=(
        "You are a ReAct-style assistant. Use tools when helpful, and provide a concise "
        "final answer. Always include relevant source URLs when you use the web search tool."
    ),
)


@agent.tool
async def tavily_search(ctx: RunContext[AgentDeps], query: str) -> list[dict[str, Any]]:
    """Search the web with Tavily."""
    results = ctx.deps.tavily.search(query=query, max_results=5)
    sources = [result.get("url") for result in results if result.get("url")]
    ctx.deps.sources.extend(sources)
    return results


@agent.tool
async def run_code(ctx: RunContext[AgentDeps], code: str) -> dict[str, Any]:
    """Execute Python code in an E2B sandbox."""
    execution = ctx.deps.code_interpreter.notebook.exec_cell(code)
    return {
        "stdout": getattr(execution, "stdout", ""),
        "stderr": getattr(execution, "stderr", ""),
        "result": getattr(execution, "result", None),
    }


def build_deps() -> AgentDeps:
    return AgentDeps(
        tavily=TavilyClient(api_key=os.getenv("TAVILY_API_KEY")),
        code_interpreter=CodeInterpreter(api_key=os.getenv("E2B_API_KEY")),
    )


def run_agent(prompt: str) -> dict[str, Any]:
    deps = build_deps()
    result = agent.run_sync(prompt, deps=deps)
    payload = result.data.model_copy()
    payload.sources = list(dict.fromkeys(deps.sources))
    return payload.model_dump()


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
