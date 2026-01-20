from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping

from pydantic import BaseModel, Field
from pydantic_ai import RunContext, Tool
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

GraphNodeCallable = Callable[
    [GraphRunContext["AdHocGraphState", "AdHocGraphDeps"], dict[str, Any]],
    Awaitable["GraphNodeResult"] | "GraphNodeResult",
]


@dataclass
class GraphNodeResult:
    """Result returned by an ad-hoc graph node function."""

    next_node: str | None = None
    output: Any | None = None
    updates: dict[str, Any] = field(default_factory=dict)
    note: str | None = None


@dataclass
class AdHocGraphState:
    """Mutable state shared between ad-hoc graph nodes."""

    data: dict[str, Any] = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class AdHocGraphDeps:
    tool_ctx: RunContext[Any]
    node_functions: Mapping[str, GraphNodeCallable]
    node_classes: Mapping[str, type[BaseNode]]
    purpose: str


class AdHocGraphNodeSpec(BaseModel):
    """Defines a node in the ad-hoc execution graph."""

    name: str = Field(..., description="Human-friendly name for the node.")
    function: str = Field(..., description="Name of the registered node function to call.")
    args: dict[str, Any] = Field(default_factory=dict, description="Arguments for the node function.")
    description: str | None = Field(default=None, description="Optional explanation of the node's job.")


class AdHocGraphSpec(BaseModel):
    """Input schema for the ad-hoc graph tool."""

    purpose: str = Field(..., description="Overall goal for the graph.")
    start_node: str = Field(..., description="Node name to start execution from.")
    nodes: list[AdHocGraphNodeSpec] = Field(..., description="Nodes to include in the graph.")
    initial_state: dict[str, Any] = Field(
        default_factory=dict,
        description="Initial shared state for the graph.",
    )


class AdHocGraphOutput(BaseModel):
    """Output of the ad-hoc graph tool."""

    purpose: str
    result: Any | None
    final_state: dict[str, Any]
    trace: list[str]
    notes: list[str]


def build_adhoc_graph_tool(node_functions: Mapping[str, GraphNodeCallable]) -> Tool:
    """
    Create a pydantic-ai tool that runs ad-hoc execution graphs using pydantic-graph.

    The node functions are defined by the agent author and can invoke an LLM, perform
    deterministic calculations, or mix both (e.g., search for data then compute results).
    Each node function receives a `GraphRunContext`, a reference to the tool context
    via `ctx.deps.tool_ctx`, and a dict of args from the graph spec.
    """

    async def run_adhoc_graph(ctx: RunContext[Any], spec: AdHocGraphSpec) -> AdHocGraphOutput:
        node_classes: dict[str, type[BaseNode]] = {}
        class_names: set[str] = set()

        for index, node_spec in enumerate(spec.nodes):
            if node_spec.function not in node_functions:
                raise ValueError(f"Unknown node function '{node_spec.function}'.")

            class_name = _make_class_name(node_spec.name, index, class_names)
            node_classes[node_spec.name] = _build_node_class(node_spec, class_name)

        if spec.start_node not in node_classes:
            raise ValueError(f"Start node '{spec.start_node}' is not defined.")

        deps = AdHocGraphDeps(
            tool_ctx=ctx,
            node_functions=node_functions,
            node_classes=node_classes,
            purpose=spec.purpose,
        )
        state = AdHocGraphState(data=dict(spec.initial_state))
        graph = Graph(nodes=tuple(node_classes.values()), name=spec.purpose)
        start_node = node_classes[spec.start_node]()
        result = await graph.run(start_node, state=state, deps=deps)
        return result.output

    return Tool(run_adhoc_graph)


def _make_class_name(name: str, index: int, existing: set[str]) -> str:
    base = re.sub(r"\W|^(?=\d)", "_", name).strip("_") or "Node"
    class_name = f"{base}_{index}"
    while class_name in existing:
        class_name = f"{base}_{index + len(existing)}"
    existing.add(class_name)
    return class_name


def _build_node_class(node_spec: AdHocGraphNodeSpec, class_name: str) -> type[BaseNode]:
    async def run(
        self, ctx: GraphRunContext[AdHocGraphState, AdHocGraphDeps]
    ) -> BaseNode[AdHocGraphState, AdHocGraphDeps, AdHocGraphOutput] | End[AdHocGraphOutput]:
        node_fn = ctx.deps.node_functions[node_spec.function]
        ctx.state.trace.append(node_spec.name)
        result = node_fn(ctx, dict(node_spec.args))
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, GraphNodeResult):
            raise TypeError(
                f"Node function '{node_spec.function}' must return GraphNodeResult, got {type(result)}."
            )
        if result.updates:
            ctx.state.data.update(result.updates)
        if result.note:
            ctx.state.notes.append(result.note)
        if result.next_node is None:
            output = AdHocGraphOutput(
                purpose=ctx.deps.purpose,
                result=result.output,
                final_state=dict(ctx.state.data),
                trace=list(ctx.state.trace),
                notes=list(ctx.state.notes),
            )
            return End(output)
        try:
            next_class = ctx.deps.node_classes[result.next_node]
        except KeyError as exc:
            raise ValueError(f"Unknown next node '{result.next_node}'.") from exc
        return next_class()

    run.__annotations__ = {
        "ctx": GraphRunContext[AdHocGraphState, AdHocGraphDeps],
        "return": BaseNode[AdHocGraphState, AdHocGraphDeps, AdHocGraphOutput]
        | End[AdHocGraphOutput],
    }

    return type(class_name, (BaseNode,), {"run": run})
