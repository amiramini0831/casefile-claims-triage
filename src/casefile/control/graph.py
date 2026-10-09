"""Run a case as a LangGraph graph, so its state is checkpointed after each step.

LangGraph adds persistence here and nothing else. Every routing and stopping
decision is still made by ``casefile.control.supervisor``: each graph node
calls ``step`` for its agent, and the edge after it follows ``next_node``. A
run through this graph therefore ends in exactly the state ``run_case``
reaches, which the tests check for every evaluation case.

With a checkpointer, the state is saved once a node has finished and before
the next one starts. If the process dies inside a node, nothing of that node
was saved, and invoking the graph again for the same run continues from the
last finished step.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from casefile.control.state import CaseState, Limits, Node
from casefile.control.supervisor import Agent, start, step

NODES: tuple[Node, ...] = ("extractor", "investigator", "reviewer")
# Supersteps LangGraph may take beyond our own step limit before it gives up.
# Never reached in practice: the supervisor stops the run first.
RECURSION_MARGIN = 5
# Write each checkpoint before the next step starts. LangGraph's default
# ("async") writes it in the background while the next step runs, so a process
# killed during that step can lose the step before it. The kill test in
# tests/test_checkpoints.py fails with the default.
DURABILITY = "sync"


class GraphState(TypedDict):
    """What LangGraph stores: the case state as plain JSON-compatible data."""

    case: dict[str, Any]


def to_graph_state(state: CaseState) -> GraphState:
    """Serialize a case state for LangGraph."""
    return {"case": state.model_dump(mode="json")}


def from_graph_state(graph_state: Mapping[str, Any]) -> CaseState:
    """Restore a case state from what LangGraph stored."""
    return CaseState.model_validate(graph_state["case"])


def build_graph(
    agents: Mapping[Node, Agent], checkpointer: BaseCheckpointSaver | None = None
) -> CompiledStateGraph:
    """Build the graph for one set of agents."""

    def node_for(name: Node):
        def run_node(graph_state: GraphState) -> GraphState:
            return to_graph_state(step(from_graph_state(graph_state), agents[name]))

        return run_node

    def route(graph_state: GraphState) -> str:
        return from_graph_state(graph_state).next_node or END

    graph = StateGraph(GraphState)
    for name in NODES:
        graph.add_node(name, node_for(name))
    targets = {name: name for name in NODES} | {END: END}
    graph.add_conditional_edges(START, route, targets)
    for name in NODES:
        graph.add_conditional_edges(name, route, targets)
    return graph.compile(checkpointer=checkpointer)


def run_config(run_id: str, limits: Limits) -> dict[str, Any]:
    """Return the LangGraph config of a run; ``run_id`` names its checkpoints."""
    return {
        "configurable": {"thread_id": run_id},
        "recursion_limit": limits.max_steps + RECURSION_MARGIN,
    }


def run_graph(
    graph: CompiledStateGraph, *, run_id: str, case_id: str, limits: Limits
) -> CaseState:
    """Run a case to its end, or continue it if ``run_id`` was started before.

    Calling this again for a run that already ended returns its final state
    without running any agent.
    """
    config = run_config(run_id, limits)
    saved = saved_state(graph, run_id, limits)
    if saved is None:
        graph_input = to_graph_state(start(case_id, limits))
    elif saved.status != "running":
        return saved
    else:
        # ``None`` tells LangGraph to continue from the last checkpoint.
        graph_input = None
    # Durability only means something with a checkpointer, and LangGraph
    # rejects the option without one.
    options = {"durability": DURABILITY} if graph.checkpointer is not None else {}
    return from_graph_state(graph.invoke(graph_input, config, **options))


def saved_state(
    graph: CompiledStateGraph, run_id: str, limits: Limits
) -> CaseState | None:
    """Return the last checkpointed state of a run, or ``None`` if there is none."""
    if graph.checkpointer is None:
        return None
    values = graph.get_state(run_config(run_id, limits)).values
    return from_graph_state(values) if values else None
