import os

from langgraph.graph import END, StateGraph

from .critic_agent import run_critic_agent
from .eda_agent import run_eda_agent
from .evaluation_agent import run_evaluation_agent
from .experiment_agent import run_experiment_agent
from .feature_agent import run_feature_agent
from .planner_agent import run_planner_agent
from .report_agent import run_report_agent
from .state import AgentState


def _should_continue_after_eda(state: AgentState) -> str:
    if state.get("status") == "failed":
        return "report"
    return "planner"


def _should_continue_after_planner(state: AgentState) -> str:
    if state.get("status") == "failed":
        return "report"
    return "features"


def _should_continue_after_features(state: AgentState) -> str:
    if state.get("status") == "failed":
        return "report"
    return "train"


def _should_continue_after_train(state: AgentState) -> str:
    if state.get("status") == "failed":
        return "report"
    return "critic"


def _route_after_critic(state: AgentState) -> str:
    if state.get("status") == "failed":
        return "report"
    history = state.get("critic_history") or []
    verdict = history[-1]["verdict"] if history else "approve"
    if verdict == "retry_features":
        return "features"
    if verdict == "retry_models":
        return "train"
    return "evaluation"


def build_graph(checkpointer):
    graph = StateGraph(AgentState)

    graph.add_node("eda", run_eda_agent)
    graph.add_node("planner", run_planner_agent)
    graph.add_node("features", run_feature_agent)
    graph.add_node("train", run_experiment_agent)
    graph.add_node("critic", run_critic_agent)
    graph.add_node("evaluation", run_evaluation_agent)
    graph.add_node("report", run_report_agent)

    graph.set_entry_point("eda")

    graph.add_conditional_edges("eda", _should_continue_after_eda,
                                 {"planner": "planner", "report": "report"})
    graph.add_conditional_edges("planner", _should_continue_after_planner,
                                 {"features": "features", "report": "report"})
    graph.add_conditional_edges("features", _should_continue_after_features,
                                 {"train": "train", "report": "report"})
    graph.add_conditional_edges("train", _should_continue_after_train,
                                 {"critic": "critic", "report": "report"})
    graph.add_conditional_edges("critic", _route_after_critic,
                                 {"features": "features", "train": "train",
                                  "evaluation": "evaluation", "report": "report"})
    graph.add_edge("evaluation", "report")
    graph.add_edge("report", END)

    return graph.compile(checkpointer=checkpointer)


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        os.makedirs("./data", exist_ok=True)
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            checkpointer = SqliteSaver.from_conn_string("./data/checkpoints.db")
        except Exception:
            try:
                import sqlite3

                from langgraph.checkpoint.sqlite import SqliteSaver as _SS
                conn = sqlite3.connect("./data/checkpoints.db", check_same_thread=False)
                checkpointer = _SS(conn)
            except Exception:
                checkpointer = None
        _graph = build_graph(checkpointer)
    return _graph
