import copy
import logging
import os
import time

from .utils import save_pickle, load_pickle

logger = logging.getLogger(__name__)

CHECKPOINT_FILENAME = "training_state.pkl"
CHECKPOINT_VERSION = 1

_AGENT_FIELDS = ["success_rate", "total_task_num", "successful_task_num",
                 "task_correlations", "decay_count_down"]
_GRAPH_FIELDS = ["edge_weight", "edge_success_rate", "agent_neighbor_dict"]


def _uses_global_router_pool(experiment):
    return bool(experiment.agent_graph_config.get("global_router_experience", False))


def save_training_state(experiment, path, meta=None):
    # No copies needed: the pickle is written immediately from live state.
    graph = experiment.agent_graph
    agents = {}
    for agent_id, agent in graph.agents.items():
        state = {field: getattr(agent, field) for field in _AGENT_FIELDS}
        state["abilities"] = dict(agent.abilities)
        state["llm"] = agent.llm
        state["executor_pool"] = agent.executor_module.executor_pool.__dict__
        if not _uses_global_router_pool(experiment):
            state["router_pool"] = agent.router_module.router_pool.__dict__
        agents[agent_id] = state

    global_router_pool = None
    if _uses_global_router_pool(experiment):
        from .global_vars import global_router_experience_pool
        global_router_pool = global_router_experience_pool.__dict__

    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "meta": dict(meta or {}, saved_at=time.strftime("%Y-%m-%d %H:%M:%S")),
        "graph": {field: getattr(graph, field) for field in _GRAPH_FIELDS},
        "agents": agents,
        "global_router_pool": global_router_pool,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    save_pickle(checkpoint, path)
    return path


def read_training_state(path):
    if os.path.isdir(path):
        path = os.path.join(path, CHECKPOINT_FILENAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No training checkpoint at {path}")
    checkpoint = load_pickle(path)
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"{path}: unsupported checkpoint version {checkpoint.get('version')!r}")
    checkpoint["path"] = path
    return checkpoint


def apply_training_state(experiment, checkpoint):
    graph = experiment.agent_graph
    saved_agents = checkpoint["agents"]
    path = checkpoint.get("path", "<checkpoint>")

    if set(saved_agents) != set(graph.agents):
        raise ValueError(f"{path}: checkpoint has agents {sorted(saved_agents)}, "
                         f"this run has {sorted(graph.agents)}")
    for agent_id, agent in graph.agents.items():
        saved_dims, dims = set(saved_agents[agent_id]["abilities"]), set(agent.abilities)
        if saved_dims != dims:
            raise ValueError(f"{path}: agent {agent_id} ability dimensions differ "
                             f"(checkpoint {sorted(saved_dims)}, run {sorted(dims)})")
    if (checkpoint["global_router_pool"] is not None) != _uses_global_router_pool(experiment):
        raise ValueError(f"{path}: global_router_experience setting differs from training")

    changed_llms = {agent_id: (saved_agents[agent_id]["llm"], agent.llm)
                    for agent_id, agent in graph.agents.items()
                    if saved_agents[agent_id]["llm"] != agent.llm}
    if changed_llms:
        logger.warning(f"Loaded training state with different agent LLMs "
                       f"(agent: trained -> now): {changed_llms}")

    for agent_id, agent in graph.agents.items():
        saved = copy.deepcopy(saved_agents[agent_id])
        # In place: manual_backend._agent_abilities holds these same dicts.
        agent.abilities.clear()
        agent.abilities.update(saved["abilities"])
        for field in _AGENT_FIELDS:
            setattr(agent, field, saved[field])
        agent.executor_module.executor_pool.__dict__.update(saved["executor_pool"])
        if "router_pool" in saved:
            agent.router_module.router_pool.__dict__.update(saved["router_pool"])

    if checkpoint["global_router_pool"] is not None:
        from .global_vars import global_router_experience_pool
        global_router_experience_pool.__dict__.update(copy.deepcopy(checkpoint["global_router_pool"]))

    for field in _GRAPH_FIELDS:
        setattr(graph, field, copy.deepcopy(checkpoint["graph"][field]))

    return checkpoint["meta"]
