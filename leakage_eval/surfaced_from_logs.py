import argparse
import json
import os
import re
import sys

import numpy as np

from common import Visits, calls, expand, load_checkpoint

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for src.pool

THRESHOLD = 0.7  # the project's retrieval threshold
PRIVATE = "\n\nPrivate information (only you know this):\n"
# the prompt layouts of the router: (major task, problem / description, completed steps)
LAYOUTS = [
    (re.compile(r"\*\*Major Task\*\*: (.*?)\n\n\*\*Your Expertise", re.S),
     re.compile(r"- Problem: (.*?)\n- Type:", re.S),
     re.compile(r"- Completed Steps: (.*?)\n\n\*\*Available Specialists", re.S)),
    (re.compile(r"<Major Task>\n(.*?)\n</Major Task/>", re.S),
     re.compile(r"- Problem: (.*?)\n</Current Task State/>", re.S),
     re.compile(r"<Completed Subtasks and Results>\n(.*?)\n</Completed Subtasks and Results/>", re.S)),
]


def retrieval_query(prompt):
    for major_re, problem_re, steps_re in LAYOUTS:
        major, problem, steps = major_re.search(prompt), problem_re.search(prompt), steps_re.search(prompt)
        if major and problem and steps:
            major = major[1].split(PRIVATE)[0]
            progress = "" if steps[1] == "None progress yet." else steps[1]
            description = "" if problem[1] == major else problem[1]
            return major + progress + description
    return None


def surfaced_tasks(agent_state, query_embedding, k):
    scores = []
    for experiences in agent_state["router_pool"]["retrieval_experiences"].values():
        for exp in experiences:
            if not exp.success:
                continue
            sim = np.dot(query_embedding, exp.embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(exp.embedding))
            scores.append((-sim, exp))
    ordered = sorted(scores, key=lambda x: x[0])  # most similar first, ties in storage order (as the project)
    found = [exp for s, exp in ordered if -s >= THRESHOLD][:k]
    texts = []
    for exp in found:  # a task stored several times fills several of the k slots but is listed once
        text = " ".join(exp.task_major_problem.split())
        if text not in texts:
            texts.append(text)
    return texts


def visited_files(folder):
    path = os.path.join(folder, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        tasks = json.load(f).get("per_task", [])
    return {t["file"]: t["agents_visited"] > t.get("non_manual_visits", 0)
            for t in tasks if t.get("mode") == "Test" and t.get("completed")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="run folder or task file")
    ap.add_argument("checkpoint", help="training_state.pkl or the save dir holding it")
    ap.add_argument("-k", type=int, required=True, help="how many relevant tasks are surfaced per router call")
    ap.add_argument("--multihop", action="store_true", help="keep every visit after the attacker, not only each agent's first (c2)")
    args = ap.parse_args()

    from src.pool import encode_model  # BGE, the encoder the project routes with

    state = load_checkpoint(args.checkpoint)
    visited_cache, skipped = {}, 0
    for path in expand([args.log]):
        folder = os.path.dirname(os.path.abspath(path))
        if folder not in visited_cache:
            visited_cache[folder] = visited_files(folder)
        visited = visited_cache[folder]
        if visited is not None and not visited.get(os.path.basename(path), False):
            continue  # the manual agent was not visited in this task
        task_id = os.path.splitext(os.path.basename(path))[0]
        visits = Visits()
        for call, agent, role, stage, query, response in calls(path):
            visits.step(agent, role)
            if stage != "router" or "benign" not in role or not visits.exposed(agent, args.multihop):
                continue
            text = retrieval_query(query)
            if text is None:
                skipped += 1
                continue
            found = surfaced_tasks(state["agents"][agent], np.asarray(encode_model.encode(text)), args.k)
            print(f"({task_id}, {agent}): {' || '.join(found)}")
    if skipped:
        print(f"# {skipped} router calls skipped: their prompt layout was not recognised", file=sys.stderr)


if __name__ == "__main__":
    main()
