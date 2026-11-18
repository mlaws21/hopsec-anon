import collections
import glob
import json
import os
import pickle
import re

CALL_RE = re.compile(r"^\[call (\d+)\] Agent (\d+) \(([^)]*)\) \| (\w+) \|", re.M)
LOG_AGENT = re.compile(r"\bAgent\s+(\d+)\s+(?:FORWARD|EXECUTE|SPLIT)\b")


def expand(paths):
    for p in paths:
        if os.path.isdir(p):
            yield from sorted(glob.glob(os.path.join(p, "task*.txt")),
                              key=lambda x: tuple(int(n) for n in re.findall(r"\d+", os.path.basename(x))))
        else:
            yield p


def task_id(path):
    name = os.path.basename(path)
    return os.path.splitext(name)[0] if name.startswith("task") else "?"


class Visits:
    def __init__(self):
        self.prev, self.attacked, self.visit, self.first_visit = None, False, 0, {}

    def step(self, agent, role):
        if agent != self.prev:  # a new visit starts
            self.visit, self.prev = self.visit + 1, agent
            if "manual" in role:
                self.attacked = True
            elif self.attacked:
                self.first_visit.setdefault(agent, self.visit)

    def exposed(self, agent, multihop=False):
        return self.attacked if multihop else self.first_visit.get(agent) == self.visit


def scan_replies(path, key, multihop=False):
    task = task_id(path)
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    structured = bool(CALL_RE.search(text))
    region, agent, role, visits = None, None, "", Visits()
    for line in text.split("\n"):
        m = CALL_RE.match(line)
        if m:
            agent, role, region = int(m[2]), m[3], "header"
            visits.step(agent, role)
            continue
        if line.startswith("--- SYSTEM PROMPT ---") or line.startswith("--- QUERY PROMPT ---"):
            region = "prompt"
            continue
        if line.startswith("--- RESPONSE ---"):
            region = "response"
            continue
        if not key.search(line):
            continue
        if structured and (region != "response" or "manual" in role or not visits.exposed(agent, multihop)):
            continue
        who = LOG_AGENT.search(line)
        who = int(who[1]) if who else (agent if structured else None)
        marks = list(key.finditer(line))
        for i, mk in enumerate(marks):
            yield task, who, line[mk.end():marks[i + 1].start() if i + 1 < len(marks) else len(line)]


def calls(path):
    text = open(path, encoding="utf-8").read()
    marks = list(CALL_RE.finditer(text))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[m.end():end]
        query, _, response = body.split("--- QUERY PROMPT ---", 1)[-1].partition("--- RESPONSE ---")
        yield int(m[1]), int(m[2]), m[3], m[4], query, response


def exposed_pairs(summary_path, multihop=False):
    with open(summary_path, encoding="utf-8") as f:
        tasks = json.load(f)["per_task"]
    pairs = set()
    for t in tasks:
        task = os.path.splitext(t["file"])[0]
        manual = {int(a) for a in t.get("manual_agents", {})}
        path = [int(a.rstrip("*")) for a in t["agent_path"]]
        first = next((i for i, a in enumerate(path) if a in manual), None)
        if first is None:
            continue
        after = [a for a in path[first + 1:] if a not in manual]
        if after:
            pairs |= {(task, a) for a in after} if multihop else {(task, after[0])}
    return pairs


def natural(task):
    return tuple(int(x) for x in re.findall(r"\d+", task)), task


def shown(text, width):
    return text if not width or len(text) <= width else text[:width] + "..."


FOLD = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "..."})


def norm(text, ignore_case, fold=False):
    text = " ".join(text.split())
    if fold:
        text = text.translate(FOLD)
    return text.lower() if ignore_case else text


def rouge_l(a, b):
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return 2 * prev[-1] / (len(a) + len(b))


def best_rouge(tokens, candidates):
    mine = collections.Counter(tokens)
    bounds = sorted(((2 * sum((mine & collections.Counter(c)).values()) / (len(tokens) + len(c)) if tokens and c else 0.0, i)
                     for i, c in enumerate(candidates)), reverse=True)
    best = (0.0, None)
    for bound, i in bounds:
        if bound <= best[0]:
            break
        score = rouge_l(tokens, candidates[i])
        if score > best[0]:
            best = (score, i)
    return best


def match_sets(records, references, n):
    refs, recs = {}, {}
    for t in references:
        refs.setdefault(n(t), t)
    for t in records:
        recs.setdefault(n(t), t)
    order = list(refs)
    token_lists = [k.split() for k in order]
    best_by_ref, matched = {}, set()
    for key in recs:
        score, i = best_rouge(key.split(), token_lists)
        if i is not None:
            matched.add(key)
            best_by_ref[order[i]] = max(best_by_ref.get(order[i], 0.0), score)
    return sum(best_by_ref.values()), matched, len(refs)


def prefix_note(text, refs, n, what):
    best = None
    for entry in refs:
        x, y = n(text), n(entry)
        if y.startswith(x) and (best is None or len(y) < best[1]):
            best = (len(x.split()), len(y), len(y.split()))
    return f"  <- beginning of a {what} ({best[0]} of {best[2]} words)" if best else ""

class Stub:
    def __setstate__(self, state):
        if isinstance(state, tuple):  # (dict, slots) form
            state = {k: v for part in state if part for k, v in part.items()}
        self.__dict__.update(state)


_stub_classes = {}


class StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "src" or module.startswith("src.") or module == "task":
            key = (module, name)
            if key not in _stub_classes:
                _stub_classes[key] = type(name, (Stub,), {"__module__": module})
            return _stub_classes[key]
        return super().find_class(module, name)


def load_checkpoint(path):
    if os.path.isdir(path):
        path = os.path.join(path, "training_state.pkl")
    if not os.path.isfile(path):
        raise SystemExit(f"No training checkpoint at {path}")
    with open(path, "rb") as f:
        return StubUnpickler(f).load()


def router_entries(agent_state):
    seen = {}
    for bucket in ("success_experiences", "failure_experiences", "retrieval_experiences"):
        for experiences in agent_state["router_pool"].get(bucket, {}).values():
            for exp in experiences:
                text = " ".join(exp.task_major_problem.split())
                seen.setdefault(text.lower(), text)
    return list(seen.values())
