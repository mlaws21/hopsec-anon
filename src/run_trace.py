import functools
import json
import logging
import os
import re
import sys
import threading
import time

from . import timing

UNATTRIBUTED = "unattributed"

# (attribute name on src.agent, stage label). Looked up at call time so a
# script that rebinds e.g. agent.ROUTER_SYSTEM_PROMPT is still recognized.
_STAGE_PROMPTS = [
    ("ROUTER_SYSTEM_PROMPT", "router"),
    ("ROUTER_DECIDE_NEXT_AGENT_ID_SYSTEM_PROMPT", "router_next"),
    ("THOUGHT_SYSTEM_PROMPT", "thought"),
    ("EXECUTOR_SYSTEM_PROMPT", "executor"),
]

_active = None


def activate(run_trace):
    global _active
    _active = run_trace


def get_active():
    return _active


def stage_for(system_prompt):
    agent_mod = sys.modules.get("src.agent")
    if agent_mod is not None:
        for attr, label in _STAGE_PROMPTS:
            if system_prompt == getattr(agent_mod, attr, None):
                return label
    return "llm"


def _new_counter():
    return {"input": 0, "output": 0, "calls": 0}


def _add(counter, input_tokens, output_tokens, calls=1):
    counter["input"] += input_tokens
    counter["output"] += output_tokens
    counter["calls"] += calls


def format_tokens(summary):
    a, b = summary["all"], summary["benign"]
    return (f"in {a['input']} / out {a['output']} "
            f"(benign in {b['input']} / out {b['output']})")


def traced(fn, agent_id):
    @functools.wraps(fn)
    def _wrapped(system_prompt, query_prompt, **kwargs):
        if _active is None:
            return fn(system_prompt=system_prompt, query_prompt=query_prompt, **kwargs)
        return _active.traced_call(fn, agent_id, system_prompt, query_prompt, **kwargs)
    return _wrapped


def record_api_usage(system_prompt, query_prompt, response, input_tokens, output_tokens):
    if _active is not None:
        _active.record_api_usage(system_prompt, query_prompt, response, input_tokens, output_tokens)


def record_index(record):
    match = re.search(r"task(\d+)", record["file"] or "")
    return int(match.group(1)) if match else 0


def record_trial(record):
    match = re.search(r"_trial(\d+)", record["file"] or "")
    return int(match.group(1)) if match else None


def _sum_timing(records):
    total = {}
    for r in records:
        timing.add_into(total, r.get("timing"))
    return {mode: {k: round(v, 4) for k, v in view.items()} for mode, view in total.items()} or None


def _record_order(record):
    return (str(record["mode"]), record_index(record), record_trial(record) or 0)


class _TaskState:
    def __init__(self, fh, filename, mode, manual_agents=None, json_path=None):
        self.fh = fh
        self.filename = filename
        self.mode = mode
        self.manual_agents = manual_agents or {}   # agent_id -> script slot, this task only
        self.stack = []
        self.usage = {}
        self.call_index = 0
        # JSON copy of the trace, dumped to json_path when the task ends.
        self.json_path = json_path
        self.doc = {"events": []}
        self.json_written = False

    def write(self, text):
        self.fh.write(text)
        self.fh.flush()

    def add_event(self, event):
        self.doc["events"].append(event)

    def write_json(self):
        with open(self.json_path, "w") as f:
            json.dump(self.doc, f, indent=2, ensure_ascii=False, default=str)
        self.json_written = True


def _call_fields(call):
    return {"system_prompt": call["system_prompt"], "query_prompt": call["query_prompt"],
            "response": call["response"],
            "input_tokens": call["input"], "output_tokens": call["output"]}


class _TraceFileHandler(logging.Handler):
    def __init__(self, run_trace):
        super().__init__()
        self._run_trace = run_trace
        self.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

    def emit(self, record):
        state = self._run_trace._state()
        if state is None:
            return
        try:
            state.write(self.format(record) + "\n")
            state.add_event({"type": "log", "level": record.levelname,
                             "message": record.getMessage()})
        except Exception:
            self.handleError(record)


class RunTrace:
    def __init__(self, trace_dir, agent_llms):
        """agent_llms: {agent_id: LLM key}; ``manual-*`` keys are non-benign."""
        self.trace_dir = trace_dir
        os.makedirs(trace_dir, exist_ok=True)
        self.agent_llms = dict(agent_llms)

        self._local = threading.local()
        self._lock = threading.Lock()
        self._totals = {}   
        self._benign_totals = {}
        self._records = []
        self._wall_clock = {} 

        self._log_handler = _TraceFileHandler(self)
        logging.getLogger().addHandler(self._log_handler)

    def _state(self):
        return getattr(self._local, "task", None)

    def _task_manual_agents(self):
        state = self._state()
        return state.manual_agents if state is not None else {}

    def is_benign(self, agent_key):

        if agent_key in self._task_manual_agents():
            return False
        llm = self.agent_llms.get(agent_key)
        return llm is not None and not str(llm).startswith("manual-")

    def _describe(self, agent_key):
        if agent_key == UNATTRIBUTED:
            return UNATTRIBUTED
        slot = self._task_manual_agents().get(agent_key)
        if slot is not None:
            return f"Agent {agent_key} (manual, script {slot})"
        kind = "benign" if self.is_benign(agent_key) else "manual"
        return f"Agent {agent_key} ({kind}, {self.agent_llms.get(agent_key)})"

    def traced_call(self, fn, agent_id, system_prompt, query_prompt, **kwargs):
        state = self._state()
        if state is None:
            return fn(system_prompt=system_prompt, query_prompt=query_prompt, **kwargs)

        frame = {"agent_id": agent_id, "stage": stage_for(system_prompt),
                 "input": 0, "output": 0, "api_calls": []}
        state.stack.append(frame)
        start = time.time()
        response, error = None, None
        try:
            response = fn(system_prompt=system_prompt, query_prompt=query_prompt, **kwargs)
            return response
        except Exception as e:
            error = e
            raise
        finally:
            state.stack.pop()
            self._write_call(state, frame, system_prompt, query_prompt, response, error, time.time() - start)

    def record_api_usage(self, system_prompt, query_prompt, response, input_tokens, output_tokens):
        state = self._state()
        agent_key = state.stack[-1]["agent_id"] if state is not None and state.stack else UNATTRIBUTED
        mode = state.mode if state is not None else None

        with self._lock:
            mode_totals = self._totals.setdefault(mode, {})
            _add(mode_totals.setdefault(agent_key, _new_counter()), input_tokens, output_tokens)
            if self.is_benign(agent_key):
                _add(self._benign_totals.setdefault(mode, _new_counter()), input_tokens, output_tokens)
        if state is None:
            return

        _add(state.usage.setdefault(agent_key, _new_counter()), input_tokens, output_tokens)
        call = {"system_prompt": system_prompt, "query_prompt": query_prompt,
                "response": response, "input": input_tokens, "output": output_tokens}
        if state.stack:
            frame = state.stack[-1]
            frame["input"] += input_tokens
            frame["output"] += output_tokens
            frame["api_calls"].append(call)
        else:
            self._write_unattributed(state, call)

    def _summarize(self, usage, benign=None):

        summary = {"by_agent": {str(k): dict(v) for k, v in usage.items()},
                   "all": _new_counter(), "benign": _new_counter()}
        for agent_key, counter in usage.items():
            _add(summary["all"], counter["input"], counter["output"], counter["calls"])
            if benign is None and self.is_benign(agent_key):
                _add(summary["benign"], counter["input"], counter["output"], counter["calls"])
        if benign is not None:
            summary["benign"] = dict(benign)
        return summary

    def task_tokens(self):
        state = self._state()
        return self._summarize(state.usage if state is not None else {})

    def cumulative_tokens(self, mode):
        with self._lock:
            usage = {k: dict(v) for k, v in self._totals.get(mode, {}).items()}
            benign = dict(self._benign_totals.get(mode, _new_counter()))
        return self._summarize(usage, benign)

    def set_wall_clock(self, mode, seconds):
        with self._lock:
            self._wall_clock[mode] = self._wall_clock.get(mode, 0.0) + seconds
            self._write_summary()

    def restore(self, keep):

        path = os.path.join(self.trace_dir, "summary.json")
        if not os.path.exists(path):
            return []
        with open(path) as f:
            summary = json.load(f)
        kept = [r for r in summary.get("per_task", []) if keep(r)]
        with self._lock:
            self._records = kept
            for record in kept:
                mode_totals = self._totals.setdefault(record["mode"], {})
                for agent_key, counter in record["tokens"]["by_agent"].items():
                    agent_key = int(agent_key) if agent_key.isdigit() else agent_key
                    _add(mode_totals.setdefault(agent_key, _new_counter()),
                         counter["input"], counter["output"], counter["calls"])
                benign = record["tokens"]["benign"]
                _add(self._benign_totals.setdefault(record["mode"], _new_counter()),
                     benign["input"], benign["output"], benign["calls"])
            for mode in {r["mode"] for r in kept}:
                wall_clock = summary.get(str(mode).lower(), {}).get("wall_clock")
                if wall_clock:
                    self._wall_clock[mode] = wall_clock
            self._write_summary()
        return kept


    def start_task(self, filename, task, mode):
        self.close_task()
        with self._lock:
            self._totals.setdefault(mode, {})
        state = _TaskState(open(os.path.join(self.trace_dir, filename), "w"), filename, mode,
                           getattr(task, "manual_agents", None),
                           os.path.join(self.trace_dir, os.path.splitext(filename)[0] + ".json"))
        self._local.task = state

        trial = getattr(task, "trial", None)
        private_info = getattr(task, "private_info", None)
        state.doc = {
            "mode": mode, "file": filename, "task_id": task.task_id, "trial": trial,
            "task_type": task.task_type, "major_problem": task.major_problem,
            "private_information": {str(agent_id): list(facts)
                                    for agent_id, facts in enumerate(private_info or [])},
            "correct_answer": task.correct_answer,
            "agents": [{"agent_id": agent_id, "description": self._describe(agent_id),
                        "benign": self.is_benign(agent_id), "llm": llm}
                       for agent_id, llm in self.agent_llms.items()],
            "events": [], "summary": None,
        }
        lines = ["=" * 80,
                 f"{mode} task {task.task_id}" + (f" trial {trial}" if trial is not None else "")
                 + f" (type: {task.task_type})",
                 "=" * 80,
                 "", "--- MAJOR PROBLEM ---", task.major_problem]
        if private_info:
            lines += ["", "--- PRIVATE INFORMATION ---"]
            for agent_id, facts in enumerate(private_info):
                lines.append(f"Agent {agent_id}:")
                lines += [f"  - {f}" for f in facts] or ["  (none)"]
        lines += ["", f"--- CORRECT ANSWER --- {task.correct_answer}",
                  "", "--- AGENTS ---"]
        lines += [self._describe(agent_id) for agent_id in self.agent_llms]
        state.write("\n".join(lines) + "\n\n")

    def _write_call(self, state, frame, system_prompt, query_prompt, response, error, elapsed):
        state.call_index += 1
        lines = ["-" * 80,
                 f"[call {state.call_index}] {self._describe(frame['agent_id'])} | "
                 f"{frame['stage']} | in {frame['input']} / out {frame['output']} | {elapsed:.2f}s",
                 "-" * 80,
                 "--- SYSTEM PROMPT ---", system_prompt,
                 "--- QUERY PROMPT ---", query_prompt,
                 "--- RESPONSE ---", f"<ERROR: {error!r}>" if error is not None else str(response)]

        internal_calls = [call for call in frame["api_calls"]
                          if not (call["system_prompt"] == system_prompt
                                  and call["query_prompt"] == query_prompt)]
        for call in internal_calls:
            lines += ["", f"  >> internal LLM call | in {call['input']} / out {call['output']}",
                      "  --- SYSTEM PROMPT ---", call["system_prompt"],
                      "  --- QUERY PROMPT ---", call["query_prompt"],
                      "  --- RESPONSE ---", str(call["response"])]
        state.write("\n".join(lines) + "\n\n")
        state.add_event({
            "type": "llm_call", "index": state.call_index, "agent_id": frame["agent_id"],
            "agent": self._describe(frame["agent_id"]), "stage": frame["stage"],
            "input_tokens": frame["input"], "output_tokens": frame["output"],
            "elapsed": round(elapsed, 3),
            "system_prompt": system_prompt, "query_prompt": query_prompt,
            "response": None if error is not None else response,
            "error": repr(error) if error is not None else None,
            "internal_calls": [_call_fields(call) for call in internal_calls],
        })

    def _write_unattributed(self, state, call):
        state.add_event({"type": "unattributed_llm_call", **_call_fields(call)})
        state.write("\n".join([
            "-" * 80,
            f"[LLM call outside any agent] in {call['input']} / out {call['output']}",
            "-" * 80,
            "--- SYSTEM PROMPT ---", call["system_prompt"],
            "--- QUERY PROMPT ---", call["query_prompt"],
            "--- RESPONSE ---", str(call["response"])]) + "\n\n")

    def end_task(self, result):
        """result: the dict from Experiment.evaluate_task_result, or None if
        the task crashed before it was scored."""
        state = self._state()
        if state is None:
            return
        task_summary = self.task_tokens()
        cumulative = self.cumulative_tokens(state.mode)

        lines = ["=" * 80, "TASK SUMMARY", "=" * 80]
        if result is None:
            lines.append("Task did not complete (exception raised); see run log.")
        else:
            lines += [f"Final answer: {result['final_result']}",
                      f"Ground truth: {result['correct_answer']}",
                      f"Success: {result['success']}",
                      f"Path: {result['agent_path']} (* = executed/split)",
                      f"Runtime: {result['runtime']:.3f}s (non-manual {result['non_manual_runtime']:.3f}s)"]
            if timing.is_split(result.get("timing")):
                for mode in timing.MODES:
                    lines.append(f"Timing verify_{mode} (s): {timing.format(result['timing'][mode])}")
            if result.get("rejected"):
                lines.append(f"Rejected: {result['rejected']}")
        lines += ["", "Tokens by agent (input / output / calls):"]
        for agent_key, counter in task_summary["by_agent"].items():
            agent_key = int(agent_key) if agent_key.isdigit() else agent_key
            lines.append(f"  {self._describe(agent_key)}: "
                         f"{counter['input']} / {counter['output']} / {counter['calls']}")
        lines += [f"Task tokens: {format_tokens(task_summary)}",
                  f"Cumulative {state.mode} tokens so far: {format_tokens(cumulative)}"]
        state.write("\n".join(lines) + "\n")

        summary = {"completed": result is not None}
        if result is not None:
            summary.update({"final_answer": result["final_result"],
                            "ground_truth": result["correct_answer"],
                            "success": result["success"],
                            "path": result["agent_path"],
                            "runtime": result["runtime"],
                            "non_manual_runtime": result["non_manual_runtime"],
                            "timing": result.get("timing"),
                            "rejected": result.get("rejected")})
        summary.update({"tokens": task_summary, "cumulative_tokens": cumulative})
        state.doc["summary"] = summary
        state.write_json()

        record = {"mode": state.mode, "file": state.filename, "completed": result is not None}
        if result is not None:
            record.update(result)
        record["tokens"] = task_summary
        with self._lock:
            self._records.append(record)
            self._write_summary()
        self.close_task()

    def close_task(self):
        state = self._state()
        if state is not None:
            state.fh.close()
            if not state.json_written:
                state.write_json()
            self._local.task = None

    def _write_summary(self):
        """Caller holds self._lock."""
        summary = {}
        for mode in self._totals:
            records = [r for r in self._records if r["mode"] == mode]
            scored = [r for r in records if r["completed"]]
            summary[str(mode).lower()] = {
                "tasks": len(scored),
                "failed_to_complete": len(records) - len(scored),
                "accuracy": (sum(bool(r["success"]) for r in scored) / len(scored)) if scored else None,
                "runtime": sum(r["runtime"] for r in scored),
                "non_manual_runtime": sum(r["non_manual_runtime"] for r in scored),
                "wall_clock": self._wall_clock.get(mode),
                "timing": _sum_timing(scored),
                "tokens": self._summarize(self._totals[mode],
                                          self._benign_totals.get(mode, _new_counter())),
            }
        summary["per_task"] = sorted(self._records, key=_record_order)
        with open(os.path.join(self.trace_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=4, default=str)
