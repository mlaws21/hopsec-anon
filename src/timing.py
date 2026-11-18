import threading
import time
from contextlib import contextmanager

BUCKETS = ("llm_router", "llm_router_next", "llm_thought", "llm_executor", "llm_other",
           "llm_manual", "memory", "crypto_sign", "verify_all", "verify_one", "trace_log")
_SHARED = tuple(b for b in BUCKETS if b not in ("verify_all", "verify_one"))
VIEW_KEYS = _SHARED + ("crypto_verify", "other", "total")
MODES = ("all", "one")

_local = threading.local()


def start():
    _local.acc = {b: 0.0 for b in BUCKETS}
    _local.depth = 0


def stop():
    """The calling thread's buckets (empty if none were started); clears them."""
    acc = getattr(_local, "acc", None)
    _local.acc = None
    _local.depth = 0
    return acc or {}


@contextmanager
def timed(bucket):
    acc = getattr(_local, "acc", None)
    if acc is None:
        yield
        return
    outer = _local.depth == 0
    _local.depth += 1
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _local.depth -= 1
        if outer:
            acc[bucket] = acc.get(bucket, 0.0) + time.perf_counter() - t0


def finalize(buckets, total):
    shared = {b: buckets.get(b, 0.0) for b in _SHARED}
    views = {}
    for mode, other_mode in (("all", "one"), ("one", "all")):
        view = dict(shared, crypto_verify=buckets.get(f"verify_{mode}", 0.0))
        view_total = total - buckets.get(f"verify_{other_mode}", 0.0)
        view["other"] = view_total - sum(view.values())
        view["total"] = view_total
        views[mode] = {k: round(v, 4) for k, v in view.items()}
    return views


def is_split(timing):
    return isinstance(timing, dict) and all(isinstance(timing.get(m), dict) for m in MODES)


def add_into(acc, timing):
    if not is_split(timing):
        return acc
    for mode in MODES:
        view_acc = acc.setdefault(mode, {})
        for k, v in timing[mode].items():
            view_acc[k] = view_acc.get(k, 0.0) + v
    return acc


def format(view):
    t = {k: (view or {}).get(k, 0.0) for k in VIEW_KEYS}
    route = t["llm_router"] + t["llm_router_next"]
    s = (f"llm route {route:.2f} (next {t['llm_router_next']:.2f}) / thought {t['llm_thought']:.2f}"
         f" / exec {t['llm_executor']:.2f}")
    if t["llm_other"]:
        s += f" / other-llm {t['llm_other']:.2f}"
    s += (f" / manual {t['llm_manual']:.2f}, mem {t['memory']:.3f},"
          f" crypto sign {t['crypto_sign']:.4f} + verify {t['crypto_verify']:.4f},"
          f" log {t['trace_log']:.4f}, other {t['other']:.3f} | total {t['total']:.2f}")
    return s
