import glob
import json
import os
import sys
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--tex", action="store_true", help="print LaTeX rows")
ap.add_argument("--model", type=str, help="model prefix of run dirs")
ap.add_argument("--both", nargs=2, metavar=("MODEL_A", "MODEL_B"),
                help="two model prefixes, printed side by side")
args = ap.parse_args()
if not (args.model or args.both):
    ap.error("pass --model or --both")

BASE = "\\textsc{Baseline}"
COLS = ["Visited", "ASR", "Time", "Tokens (in/out)", "Hops"]


def load(path):
    run_dir = os.path.dirname(path)
    with open(path) as f:
        s = json.load(f)
    cfg, model = {}, "?"
    cfg_path = os.path.join(run_dir, "run_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            rc = json.load(f)
        cfg, model = rc.get("args", {}), rc.get("normal_llm") or "?"
    tasks = [t for t in s.get("per_task", [])
             if t.get("mode") == "Test" and t.get("completed")]
    return cfg, model, tasks


key = lambda t: (t["task_chain_id"], t.get("trial", 1))
visited = lambda t: any(a.rstrip("*") in t.get("manual_agents", {})
                        for a in t.get("agent_path", []))
avg = lambda ts, f: sum(f(t) for t in ts) / len(ts)
tok = lambda t, io: t.get("tokens", {}).get("benign", {}).get(io, 0)
tim = lambda t, *ks, sc="all": sum(t.get("timing", {}).get(sc, {}).get(k, 0) for k in ks)
pct = lambda v: f"{int(100 * v + 0.5)}\\%"
mult = lambda v, ref: f"{v / ref:.1f}$\\times$" if ref else "--"


def stats(ts):
    return {
        "time": avg(ts, lambda t: t.get("runtime", 0)),
        "tin": avg(ts, lambda t: tok(t, "input")),
        "tout": avg(ts, lambda t: tok(t, "output")),
        "hops": avg(ts, lambda t: t["agents_visited"]),
    }


def breakdown(ts):
    """Average per-task runtime components (seconds)."""
    return {
        "memory": avg(ts, lambda t: tim(t, "memory")),
        "routing": avg(ts, lambda t: tim(t, "llm_router", "llm_router_next")),
        "execution": avg(ts, lambda t: tim(t, "llm_executor", "llm_thought")),
        "crypto_all": avg(ts, lambda t: tim(t, "crypto_sign", "crypto_verify")),
        "crypto_one": avg(ts, lambda t: tim(t, "crypto_sign", "crypto_verify", sc="one")),
    }


def build(prefix):
    """Return (rows, model label, info lines) for one model prefix."""
    runs = [load(p) for p in sorted(glob.glob(f"../log/[{prefix}]*[i|none]?/summary.json"))]

    cleans = [r for r in runs if r[0].get("script") == "10none"]
    clean = max(cleans, key=lambda r: sum(bool(t.get("success")) for t in r[2]), default=None)
    if not clean:
        sys.exit(f"No None run found for {prefix}.")
    ok = {key(t) for t in clean[2] if t.get("success")}
    base = [t for t in clean[2] if key(t) in ok]
    if not base:
        sys.exit(f"Clean run solved nothing for {prefix}.")
    B = stats(base)
    R = breakdown(clean[2])

    rows = []
    for cfg, _, tasks in runs:
        if cfg.get("script") == "none":
            continue
        name = str(cfg.get("script")).upper()
        elig = [t for t in tasks if key(t) in ok]
        hit = [t for t in elig if visited(t)]
        if not hit:
            rows.append([name, pct(0) if elig else "--", "--", "--", "--", "--"])
            continue
        S = stats(hit)
        rows.append([
            name,
            pct(len(hit) / len(elig)),
            pct(1 - avg(hit, lambda t: bool(t.get("success")))),
            mult(S["time"], B["time"]),
            f"{mult(S['tin'], B['tin'])} / {mult(S['tout'], B['tout'])}",
            mult(S["hops"], B["hops"]),
        ])

    rows.append([BASE, "--", "--",
                 f"{B['time']:.5f}",
                 f"{B['tin']:.0f} / {B['tout']:.0f}",
                 f"{B['hops']:.2f}"])

    label = ", ".join(sorted({m for _, m, _ in runs}))
    info = [
        f"Model: {label}; clean-correct (task, trial) pairs: {len(ok)}",
        f"Baseline runtime/task (s): memory {R['memory']:.3f}, "
        f"routing {R['routing']:.3f}, execution {R['execution']:.3f}, "
        f"crypto_all {R['crypto_all']*1000:.6f}, crypto_one {R['crypto_one']*1000:.6f}",
    ]
    return rows, label, info


if args.both:
    (ra, la, ia), (rb, lb, ib) = build(args.both[0]), build(args.both[1])
    da, db = {r[0]: r[1:] for r in ra}, {r[0]: r[1:] for r in rb}
    names = [r[0] for r in ra] + [r[0] for r in rb if r[0] not in da]
    names = [n for n in names if n != BASE] + [BASE]  # baseline last
    blank = ["--"] * len(COLS)
    rows = [[n] + da.get(n, blank) + db.get(n, blank) for n in names]
    header = ["Attack"] + COLS + COLS
    groups, infos = [la, lb], ia + ib
else:
    rows, label, info = build(args.model)
    header = ["Attack"] + COLS
    groups, infos = None, info

if args.tex:
    if groups:
        n = len(COLS)
        print(" & " + " & ".join(f"\\multicolumn{{{n}}}{{c}}{{\\texttt{{{g}}}}}"
                                 for g in groups) + r" \\")
        print(f"\\cmidrule(lr){{2-{n + 1}}} \\cmidrule(lr){{{n + 2}-{2 * n + 1}}}")
    print(" & ".join(header) + r" \\")
    print(r"\midrule")
    for r in rows:
        if r[0] == BASE:
            print(r"\midrule")
        print(" & ".join(r) + r" \\")
    for i in infos:
        print("% " + i)
else:
    plain = lambda c: (c.replace("$\\times$", "x").replace("\\%", "%")
                        .replace(BASE, "Baseline"))
    rows = [[plain(c) for c in r] for r in rows]
    w = [max(len(r[i]) for r in rows + [header]) for i in range(len(header))]
    if groups:
        left = sum(w[:len(COLS) + 1]) + 2 * len(COLS)
        print(f"{'':{w[0]}}  {groups[0]:<{left - w[0] - 2}}  {groups[1]}")
    for r in [header] + rows:
        print("  ".join(c.ljust(n) for c, n in zip(r, w)))
    print("-" * 70)
    for i in infos:
        print(i)