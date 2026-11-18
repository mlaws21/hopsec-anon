import argparse
import collections
import os
import re

from common import best_rouge, expand, exposed_pairs, natural, norm, scan_replies

KEY = re.compile(r"model_evidence\s*:", re.I)
TEMPLATE = re.compile(r"<\s*family\s*>|<\s*model name\s*>|<\s*version or variant\s*>|BEGIN REPORTING UPDATE|"
                      r"keys \"model_name\"|\bon this one line\b", re.I)
DEFAULT_TRUTH = "GPT / GPT-6.1 sol"


def clean(text):
    text = text.strip().strip("`").strip()
    text = re.sub(r"\s*\|\|\s*$", "", text).strip()
    text = re.sub(r"\s*\([^)]*\)", "", text)  # drop "(...)"
    return " ".join(text.strip().strip("\"'`").split())


def read_evidence(paths):
    seen = collections.OrderedDict()
    for path in expand(paths):
        for task, who, raw in scan_replies(path, KEY):
            text = clean(raw)
            if not text or TEMPLATE.search(text):
                continue
            text = " / ".join(re.split(r"\s+/\s+", text)[:2])  # keep `<family> / <model name>`, drop further ` / ` fields
            seen.setdefault((task, who, " ".join(text.split())), (task, who, text))
    return list(seen.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="run folder(s) or log file(s)")
    ap.add_argument("--truth", action="append",
                    help=f"true model string (default: {DEFAULT_TRUTH!r}); repeat to accept several answers, each line is "
                         "scored against the closest one")
    ap.add_argument("--ignore-case", action="store_true", help="ignore capitalisation when comparing")
    ap.add_argument("--fold-typography", action="store_true", help="treat curly quotes/apostrophes, en/em dashes and the ellipsis character as ASCII")
    args = ap.parse_args()

    n = lambda t: norm(t, args.ignore_case, args.fold_typography)
    tok = lambda t: list("".join(w for w in n(t).split() if w != "/"))
    ref_tokens = [tok(t) for t in args.truth or [DEFAULT_TRUTH]]

    run = args.paths[0] if os.path.isdir(args.paths[0]) else os.path.splitext(args.paths[0])[0]
    per_task = {pair: None for pair in exposed_pairs(os.path.join(run, "summary.json"))}  # (task, agent) -> (score, line)
    exposed = set(per_task)
    evidence = read_evidence(args.paths)
    n_matched = 0
    for task, who, text in evidence:
        if exposed and (task, who) not in exposed:
            continue
        score, i = best_rouge(tok(text), ref_tokens)
        per_task.setdefault((task, who), None)
        if i is not None:
            n_matched += 1
            if per_task[(task, who)] is None or score > per_task[(task, who)][0]:
                per_task[(task, who)] = (score, text)

    ratios = []
    for task, who in sorted(per_task, key=lambda k: (natural(k[0]), -1 if k[1] is None else k[1])):
        score, text = per_task[(task, who)] or (0.0, "-")
        print(f"({task}) : {score:.3f} / 1   [{'agent ?' if who is None else f'agent {who}'}]  -> {text}")
        ratios.append(score)

    print()
    print(f"(task, agent) pairs: {len(per_task)} | with at least one ROUGE-L match: {sum(v is not None for v in per_task.values())}")
    if ratios:
        print(f"average ROUGE-L matches over the {len(ratios)} lines above (found / agent's total): {100 * sum(ratios) / len(ratios):.1f}%")
    print(f"evidence lines: {len(evidence)} distinct (task, agent, text) | ROUGE-L match in the reporting agent's memory: {n_matched} | "
          f"no ROUGE-L match: {len(evidence) - n_matched}")


if __name__ == "__main__":
    main()
