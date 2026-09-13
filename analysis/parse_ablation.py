#!/usr/bin/env python3
"""Parse ablation logs: average final-iteration accuracy per CFG over all tasks.

Log format:
    CFG: <cfg path> | GPU: <n> | DA: <mode>
    ...
    Task: AD, Iter:120/120; Accuracy = 95.98%;loss = ...
    ...
"""
import re
import sys
from collections import OrderedDict

CFG_RE = re.compile(r"^CFG:\s*(\S+)")
TASK_RE = re.compile(
    r"Task:\s*(\S+?),\s*Iter:\s*(\d+)\s*/\s*(\d+)\s*;\s*Accuracy\s*=\s*([\d.]+)%"
)


NO_CFG = "(no CFG header)"


def parse(path):
    """Return OrderedDict: cfg -> OrderedDict(task -> final accuracy)."""
    results = OrderedDict()
    cfg = None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = CFG_RE.match(line)
        if m:
            cfg = m.group(1)
            results.setdefault(cfg, OrderedDict())
            continue
        m = TASK_RE.search(line)
        if m:
            # logs from a plain single run have no CFG header
            cfg = cfg or NO_CFG
            results.setdefault(cfg, OrderedDict())
            task, cur, total, acc = m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(4))
            # keep only the final iteration of each task
            if cur == total:
                results[cfg][task] = acc
    return results


def main(path):
    results = parse(path)
    if not results:
        print("no CFG/Task lines found in", path)
        return

    grand = []
    for cfg, tasks in results.items():
        print(f"\nCFG: {cfg}")
        if not tasks:
            print("  (no completed tasks)")
            continue
        for task, acc in tasks.items():
            print(f"  {task:<6} {acc:7.2f}")
        accs = list(tasks.values())
        avg = sum(accs) / len(accs)
        grand.extend(accs)
        print(f"  {'-' * 20}")
        print(f"  Avg over {len(accs)} task{'s' if len(accs) > 1 else ''}: {avg:.2f}")

    if len(results) > 1:
        print(f"\nOverall average over all CFGs ({len(grand)} tasks): "
              f"{sum(grand) / len(grand):.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "/home/yishouzhuo/work/sfda/analysis/ablation_difo_o31.log")
