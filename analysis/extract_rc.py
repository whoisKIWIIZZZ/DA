"""Extract the RC-task rows (Task line + Gate diagnostics) from a ProDe log.

Usage:
    python extract_rc.py [LOG_PATH]

Default input: ../office_home_prode.log (relative to this file's folder).
Output: rc_data.csv in this folder - one row per logged iteration of the
RC task, holding every field of the ``Gate:`` diagnostics line.

Why RC: the log covers all 12 Office-Home tasks; the ``Task: RC,`` marker
selects the RC source/target pair (lines 713-771 of the original log).
"""
import csv
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_LOG = HERE.parent / "office_home_prode.log"

TASK_RE = re.compile(
    r"Task: (\w+), Iter:(\d+)/(\d+); Accuracy = ([\d.]+)%;loss =(-?[\d.eE+-]+)"
)
TASK_ALPHA_RE = re.compile(r"alpha_mean=([\d.]+)")

# (csv columns, regex).  Each regex produces one group per column.
GATE_SPEC = [
    ("n", r"n=(\d+)"),
    ("alpha_mean", r"alpha mean=([\d.]+)"),
    ("alpha_p10,alpha_p50,alpha_p90",
     r"p10/50/90=([\d.]+)/([\d.]+)/([\d.]+)"),
    ("prior", r"prior=([\d.]+)"),
    ("conf_target,conf_clip", r"conf target/clip=([\d.]+)/([\d.]+)"),
    ("stab_target,stab_clip", r"stability target/clip=([\d.]+)/([\d.]+)"),
    ("sat_low,sat_high", r"sat low/high=([\d.]+)%/([\d.]+)%"),
    ("disagree", r"disagree=([\d.]+)%"),
    ("disacc_target,disacc_clip,disacc_selected",
     r"disagree acc target/clip/selected=([\d.]+)%/([\d.]+)%/([\d.]+)%"),
    ("oracle_target,oracle_clip",
     r"oracle alpha target-only/clip-only=([\d.]+)/([\d.]+)"),
    ("both_low_conf", r"both-low-conf=([\d.]+)%"),
    ("term_prior_mean,term_rel_mean,term_loss_mean",
     r"terms prior/reliability/loss mean=(-?[\d.]+)/(-?[\d.]+)/(-?[\d.]+)"),
    ("term_prior_p50,term_rel_p50,term_loss_p50",
     r"terms p50=(-?[\d.]+)/(-?[\d.]+)/(-?[\d.]+)"),
    ("term_prior_p90,term_rel_p90,term_loss_p90",
     r"p90=(-?[\d.]+)/(-?[\d.]+)/(-?[\d.]+)"),
    ("loss_sat", r"loss sat=([\d.]+)%"),
    ("gap_mean,gap_p50", r"gap mean/p50=(-?[\d.]+)/(-?[\d.]+)"),
    ("obs_mean,obs_p50", r"obs mean/p50=([\d.]+)/([\d.]+)"),
    ("tCE_mean,cCE_mean", r"scores targetCE/clipCE mean=([\d.]+)/([\d.]+)"),
    ("tCE_p50,cCE_p50", r"p50=([\d.]+)/([\d.]+)\s*$"),
]


def parse_gate(line):
    """Parse one ``Gate: ...`` line into a dict of floats (None if missing)."""
    out = {}
    for cols, pattern in GATE_SPEC:
        m = re.search(pattern, line)
        names = cols.split(",")
        if m:
            for col, val in zip(names, m.groups()):
                out[col] = float(val)
        else:
            for col in names:
                out[col] = None
    return out


def main(log_path):
    rows, cur = [], None
    with open(log_path) as f:
        for line in f:
            line = line.rstrip("\n")
            m = TASK_RE.search(line)
            if m:
                am = TASK_ALPHA_RE.search(line)
                cur = {
                    "task": m.group(1),
                    "iter": int(m.group(2)),
                    "accuracy": float(m.group(4)),
                    "alpha_mean_task": float(am.group(1)) if am else None,
                }
                continue
            if cur is not None and line.startswith("Gate:"):
                g = parse_gate(line)
                if g.get("n") is not None:  # skip "Gate: no observations"
                    rows.append({**cur, **g})

    rc = [r for r in rows if r["task"] == "RC"]
    if not rc:
        sys.exit("no Task: RC rows found in %s" % log_path)

    cols = ["task", "iter", "accuracy", "alpha_mean_task"]
    for colspec, _ in GATE_SPEC:
        cols += colspec.split(",")
    out_csv = HERE / "rc_data.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rc:
            w.writerow(r)

    last = rc[-1]
    print("RC rows: %d (iter %d -> %d)" % (len(rc), rc[0]["iter"], last["iter"]))
    print("wrote %s" % out_csv)
    print("final accuracy = %.2f%%; disagree = %.1f%%"
          % (last["accuracy"], last["disagree"]))
    print("final disag acc target/clip/selected = %.2f%% / %.2f%% / %.2f%%"
          % (last["disacc_target"], last["disacc_clip"], last["disacc_selected"]))
    print("final alpha mean/p10/p50/p90 = %.3f / %.3f / %.3f / %.3f"
          % (last["alpha_mean"], last["alpha_p10"],
             last["alpha_p50"], last["alpha_p90"]))
    gains = [(r["disacc_selected"] - max(r["disacc_target"], r["disacc_clip"]),
              r["iter"]) for r in rc]
    g, it = max(gains)
    print("max selection gain +%.2f pp over the better single branch at iter %d"
          % (g, it))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LOG)
