#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Parse ablation logs: for each CFG section, collect the FINAL-iteration accuracy
of every task run (lines where Iter:x/x has equal numbers), then report per-CFG.

Usage:  python analysis/parse_ablation_acc.py [log_path]
"""
import re
import sys
from collections import OrderedDict

CFG_RE = re.compile(r'^CFG:\s*(\S+)\s*\|')
# Task: AC, Iter:1035/1035; Accuracy = 72.94%;loss=...
TASK_RE = re.compile(
    r'Task:\s*(?P<task>[A-Za-z0-9_>.\-]+)\s*,\s*'
    r'Iter:\s*(?P<cur>\d+)\s*/\s*(?P<tot>\d+)\s*;\s*'
    r'Accuracy\s*=\s*(?P<acc>-?\d+(?:\.\d+)?)\s*%'
)

# canonical Office-Home order (source->target)
TASK_ORDER = ['AC', 'AP', 'AR', 'CA', 'CP', 'CR',
              'PA', 'PC', 'PR', 'RA', 'RC', 'RP']


def norm_task(name):
    """'A->C' / 'A_C' / 'ac' -> 'AC'"""
    return ''.join(ch for ch in name.upper() if ch.isalnum())


def parse(path):
    """-> OrderedDict{cfg: {task: acc}}"""
    cfgs = OrderedDict()
    cfg = None
    with open(path, 'r', errors='replace') as f:
        for lineno, line in enumerate(f, 1):
            m = CFG_RE.match(line)
            if m:
                cfg = m.group(1)
                cfgs.setdefault(cfg, {})
                continue
            m = TASK_RE.search(line)
            if not m:
                continue
            if m.group('cur') != m.group('tot'):   # only the last checkpoint
                continue
            if cfg is None:                        # task line before any CFG header
                cfg = '<no-CFG-header>'
                cfgs.setdefault(cfg, {})
            task = norm_task(m.group('task'))
            acc = float(m.group('acc'))
            if task in cfgs[cfg]:
                print('WARN %s:%d duplicate final line for %s (old %.2f -> new %.2f)'
                      % (path, lineno, task, cfgs[cfg][task], acc), file=sys.stderr)
            cfgs[cfg][task] = acc
    return cfgs


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'analysis/ablation_difo_ab.log'
    cfgs = parse(path)

    # per-CFG rows
    print('\n' + '=' * 100)
    print('%-52s %s' % ('CFG', ' '.join('%7s' % t for t in TASK_ORDER) + ' %8s' % 'MEAN'))
    print('-' * 100)
    incomplete = []
    for cfg, res in cfgs.items():
        cells, vals = [], []
        for t in TASK_ORDER:
            if t in res:
                cells.append('%7.2f' % res[t])
                vals.append(res[t])
            else:
                cells.append('%7s' % '-')
        mean = sum(vals) / len(vals) if vals else float('nan')
        missing = [t for t in TASK_ORDER if t not in res]
        if missing:
            incomplete.append((cfg, missing))
        print('%-52s %s %8.2f%s' % (cfg, ' '.join(cells), mean,
                                    '   (n=%d, missing %s)' % (len(vals), ','.join(missing))
                                    if missing else ''))
    print('=' * 100)

    # explicit note: tasks found but not in TASK_ORDER
    for cfg, res in cfgs.items():
        extra = [t for t in res if t not in TASK_ORDER]
        if extra:
            print('NOTE %s: unexpected task names in log: %s' % (cfg, extra))

    if incomplete:
        print('\nINCOMPLETE RUNS (no Iter:x/x line found):')
        for cfg, missing in incomplete:
            print('  %s -> %s' % (cfg, ', '.join(missing)))


if __name__ == '__main__':
    main()
