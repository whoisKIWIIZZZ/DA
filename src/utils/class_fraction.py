from collections import Counter, defaultdict


def first_per_class(lines, fraction):
    if not 0 < fraction <= 1:
        raise ValueError('fraction must be in (0, 1]')
    if fraction == 1:
        return list(lines), []

    counts = Counter(int(line.rsplit(None, 1)[1]) for line in lines)
    limits = {label: int(count * fraction) for label, count in counts.items()}
    remaining = int(len(lines) * fraction) - sum(limits.values())
    for label in sorted(counts, key=lambda label: (-(counts[label] * fraction - limits[label]), label))[:remaining]:
        limits[label] += 1
    seen = defaultdict(int)
    selected, remainder = [], []
    for line in lines:
        label = int(line.rsplit(None, 1)[1])
        (selected if seen[label] < limits[label] else remainder).append(line)
        seen[label] += 1
    return selected, remainder
