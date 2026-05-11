"""Verify 3-view availability per (Setup, Performer, Replication, Action) sample.

NTU naming: S<setup>C<camera>P<performer>R<replication>A<action>
We group by (S, P, R, A) and check how many of the 3 cameras (C001/C002/C003)
are present.  Reports per-split coverage, which is critical for any model
that requires all 3 views at inference (X-Sub protocol).
"""

import argparse
import pickle
import re
from collections import Counter, defaultdict


PAT = re.compile(r'^S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})$')


def parse(name: str):
    m = PAT.match(name)
    if not m:
        return None
    s, c, p, r, a = (int(x) for x in m.groups())
    return s, c, p, r, a


def coverage(split_name: str, sample_names: list[str]) -> None:
    groups = defaultdict(set)        # (S,P,R,A) -> {cameras}
    skipped = 0
    for name in sample_names:
        parsed = parse(name)
        if parsed is None:
            skipped += 1
            continue
        s, c, p, r, a = parsed
        groups[(s, p, r, a)].add(c)

    counts = Counter(len(cams) for cams in groups.values())
    n_groups = len(groups)
    print(f'\n--- {split_name} ---')
    print(f'  raw samples         : {len(sample_names):>7d}  (skipped: {skipped})')
    print(f'  unique (S,P,R,A)    : {n_groups:>7d}')
    for k in sorted(counts):
        pct = 100.0 * counts[k] / n_groups
        print(f'  groups with {k} view(s): {counts[k]:>7d}   ({pct:5.1f}%)')

    # Any sample that uses a non-{1,2,3} camera?
    cams_seen = Counter(c for cams in groups.values() for c in cams)
    print(f'  camera IDs seen     : {dict(sorted(cams_seen.items()))}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pkl', default='/data/3dpose/ntu-rgbd/processed/hrnet/ntu60_hrnet.pkl')
    args = p.parse_args()

    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)

    print(f'pkl: {args.pkl}')
    print(f'splits available: {list(data["split"].keys())}')

    for split_name, names in data['split'].items():
        coverage(split_name, names)


if __name__ == '__main__':
    main()
