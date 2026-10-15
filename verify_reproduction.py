import argparse
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--reference', type=Path, default=Path('results'))
parser.add_argument('--candidate', type=Path, default=Path('reproduced_results'))
args = parser.parse_args()
reference = json.loads((args.reference / 'results.json').read_text())
candidate = {row['run_id']: row for row in json.loads((args.candidate / 'results.json').read_text())}
if {row['run_id'] for row in reference} != set(candidate):
    raise SystemExit('Run identifiers differ')
failures = []
for row in reference:
    name = row['run_id']
    other = candidate[name]
    if {k: v for k, v in row.items() if k != 'runtime_ms'} != {k: v for k, v in other.items() if k != 'runtime_ms'}:
        failures.append(name + ': non-timing fields differ')
    for filename in ('response.npy', 'corners.npy'):
        a = np.load(args.reference / name / filename)
        b = np.load(args.candidate / name / filename)
        if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
            failures.append(name + ': ' + filename + ' differs')
print(json.dumps(dict(runs=len(reference), exact_match=not failures, failures=failures), indent=2))
raise SystemExit(1 if failures else 0)
