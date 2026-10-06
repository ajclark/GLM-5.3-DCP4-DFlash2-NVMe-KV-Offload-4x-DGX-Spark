#!/usr/bin/env python3
"""costs.json for the verify cap from cycle_bench.py fixed-K results (k1/k3/k5/k7.json).

    python3 make_costs.py ../../../results/verify-cost-screen-20260929 > costs.json
"""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
runs = {k: json.loads((d / f"k{k}.json").read_text())["summary"] for k in (1, 3, 5, 7)}
contexts = sorted({int(c) for r in runs.values() for c in r}, key=int)
points = [{"context": c, "cycle_ms": {str(k): round(runs[k][str(c)]["cycle_ms_server_median"], 2) for k in runs}}
          for c in contexts if all(str(c) in runs[k] for k in runs)]
print(json.dumps({"source": str(d), "points": points}, indent=1))
