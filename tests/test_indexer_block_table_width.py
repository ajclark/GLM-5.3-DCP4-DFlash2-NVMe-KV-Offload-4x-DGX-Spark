"""Run the source-executing indexer regression gate in its own interpreter."""
from pathlib import Path
import subprocess
import sys


def test_fixed_indexer_block_table_widths():
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, str(root / "bench/repro/check_indexer_block_table_fix.py"),
         "--source", str(root / "overlay/vllm/v1/attention/backends/mla/indexer.py"),
         "--device", "cpu"],
        check=True, capture_output=True, text=True,
    )
