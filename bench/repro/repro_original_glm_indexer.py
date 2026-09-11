#!/usr/bin/env python3
"""Check the original GLM DFlash image's indexer, without loading a model.

Uses actual source definitions with CPU Torch / Triton interpretation. The source
allocation includes a guard row so the original kernel never reads unallocated
CPU memory. No vLLM import or production request is made.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import indexer_block_table_width as H


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--json-output', type=Path)
    args = parser.parse_args()
    source = args.source.read_text()
    original = H.load(args.source)
    report = {'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
              'torch': H.torch.__version__, 'triton': H.triton.__version__,
              'cp_size': 1, 'block_size': 64, 'decode_lengths': [5, 8, 8],
              'configurations': []}
    with tempfile.TemporaryDirectory() as tmp:
        fixed_path = Path(tmp) / 'indexer.py'
        fixed_path.write_text(H.candidate_source(source))
        fixed = H.load(fixed_path)
        for max_len in (80000, 80064, 180224, 270000):
            unaligned = (max_len + 63) // 64
            # V2 runner aligns to 128 tokens; original indexer adds one column.
            runner_width = ((unaligned + 1) // 2) * 2
            buffer_width = unaligned + 1
            params, _ = H.case(runner_width, [5, 8, 8],
                               slack=buffer_width - runner_width)
            result = {'max_model_len': max_len, 'runner_width': runner_width,
                      'indexer_width': buffer_width}
            try:
                original(*params)
            except RuntimeError as exc:
                assert 'expanded size' in str(exc)
                result['original_error'] = str(exc)
                assert runner_width != buffer_width
            else:
                result['original_error'] = None
                assert runner_width == buffer_width
            params, _ = H.case(runner_width, [5, 8, 8],
                               slack=buffer_width - runner_width)
            H.check_result(params, fixed(*params))
            result['fixed_passed'] = True
            report['configurations'].append(result)

        params, backing = H.case(1250, [8, 8, 8])
        extra = original(*params)[1][:, 1250].tolist()
        expected = [int(backing[i + 1, 0]) for i in range(3) for _ in range(8)]
        assert extra == expected
        report['uniform_slack_values'] = extra
        report['guard_row_first_value'] = int(backing[3, 0])
        params, _ = H.case(1250, [8, 8, 8])
        H.check_result(params, fixed(*params))
        report['fixed_uniform_passed'] = True
    rendered = json.dumps(report, indent=2) + '\n'
    print(rendered, end='')
    if args.json_output:
        args.json_output.write_text(rendered)


if __name__ == '__main__':
    main()
