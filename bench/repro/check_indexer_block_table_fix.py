#!/usr/bin/env python3
"""Validate actual fixed source on CPU or CUDA, including graph replay.

python check_indexer_block_table_fix.py --source /path/to/indexer.py --device cuda
CUDA source allocations are exact-sized in contiguous cases, suitable for memcheck.
Only the two source definitions are loaded; no model or serving process is changed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--json-output", type=Path)
    args = ap.parse_args()
    os.environ["TRITON_INTERPRET"] = "1" if args.device == "cpu" else "0"
    import torch
    import triton
    import indexer_block_table_width as H

    torch.set_num_threads(1)
    fixed, forced = H.load(args.source), H.load(args.source, True)

    def make(width, lengths, slack=1, pad=0, row_padding=0):
        params, backing = H.case(width, lengths, slack, pad, row_padding)
        params = list(params)
        b = params[0]
        for name, value in vars(b).items():
            setattr(b, name, value.to(args.device))
        params[1] = params[1].to(args.device)
        # Exclude the guard row: fixed kernels must respect the actual allocation.
        storage = backing[:len(lengths)].clone().to(args.device)
        params[2] = storage[:, :width]
        params[3] = b.decode_lens_buffer[:len(lengths)]
        params[5] = params[5].to(args.device)
        return tuple(params)

    def check(params, output):
        H.check_result(params, output)

    count = 0
    for width in (1, 4, 704, 1408, 2816):
        for slack in (0, 1, 3):
            for lens in ([5, 8, 8], [1, 8], [3, 1, 4, 0], [8], [2, 2, 2]):
                for row_padding in (0, 2):
                    params = make(width, lens, slack, row_padding=row_padding)
                    before = params[2].clone()
                    check(params, fixed(*params))
                    torch.testing.assert_close(params[2], before)
                    count += 1
    for pad in (1, 3, 11):
        params = make(1408, [3, 1, 4, 0], pad=pad)
        check(params, fixed(*params))
        count += 1
    for lens in ([8], [2, 2, 2], [8, 8, 8]):
        first, second = make(1408, lens), make(1408, lens)
        a, b = fixed(*first), forced(*second)
        for x, y in zip(a[:3], b[:3]):
            torch.testing.assert_close(x, y)
        count += 1
    for lens in ([1, 8], [2, 2]):
        params = make(4, lens, slack=-1)
        try:
            fixed(*params)
        except ValueError as exc:
            assert "narrower" in str(exc)
        else:
            raise AssertionError("Expected explicit capacity rejection")
        count += 1
    graph_checks = 0
    if args.device == "cuda":
        for lens in ([5, 8, 8], [8, 8, 8]):
            params = make(1408, lens)
            # Warm allocations and Triton on a side stream before capture.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    params[0].decode_lens_buffer[:len(lens)].copy_(params[4])
                    fixed(*params)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            device_lens = params[4].to("cuda")
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                params[0].decode_lens_buffer[:len(lens)].copy_(device_lens)
                result = fixed(*params)
            for poison in (991, 1234):
                params[0].expanded_block_table_buffer.fill_(poison)
                graph.replay()
                torch.cuda.synchronize()
                check(params, result)
                graph_checks += 1
        torch.cuda.synchronize()
    report = {"source": str(args.source), "sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
              "device": args.device, "torch": torch.__version__, "triton": triton.__version__,
              "cases_passed": count, "cuda_graph_replays_passed": graph_checks}
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.json_output:
        args.json_output.write_text(rendered)


if __name__ == "__main__":
    main()
