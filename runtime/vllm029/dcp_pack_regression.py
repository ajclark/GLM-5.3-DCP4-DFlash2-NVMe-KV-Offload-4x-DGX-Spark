#!/usr/bin/env python3
"""Model-free GPU repro for DCP pack stride specialization (Triton 3.7.1).

Extracts the exact wrapper/kernel from --source, avoiding model/vLLM startup.
No source changes or serving-process hooks. Each invocation needs a fresh process.
The fault mode simulates driver rejection; it does NOT reproduce driver aging.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import statistics
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import torch
import triton
from triton.runtime import driver


KERNEL = "_pack_dcp_topk_candidates_triton_kernel"
WRAPPER = "pack_dcp_topk_candidates_cutedsl"
STRIDES = (35072, 39936, 44544, 22528, 16640, 7424)


def emit(**fields):
    print(json.dumps(fields, sort_keys=True), flush=True)


def load_source(path, directory):
    source = path.read_text()
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in (KERNEL, WRAPPER)]
    assert {n.name for n in nodes} == {KERNEL, WRAPPER}
    # Real files are necessary for Triton's inspect.getsource().
    extracted = Path(directory) / "dcp_pack_extracted.py"
    lines = source.splitlines(keepends=True)
    extracted.write_text("import torch\nimport triton\nimport triton.language as tl\n\n"
                         + "\n".join("".join(lines[min(
                             [n.lineno] + [d.lineno for d in n.decorator_list]
                         ) - 1:n.end_lineno]) for n in nodes))
    spec = importlib.util.spec_from_file_location("dcp_pack_extracted", extracted)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, hashlib.sha256(source.encode()).hexdigest()


def make_case(width, rows=4, topk=2048, layout=0, has_starts=True,
              dtype=torch.float32):
    # Padded/sliced layouts exercise every stride argument, including unit,
    # aligned and unaligned values. Guard bytes expose unintended writes.
    col_step = (1, 2, 3)[layout % 3]
    row_stride = width * col_step + (0, 7, 16)[layout % 3]
    logits = torch.empty_strided((rows, width), (row_stride, col_step),
                                 device="cuda", dtype=dtype)
    values = ((torch.arange(rows * width, device="cuda").reshape(rows, width)
               % 1999) - 999).to(dtype)
    logits.copy_(values)
    idx_step = (1, 3, 2)[layout % 3]
    indices = torch.empty_strided((rows, topk), (topk * idx_step + layout, idx_step),
                                  device="cuda", dtype=torch.int32)
    indices.copy_((torch.arange(topk, device="cuda") % width)[None, :])
    indices[:, ::7] = -1
    indices[:, -1] = width + 5  # Existing wrapper clamps the score lookup.
    backing = torch.full((rows + 1, topk + (3 if layout else 0), 2 * (layout + 1)),
                         12345., device="cuda")
    packed = backing[:rows, :topk, :2 * (layout + 1):layout + 1]
    starts = (torch.arange(rows, device="cuda", dtype=torch.int32)
              if has_starts else None)
    return logits, indices, packed, starts, backing


def run_case(module, case, rank=0, world=2, interleave=1):
    logits, indices, packed, starts, _ = case
    getattr(module, WRAPPER)(logits, indices, packed, rank, world, interleave, starts)


def check_case(case, rank=0, world=2, interleave=1):
    logits, indices, packed, starts, backing = case
    ids = indices.long()
    valid = ids >= 0
    safe = ids.clamp_min(0)
    cols = safe + (starts[:, None] if starts is not None else 0)
    scores = logits.float().gather(1, cols.clamp_max(logits.shape[1] - 1))
    scores = scores.masked_fill(~valid, -float("inf"))
    global_ids = ((safe // interleave) * (world * interleave)
                  + rank * interleave + safe % interleave).masked_fill(~valid, -1)
    expected = torch.stack((scores, global_ids.float()), dim=-1)
    torch.testing.assert_close(packed, expected, rtol=0, atol=0)
    guards = torch.full_like(backing, 12345.)
    guards[:indices.shape[0], :indices.shape[1],
           :backing.shape[2]:backing.shape[2] // 2] = expected
    torch.testing.assert_close(backing, guards, rtol=0, atol=0)


class LoadMonitor:
    def __init__(self):
        self.loads = []
        self.reject = False
        self.original = driver.active.utils.load_binary

    def __call__(self, name, binary, *args, **kwargs):
        if name == KERNEL:
            self.loads.append({"name": name, "bytes": len(binary),
                               "cubin_sha256": hashlib.sha256(binary).hexdigest(),
                               "injected": self.reject})
            if self.reject:
                raise RuntimeError("INJECTED: Triton Error [CUDA]: operation not permitted")
        return self.original(name, binary, *args, **kwargs)


def sweep(module, monitor, require_reuse):
    # Hold all compile-time configuration fixed, change only shape/strides.
    for width in (*STRIDES, 7423, 7425, 1, 16):
        case = make_case(width, layout=0)
        run_case(module, case)
        check_case(case)
        emit(test="stride", width=width, loads=len(monitor.loads))
    for layout in (1, 2):
        case = make_case(7424, layout=layout)
        run_case(module, case)
        check_case(case)
        emit(test="layout", layout=layout, loads=len(monitor.loads))
    if require_reuse:
        assert len(monitor.loads) == 1, monitor.loads


def fault(module, monitor, require_reuse):
    first, unseen = make_case(35072), make_case(7424)
    run_case(module, first)
    check_case(first)
    assert len(monitor.loads) == 1
    monitor.reject = True
    # A warmed kernel remains usable even if subsequent module loads fail.
    run_case(module, first)
    check_case(first)
    try:
        run_case(module, unseen)
        check_case(unseen)
    except RuntimeError as exc:
        if not str(exc).startswith("INJECTED:"):
            raise
        emit(test="fault", outcome="injected_failure_reproduced", error=str(exc))
        if require_reuse:
            raise AssertionError("Unseen stride attempted a late module load") from exc
    else:
        emit(test="fault", outcome="no_late_load", loads=len(monitor.loads))
        assert len(monitor.loads) == 1


def correctness(module):
    count = 0
    # All DCP1/2/4 rank mappings, interleaving, start offsets, FP32/FP16/BF16,
    # invalid indices, empty shards, partial tiles and multi-tile TOPK.
    for world in (1, 2, 4):
        for rank in range(world):
            for interleave in (1, 16):
                for has_starts in (False, True):
                    for topk in (1, 511, 512, 513, 2048):
                        layout = count % 3
                        dtype = (torch.float32, torch.float16, torch.bfloat16)[layout]
                        case = make_case(97, topk=topk, layout=layout,
                                         has_starts=has_starts, dtype=dtype)
                        case[1][0, :] = -1  # Entire empty DCP shard row.
                        run_case(module, case, rank, world, interleave)
                        check_case(case, rank, world, interleave)
                        count += 1
    # The packing operation remains usable in a warmed CUDA graph.
    case = make_case(7424)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        run_case(module, case)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_case(module, case)
    case[0].add_(1)
    graph.replay()
    check_case(case)
    emit(test="correctness", cases=count, graph_replay="passed")


def benchmark(module):
    for rows in (1, 64, 2048):
        case = make_case(7424, rows=rows, topk=2048)
        run_case(module, case)
        # Event batches avoid the GB10 launch latency/CPU submission gap.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            run_case(module, case)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(100):
                run_case(module, case)
        times = []
        for _ in range(7):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000 / 100)
        check_case(case)
        emit(test="benchmark", rows=rows, topk=2048,
             median_us=statistics.median(times), samples_us=times)


def soak(module, monitor, seconds, interval, require_reuse):
    # Real, non-injected driver-aging experiment: same process/CUDA context.
    control = make_case(35072)
    run_case(module, control)
    check_case(control)
    deadline = time.monotonic() + seconds
    step = 0
    while time.monotonic() < deadline:
        width = 256 * (1 + step % 512)
        case = make_case(width)
        run_case(module, control)
        check_case(control)
        emit(test="soak_control", step=step, outcome="cached_kernel_passed")
        run_case(module, case)
        check_case(case)
        emit(test="soak", step=step, width=width, loads=len(monitor.loads),
             time=time.time())
        if require_reuse:
            assert len(monitor.loads) == 1
        step += 1
        time.sleep(min(interval, max(0, deadline - time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Defaults to the installed vLLM kernel")
    parser.add_argument("--mode", choices=("sweep", "fault", "correctness", "benchmark", "soak"),
                        default="sweep")
    parser.add_argument("--require-reuse", action="store_true")
    parser.add_argument("--seconds", type=float, default=4 * 24 * 3600)
    parser.add_argument("--interval", type=float, default=600)
    args = parser.parse_args()
    if args.seconds <= 0 or args.interval <= 0:
        parser.error("--seconds and --interval must be positive")
    if args.source is None:
        spec = importlib.util.find_spec("vllm")
        if spec is None:
            parser.error("vLLM is not installed; supply --source")
        args.source = (Path(spec.origin).parent
                       / "model_executor/kernels/attention/dsa/dcp_indexer_cutedsl.py")
    torch.cuda.init()
    torch.manual_seed(20260920)
    with tempfile.TemporaryDirectory(prefix="dcp-pack-") as directory:
        module, digest = load_source(args.source, directory)
        emit(source=str(args.source), source_sha256=digest, torch=torch.__version__,
             triton=triton.__version__, gpu=torch.cuda.get_device_name(), mode=args.mode)
        monitor = LoadMonitor()
        try:
            with patch.object(driver.active.utils, "load_binary", monitor):
                if args.mode == "sweep":
                    sweep(module, monitor, args.require_reuse)
                elif args.mode == "fault":
                    fault(module, monitor, args.require_reuse)
                elif args.mode == "correctness":
                    correctness(module)
                elif args.mode == "benchmark":
                    benchmark(module)
                else:
                    soak(module, monitor, args.seconds, args.interval, args.require_reuse)
        finally:
            emit(loads=monitor.loads)
    emit(result="passed")


if __name__ == "__main__":
    main()
