#!/usr/bin/env python3
"""Exercise the pinned indexer source on CPU, including real Triton interpretation.

Run: .venv/bin/python bench/repro/indexer_block_table_width.py
No vLLM import, model, CUDA device, or serving requests are needed. The candidate
is applied ONLY to temporary source files. --patch-output emits a reviewable diff.
The old uniform kernel is given an extra backing row: its illegal logical reads
are observable sentinels, never accesses outside the actual CPU allocation.
"""

import argparse
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

os.environ.setdefault("TRITON_INTERPRET", "1")

import torch
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parent.parent.parent
REL = Path("vllm/v1/attention/backends/mla/indexer.py")
CLASS = "DeepseekV32IndexerMetadataBuilder"
METHOD = "_prepare_decode_tensors"
KERNEL = "_prepare_uniform_decode_kernel"


def candidate_source(source):
    """Minimal containment proposal, preserving contiguous destination rows."""
    edits = [
        ("    block_table_stride,\n", "    block_table_stride,\n    num_block_table_cols,\n"),
        ("        src_block = tl.load(src + off, mask=mask)\n",
         "        src_block = tl.load(\n"
         "            src + off, mask=mask & (off < num_block_table_cols), other=0\n"
         "        )\n"),
        ("        if not use_native and max_decode_len > 1:\n",
         "        if not use_native and max_decode_len > 1:\n"
         "            num_blocks = block_table.shape[1]\n"
         "            if num_blocks > self.expanded_block_table_buffer.shape[1]:\n"
         "                raise ValueError(\n"
         '                    "Indexer expanded block table is narrower than the input"\n'
         "                )\n"
         "            if block_table.stride(1) != 1:\n"
         '                raise ValueError("Indexer block table columns must have stride 1")\n'),
        ("                    block_table.stride(0),\n",
         "                    block_table.stride(0),\n                    num_blocks,\n"),
        ("                self.expanded_block_table_buffer[:actual_expanded] = (\n",
         "                self.expanded_block_table_buffer[:actual_expanded, :num_blocks] = (\n"),
        ("                if actual_expanded < num_decode_tokens:\n",
         "                self.expanded_block_table_buffer[:actual_expanded, num_blocks:] = 0\n"
         "                if actual_expanded < num_decode_tokens:\n"),
        ("                        actual_expanded:num_decode_tokens, 0\n",
         "                        actual_expanded:num_decode_tokens, :\n"),
    ]
    for before, after in edits:
        assert source.count(before) == 1, f"Source drift: {before!r}"
        source = source.replace(before, after)
    return source


def definitions(path):
    tree = ast.parse(path.read_text())
    kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == KERNEL)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == CLASS)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == METHOD)
    return kernel, method


def load(path, force_variable=False):
    kernel, method = definitions(path)
    if force_variable:
        # Exercise both source branches on identical uniform input, changing
        # only branch selection. Tensor operations are not reimplemented.
        branch = next(n for n in ast.walk(method) if isinstance(n, ast.If)
                      and ast.unparse(n.test) == "min_decode_len == max_decode_len")
        branch.test = ast.Constant(False)
    module = ast.fix_missing_locations(ast.Module(body=[kernel, method], type_ignores=[]))
    ns = {"torch": torch, "triton": triton, "tl": tl,
          "logger": SimpleNamespace(info_once=lambda *args: None)}
    exec(compile(module, str(path), "exec"), ns)
    return ns[METHOD]


def case(width, lengths, slack=1, pad=0, row_padding=0, seq_lens=None):
    batch = len(lengths)
    stride = width + row_padding
    # Deliberately retain allocated storage beyond the logical last row.
    backing = torch.arange((batch + 1) * stride, dtype=torch.int32).view(batch + 1, stride) + 100
    table = backing[:batch, :width]
    capacity = max(sum(lengths) + pad, batch * max(lengths), batch) + 8
    builder = SimpleNamespace(
        decode_seq_lens_buffer=torch.full((capacity,), -777, dtype=torch.int32),
        decode_lens_buffer=torch.full((capacity,), -777, dtype=torch.int32),
        expanded_block_table_buffer=torch.full((capacity, width + slack), -777, dtype=torch.int32),
        arange_buffer=torch.arange(capacity, dtype=torch.int32),
        offsets_buffer=torch.arange(max(lengths), dtype=torch.int32),
    )
    # Match build(): decode_lens aliases the persistent output buffer.
    lens = torch.tensor(lengths, dtype=torch.int32)
    builder.decode_lens_buffer[:batch] = lens
    seqs = torch.tensor(seq_lens if seq_lens is not None else
                        [64 + i * 31 if n else 0 for i, n in enumerate(lengths)], dtype=torch.int32)
    starts = torch.tensor([sum(lengths[:i]) for i in range(batch)], dtype=torch.int32)
    args = (builder, seqs, table, builder.decode_lens_buffer[:batch], lens.clone(),
            starts, batch, sum(lengths) + pad, False, max(lengths), max(lengths))
    return args, backing


def check_result(args, output):
    builder, seqs, table, _, cpu_lens, _, _, n, _, _, _ = args
    out_seqs, out_table, out_lens, batch, requires_padding = output
    lengths = cpu_lens.tolist()
    actual = sum(lengths)
    expected_seqs = [int(end) - length + j + 1
                     for end, length in zip(seqs, lengths) for j in range(length)]
    expected_rows = [row.tolist() for row, length in zip(table, lengths) for _ in range(length)]
    assert out_seqs.tolist() == expected_seqs + [0] * (n - actual)
    assert out_table[:actual, :table.shape[1]].tolist() == expected_rows
    assert torch.count_nonzero(out_table[:actual, table.shape[1]:]) == 0
    assert torch.count_nonzero(out_table[actual:]) == 0
    assert out_lens.tolist() == [1] * n
    assert batch == n and requires_padding is False
    assert out_table.data_ptr() == builder.expanded_block_table_buffer.data_ptr()
    assert out_table.is_contiguous()


def investigate(path, tmp):
    source = path.read_text()
    fixed_path = tmp / (path.relative_to(ROOT).parts[0] + "_indexer.py")
    fixed_path.write_text(candidate_source(source))
    original, fixed, forced = load(path), load(fixed_path), load(fixed_path, True)
    summary = {"source": str(path.relative_to(ROOT)),
               "sha256": hashlib.sha256(source.encode()).hexdigest(), "mixed_failures": []}
    for cp in (1, 2, 4):
        width = -(-180224 // (64 * cp))
        args, _ = case(width, [5, 8, 8], seq_lens=[49229, 56948, 49184])
        try:
            original(*args)
        except RuntimeError as exc:
            assert "expanded size" in str(exc)
            summary["mixed_failures"].append({"cp": cp, "source_width": width, "error": str(exc)})
        else:
            raise AssertionError("The pinned mixed-length failure did not reproduce")

    # W=4: row 0's fifth read is row 1 col 0; last row's fifth read
    # hits a guard outside the logical table, but inside allocated backing.
    args, backing = case(4, [2, 2, 2])
    old_out = original(*args)
    extra = old_out[1][:, 4].tolist()
    expected = [int(backing[i + 1, 0]) for i in range(3) for _ in range(2)]
    assert extra == expected
    summary["uniform_cross_row_reads"] = extra
    summary["logical_end_guard"] = int(backing[3, 0])

    # Width one broadcasts instead of raising: prove why a shape-only repro
    # must not choose W=1 and expect the same exception.
    args, _ = case(1, [1, 2])
    out = original(*args)
    assert torch.equal(out[1][:, 0], out[1][:, 1])
    summary["width_one_broadcast"] = out[1].tolist()

    passed = 0
    for width in (1, 4, 704, 1408, 2816):
        for slack in (0, 1, 3):
            for lengths in ([5, 8, 8], [1, 8], [3, 1, 4, 0], [8], [2, 2, 2]):
                for row_padding in (0, 2):
                    args, _ = case(width, lengths, slack, row_padding=row_padding)
                    before = args[2].clone()
                    out = fixed(*args)
                    check_result(args, out)
                    torch.testing.assert_close(args[2], before)
                    passed += 1
    for pad in (1, 3, 11):
        args, _ = case(4, [3, 1, 4, 0], pad=pad)
        check_result(args, fixed(*args))
        passed += 1
    for lengths in ([8], [2, 2, 2], [8, 8, 8]):
        args, _ = case(1408, lengths)
        out = fixed(*args)
        other_args, _ = case(1408, lengths)
        other = forced(*other_args)
        for a, b in zip(out[:3], other[:3]):
            torch.testing.assert_close(a, b)
        passed += 1
    for lengths in ([1, 8], [2, 2]):
        args, _ = case(4, lengths, slack=-1)
        try:
            fixed(*args)
        except ValueError as exc:
            assert "narrower" in str(exc)
        else:
            raise AssertionError("Undersized capacity was not rejected")
        passed += 1
    # Reuse poisoned workspaces to expose stale slack and stale padded rows.
    args, _ = case(4, [1, 8], pad=3)
    for poison in (991, 1234):
        args[0].expanded_block_table_buffer.fill_(poison)
        args[0].decode_lens_buffer[:2] = args[4]
        check_result(args, fixed(*args))
        passed += 1
    # Native mixed and plain decode bypass expansion and retain their table.
    for lengths, native in (([1, 1], False), ([1, 8], True)):
        args, _ = case(4, lengths)
        args = (*args[:8], native, *args[9:])
        out = fixed(*args)
        assert out[1] is args[2]
        assert out[3] == len(lengths)
        assert out[4] == (min(lengths) != max(lengths))
        passed += 1
    summary["candidate_checks_passed"] = passed
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-output", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--baseline-only", action="store_true",
                        help="Reproduce from the preserved baseline after the overlay is fixed")
    opts = parser.parse_args()
    torch.set_num_threads(1)
    paths = [ROOT / tree / REL for tree in ("baseline", "overlay")]
    if opts.baseline_only:
        paths = paths[:1]
    same = (all(ast.dump(a) == ast.dump(b) for a, b in zip(definitions(paths[0]), definitions(paths[1])))
            if len(paths) == 2 else None)
    # Compare semantics independent of line offsets, then execute each tree.
    assert same is not False, "Baseline and overlay definitions have diverged; use --baseline-only and validate the fixed overlay separately"
    with tempfile.TemporaryDirectory(prefix="indexer-width-") as td:
        reports = [investigate(p, Path(td)) for p in paths]
    report = {"torch": torch.__version__, "triton": triton.__version__,
              "execution": "CPU Torch + real source Triton interpreter; not a full vLLM/GPU run",
              "baseline_overlay_definitions_equal": same, "reports": reports}
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if opts.json_output:
        opts.json_output.write_text(rendered)
    if opts.patch_output:
        source = paths[-1].read_text()
        relative = str(paths[-1].relative_to(ROOT))
        diff = difflib.unified_diff(source.splitlines(True), candidate_source(source).splitlines(True),
                                    fromfile="a/" + relative, tofile="b/" + relative)
        opts.patch_output.write_text("".join(diff))


if __name__ == "__main__":
    main()
