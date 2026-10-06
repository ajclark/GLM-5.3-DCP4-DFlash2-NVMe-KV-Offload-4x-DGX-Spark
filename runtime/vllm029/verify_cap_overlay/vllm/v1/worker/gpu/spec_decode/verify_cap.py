# SPDX-License-Identifier: Apache-2.0
"""Draft-aware verification cap for DFlash2, for single requests and small uniform batches.

The drafter always proposes 7 tokens. Right after it does, this module records the
selector's confidence in each drafted token (p_sel: softmax probability of the chosen
candidate given its predecessor, read from the scores the path walk already stored).
At the start of the next step, if 1..BATCH_MAX greedy decode requests are scheduled (and
nothing else), the worker verifies only the first K drafts of every request, one
K in {1, 3, 5, 7} for the whole batch, chosen to maximize

    sum_r E_r(K) - lambda_n * T(n, K, context)          (policy "lambda", default)
    E_r(K) = 1 + sum_{j<=K} prod_{i<=j} q_i,   q_i = calibration[i][bin(p_sel_i)]

where T is the measured cycle cost for n requests and lambda_n tracks the realized
expected-tokens-per-ms for batch size n (an EMA of the chosen ratio; the Dinkelbach rule
for maximizing long-run tokens per ms). Policy "ratio" maximizes sum_r E_r(K) / T instead
(the v3 rule). One K per batch keeps the verify batch uniform, which the sparse MLA
backend needs for its FULL CUDA graphs; every (n, K+1) shape gets its own graph.

The calibration is corrected online in log-odds space by a slow global bias and a
per-request bias, both learned from the outcomes of the drafts that were verified
(positions up to the first rejection; unverified positions are never counted), so a
request whose drafts keep being accepted (e.g. code) earns longer verification.
To keep the host's step preparation overlapped with the GPU, the worker first prepares
the step with a predicted cap (the most frequent recent cap for that request, or for that
batch size), then waits for the draft, decides, and prepares again only when the
decision differs. Unverified drafts count as rejected, so the scheduler's normal rollback
applies. Only a prefix of the same greedy draft is checked, so outputs match K=7 up to
the runtime's numerical nondeterminism.

Rank 0 decides and broadcasts K over the TP CPU group so every rank verifies alike.
Whether a step takes part in the broadcast depends only on the scheduler output, which
is identical on every rank; runtime control (below) only changes rank 0's decision.

Enable with VLLM_VERIFY_CAP=1 plus VLLM_VERIFY_CAP_CAL (calibration JSON) and
VLLM_VERIFY_CAP_COSTS (cycle-cost JSON). VLLM_VERIFY_CAP_BATCH_MAX (default 4) sets the
largest batch that gets short verification (1 = single requests only; it also sets
which batch graphs are captured). VLLM_VERIFY_CAP_FIXED=K starts in fixed mode.

Runtime control (no restart): VLLM_VERIFY_CAP_CONTROL names a JSON file that rank 0
re-reads when it changes (checked about once a second):
    {"mode": "auto" | "fixed" | "off", "fixed_k": 7, "batch": true,
     "policy": "lambda" | "ratio", "alpha": 0.05, "costs": "costs.json"}
"costs" (relative to the control file's directory) replaces the cost table. Rank 0 also
writes live cycle periods per (batch size, K, context band) to periods.json next to it.

Rank-local decision (VLLM_VERIFY_CAP_LOCAL_DECIDE=1; default 0 = rank 0 decides and
broadcasts over gloo, as above): every rank runs the same float64 decision, so no rank
waits on a per-step host broadcast. The drafter's p_sel is NOT bitwise identical across
ranks: the replicated int8 conv kernel_projection (6144 -> 1536) runs Marlin's atomic-add
split-K (VLLM_MARLIN_USE_ATOMIC_ADD=1), and its output enters the drafter's residual
stream after the all-reduce, so the hidden state that feeds the selector differs in the
last bits per rank. Right after drafting, rank 0's p_sel and sampled counts are therefore
broadcast on the GPU stream (one NCCL broadcast of n x 8 floats on the TP communicator; no
host sync). Every other input is identical by construction: the batch, slots and context
come from the scheduler output, and the learned biases and lambda are functions of those.
Rank 0's control-file and cost-table changes reach the other ranks at sync points, one
every VLLM_VERIFY_CAP_SYNC_EVERY decisions (default 32), run right after a verify launch
(off the critical path); they apply from the next decision on every rank. The same sync
compares a hash of each rank's decisions and decision state; a mismatch is logged as an
error and every rank falls back to rank-0-decide + broadcast for the rest of the run.
VLLM_VERIFY_CAP_EARLY=1 (with local decide) decides already in pre_trim when the draft is
complete there, so the step is prepared once, with the actual cap.
VLLM_VERIFY_CAP_GAP_EVENTS=1 times, with CUDA events on rank 0, the GPU's gap between
draft end and the next verify start.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import struct
import time
import zlib
from collections import Counter, deque

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Caps come from VLLM_VERIFY_CAP_CAPS (default 1,3,5,7; must include 7). Only the
# listed short caps get extra target graphs.
CAPS = tuple(sorted({int(x) for x in os.environ.get("VLLM_VERIFY_CAP_CAPS", "1,3,5,7").split(",")} | {7}))
if not all(1 <= k <= 7 for k in CAPS):
    raise ValueError("VLLM_VERIFY_CAP_CAPS must list values in 1..7")
QUERY_LENS = tuple(k + 1 for k in CAPS if k < 7)   # extra target graphs, per batch size
BATCH_MAX = int(os.environ.get("VLLM_VERIFY_CAP_BATCH_MAX", "4"))
if not 1 <= BATCH_MAX <= 12:
    raise ValueError("VLLM_VERIFY_CAP_BATCH_MAX must be 1..12")
MODES = ("auto", "fixed", "off")
POLICIES = ("lambda", "ratio")
CTX_BANDS = (8192, 49152)          # periods.json context bands: <8K, 8K-48K, >=48K


def enabled() -> bool:
    return os.environ.get("VLLM_VERIFY_CAP", "0") == "1"


def graph_shapes() -> tuple[tuple[int, int], ...]:
    """(num_reqs, query_len) of the extra uniform decode graphs to capture."""
    return tuple((n, q) for n in range(1, BATCH_MAX + 1) for q in QUERY_LENS)


class CostCurve:
    """Cycle ms per cap, linearly interpolated over context length, per batch size.

    {"points": [{"context": c, "cycle_ms": {"1": ms, ...}}, ...],        # 1 request
     "batch": {"2": {"points": [...]}, "3": {...}}}                      # optional
    """

    def __init__(self, data: dict):
        self.curves: dict[int, list] = {1: self._points(data["points"])}
        for n, d in (data.get("batch") or {}).items():
            self.curves[int(n)] = self._points(d["points"])

    @staticmethod
    def _points(raw: list) -> list:
        pts = sorted(raw, key=lambda p: p["context"])
        out = [(int(p["context"]), {int(k): float(v) for k, v in p["cycle_ms"].items()}) for p in pts]
        for _, c in out:
            if not set(CAPS) <= set(c) or not all(v > 0 for v in c.values()):
                raise ValueError(f"cost curve needs positive cycle_ms for caps {CAPS}")
        if not out:
            raise ValueError("cost curve has no points")
        return out

    def at(self, ctx: int, n: int = 1) -> dict[int, float] | None:
        pts = self.curves.get(n)
        if pts is None:
            return None
        if ctx <= pts[0][0]:
            return pts[0][1]
        for (x0, c0), (x1, c1) in zip(pts, pts[1:]):
            if ctx <= x1:
                w = (ctx - x0) / (x1 - x0)
                return {k: c0[k] + w * (c1[k] - c0[k]) for k in c0 if k in c1}
        return pts[-1][1]


class Control:
    """Rank-0 runtime settings, re-read from a JSON file when it changes."""

    def __init__(self, fixed: int | None):
        self.path = os.environ.get("VLLM_VERIFY_CAP_CONTROL") or None
        self.mode = "fixed" if fixed is not None else "auto"
        self.fixed_k = fixed if fixed is not None else 7
        self.batch = True
        self.policy = os.environ.get("VLLM_VERIFY_CAP_POLICY", "lambda")
        self.alpha = 0.05
        self.costs_file: str | None = None
        self.periods_epoch = 0                # bump in the file to restart the live periods (A/B arms)
        self._mtime = None
        self._next_check = 0.0

    def poll(self) -> bool:
        """True when the file changed and was applied."""
        if self.path is None:
            return False
        now = time.monotonic()
        if now < self._next_check:
            return False
        self._next_check = now + 1.0
        try:
            mtime = os.stat(self.path).st_mtime
        except OSError:
            return False
        if mtime == self._mtime:
            return False
        self._mtime = mtime
        try:
            d = json.loads(open(self.path).read())
            mode = d.get("mode", self.mode)
            fixed_k = int(d.get("fixed_k", self.fixed_k))
            policy = d.get("policy", self.policy)
            if mode not in MODES or fixed_k not in CAPS or policy not in POLICIES:
                raise ValueError(f"mode {mode!r}, fixed_k {fixed_k}, policy {policy!r}")
            self.mode, self.fixed_k, self.policy = mode, fixed_k, policy
            self.batch = bool(d.get("batch", self.batch))
            self.alpha = float(d.get("alpha", self.alpha))
            self.costs_file = d.get("costs") or None
            self.periods_epoch = int(d.get("periods_epoch", self.periods_epoch))
        except Exception as e:  # noqa: BLE001 -- a bad edit must not stop serving
            logger.warning("Verify cap: ignoring control file %s: %s", self.path, e)
            return False
        logger.info("Verify cap control: mode %s (fixed K %d), batch %s, policy %s, alpha %.3f, costs %s",
                    self.mode, self.fixed_k, self.batch, self.policy, self.alpha, self.costs_file)
        return True


class GapTimer:
    """VLLM_VERIFY_CAP_GAP_EVENTS=1, rank 0: CUDA timing events around each verify-cap
    step, read later without a host sync. a: right after the verify launch (completes when
    the verify ends); b: after the draft (record); c: just before the next verify launch.
    a->b is verify end -> draft end on the GPU (sampling, drafter, p_sel); b->c is the time
    the GPU spends between the draft and the next verify, i.e. waiting for the host plus
    the step's input-preparation kernels."""

    SLOTS = 4

    def __init__(self, make_event=None):
        make_event = make_event or (lambda: torch.cuda.Event(enable_timing=True))
        self.ev = [[make_event() for _ in range(3)] for _ in range(self.SLOTS)]
        self.state = [0] * self.SLOTS        # 0 free, 1 a, 2 a+b, 3 a+b+c (to be read)
        self.cur: int | None = None
        self.i = 0
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.draft_ms = self.gap_ms = self.gap_max = 0.0

    def _harvest(self) -> None:
        for s in range(self.SLOTS):
            if self.state[s] == 3 and self.ev[s][2].query():
                a, b, c = self.ev[s]
                g = b.elapsed_time(c)
                self.draft_ms += a.elapsed_time(b)
                self.gap_ms += g
                self.gap_max = max(self.gap_max, g)
                self.n += 1
                self.state[s] = 0

    def after_launch(self) -> None:
        self._harvest()
        s = self.i % self.SLOTS
        self.i += 1
        self.ev[s][0].record()
        self.state[s] = 1
        self.cur = s

    def record(self) -> None:
        s, self.cur = self.cur, None
        if s is None:
            return
        if self.state[s] == 1:
            self.ev[s][1].record()
            self.state[s] = 2
            self.cur = s
        else:                                # a second draft without a verify between
            self.state[s] = 0

    def before_launch(self) -> None:
        s, self.cur = self.cur, None
        if s is not None and self.state[s] == 2:
            self.ev[s][2].record()
            self.state[s] = 3

    def summary(self) -> str:
        if not self.n:
            return "no samples"
        return (f"verify end->draft end {self.draft_ms / self.n:.2f} ms, draft end->verify start "
                f"{self.gap_ms / self.n:.2f} ms (max {self.gap_max:.2f}, {self.n} steps)")


class VerifyCap:
    def __init__(self, vllm_config, device: torch.device):
        spec = vllm_config.speculative_config
        if spec is None or spec.method != "dflash" or vllm_config.num_speculative_tokens != 7:
            raise ValueError("VLLM_VERIFY_CAP requires DFlash with num_speculative_tokens=7")
        cal = json.loads(open(os.environ["VLLM_VERIFY_CAP_CAL"]).read())
        self.bins = int(cal["bins"])
        self.table = [[float(x) for x in row] for row in cal["q"]]      # [7][bins]
        if len(self.table) != 7 or any(len(r) != self.bins for r in self.table):
            raise ValueError("calibration table must be 7 x bins")
        if cal.get("binning") != "neglog10_1mp_over4":
            raise ValueError("calibration binning must be neglog10_1mp_over4")
        self._costs_data = json.loads(open(os.environ["VLLM_VERIFY_CAP_COSTS"]).read())
        self.costs = CostCurve(self._costs_data)
        fixed = os.environ.get("VLLM_VERIFY_CAP_FIXED")
        fixed = int(fixed) if fixed else None
        if fixed is not None and fixed not in CAPS:
            raise ValueError(f"VLLM_VERIFY_CAP_FIXED must be one of {CAPS}")
        self.ctl = Control(fixed)
        max_reqs = vllm_config.scheduler_config.max_num_seqs
        self.batch_max = min(BATCH_MAX, max_reqs)
        self.p_gpu = torch.zeros(max_reqs, 7, dtype=torch.float32, device=device)
        self.p_cpu = torch.zeros(max_reqs, 7, dtype=torch.float32, pin_memory=device.type == "cuda")
        self.acc_gpu = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        self.acc_cpu = torch.zeros(max_reqs, dtype=torch.int32, pin_memory=device.type == "cuda")
        self.slot_step = [-1] * max_reqs     # record() step that last wrote each slot
        self.req: dict[str, dict] = {}       # rank 0: last decision per request
        self.bias_global = 0.0
        self.lr_req = float(os.environ.get("VLLM_VERIFY_CAP_LR", "0.15"))
        self.lr_global = float(os.environ.get("VLLM_VERIFY_CAP_LR_GLOBAL", "0.01"))
        self.lam: dict[int, float] = {}      # rank 0: lambda per batch size (tokens per ms)
        self.step = 0
        self.event: torch.cuda.Event | None = None
        self.eligible: dict[str, bool] = {}
        self.counts: Counter = Counter()     # (n, k) decisions in the logging window
        self.t_sync = self.t_bcast = self.t_total = 0.0   # seconds, per logging window
        self.t_redo = 0.0                     # host re-preparation after mispredicts (model_runner)
        self.mispredicts = 0
        self.pending = None                   # (batch [(req_id, drafts)], predicted cap)
        self.hist: dict[str, deque] = {}      # all ranks: recent single-request caps per request
        self.hist_n: dict[int, deque] = {}    # all ranks: recent caps per batch size (n >= 2)
        self.recent: deque = deque(maxlen=64)
        self.last_log = time.monotonic()
        # rank 0: live cycle periods (time between consecutive decisions while decoding)
        self.periods: dict[tuple, deque] = {}
        self._periods_epoch = 0
        self.prev_decision = None             # (t, step, n, k, ctx)
        self.last_dump = time.monotonic()
        from vllm.distributed.parallel_state import get_tp_group
        self.tp = get_tp_group()
        self.k_buf = torch.zeros(1, dtype=torch.int32)
        # Step 4 (host critical path). All off by default: the rank-0 broadcast path.
        self.local = os.environ.get("VLLM_VERIFY_CAP_LOCAL_DECIDE", "0") == "1"
        self.early = os.environ.get("VLLM_VERIFY_CAP_EARLY", "0") == "1"
        self.sync_every = int(os.environ.get("VLLM_VERIFY_CAP_SYNC_EVERY", "32"))
        if self.sync_every < 1:
            raise ValueError("VLLM_VERIFY_CAP_SYNC_EVERY must be >= 1")
        self.pynccl = None                    # TP PyNccl communicator (local decide, world > 1)
        self.stage_gpu = None                 # [max_reqs, 8]: rank 0's p_sel (7) + sampled count
        self.drift_gpu = self.drift_cpu = None
        self.drift_rows = 0                   # drafted rows shared so far (this rank)
        self.drift: list | None = None        # rank 0: [(rows differing, rows)] per TP rank
        self.ctl_src: Control | None = None   # rank 0, local decide: the control-file reader
        self.ctl_seq = self._staged_seq = 0   # applied / staged (rank 0) control snapshot
        self.n_dec = 0                        # decisions so far (identical on every rank)
        self._hash = 0                        # crc32 of this sync window's decisions
        self.checks = 0
        self.check_fail = None                # (decision index, hashes) after a mismatch
        self._open = False                    # resolve ran for the step being launched
        self._last = None                     # local decide: (n, k, ctx) awaiting bookkeeping
        self._t_pre = 0.0
        self._t_record = None
        self.t_prep = self.t_host = self.t_window = self.t_dbcast = self.t_syncpt = 0.0
        self.n_window = self.n_early = 0
        self.gap = None
        if (os.environ.get("VLLM_VERIFY_CAP_GAP_EVENTS", "0") == "1" and device.type == "cuda"
                and self.tp.rank_in_group == 0):
            self.gap = GapTimer()
        if self.early and not self.local:
            logger.warning("Verify cap: VLLM_VERIFY_CAP_EARLY=1 needs VLLM_VERIFY_CAP_LOCAL_DECIDE=1; ignored")
        if self.local:
            self._init_local(fixed, max_reqs, device)
        if not self.local and self.tp.rank_in_group == 0:
            self._poll_control()
        logger.info("Verify cap enabled: caps %s, batch max %d, bins %d, mode %s, policy %s, "
                    "cost tables for n=%s, control %s, decide %s%s", CAPS, self.batch_max, self.bins,
                    self.ctl.mode, self.ctl.policy, sorted(self.costs.curves), self.ctl.path,
                    f"local (sync every {self.sync_every})" if self.local else "rank 0 + broadcast",
                    ", early" if self.local and self.early else "")

    # -- rank-local decision: shared inputs, control sync, rank check -----------------
    def _init_local(self, fixed: int | None, max_reqs: int, device: torch.device) -> None:
        """Every rank, at construction (the same point on all ranks): find the TP PyNccl
        communicator and give every rank rank 0's control state and cost table."""
        if self.tp.world_size > 1:
            comm = getattr(getattr(self.tp, "device_communicator", None), "pynccl_comm", None)
            if comm is None or getattr(comm, "disabled", True):
                logger.warning("Verify cap: VLLM_VERIFY_CAP_LOCAL_DECIDE=1 needs the TP group's PyNccl "
                               "communicator; using the rank-0 broadcast")
                self.local = False
                return
            self.pynccl = comm
            self.stage_gpu = torch.zeros(max_reqs, 8, dtype=torch.float32, device=device)
            # Diagnostic: drafted rows whose own p_sel differed from rank 0's (all-gathered
            # at sync points, logged by rank 0). Read without a sync; may lag one step.
            self.drift_gpu = torch.zeros(1, dtype=torch.int64, device=device)
            self.drift_cpu = torch.zeros(1, dtype=torch.int64, pin_memory=device.type == "cuda")
        snap = None
        if self.tp.rank_in_group == 0:
            self.ctl_src = Control(fixed)
            snap = self._control_snapshot(full=True)
        if self.tp.world_size > 1:
            box = [snap]
            torch.distributed.broadcast_object_list(box, src=self.tp.ranks[0], group=self.tp.cpu_group)
            snap = box[0]
        self._apply_snapshot(snap)

    def _control_snapshot(self, full: bool = False) -> dict | None:
        """Rank 0: poll the control file; its settings (plus the cost data it names, checked)
        as a snapshot, or None when nothing changed. `full` always returns one, carrying the
        current cost data when the file names none."""
        src = self.ctl_src
        changed = src.poll()
        if not changed and not full:
            return None
        costs = None
        if changed and src.costs_file:
            path = os.path.join(os.path.dirname(src.path), src.costs_file)
            try:
                data = json.loads(open(path).read())
                CostCurve(data)
                costs = data
            except Exception as e:  # noqa: BLE001
                logger.warning("Verify cap: keeping the current cost table; %s: %s", path, e)
        if full and costs is None:
            costs = self._costs_data
        self._staged_seq += 1
        return {"seq": self._staged_seq, "mode": src.mode, "fixed_k": src.fixed_k, "batch": src.batch,
                "policy": src.policy, "alpha": src.alpha, "costs_file": src.costs_file, "costs": costs}

    def _apply_snapshot(self, snap: dict) -> None:
        """Every rank, at the same decision index: adopt rank 0's control snapshot."""
        c = self.ctl
        c.mode, c.fixed_k, c.batch = snap["mode"], snap["fixed_k"], snap["batch"]
        c.policy, c.alpha, c.costs_file = snap["policy"], snap["alpha"], snap["costs_file"]
        if snap["costs"] is not None:
            self.costs = CostCurve(snap["costs"])
            self._costs_data = snap["costs"]
            self.lam.clear()
        self.ctl_seq = snap["seq"]
        if self.tp.rank_in_group == 0:
            logger.info("Verify cap control (local decide; from decision %d): mode %s (fixed K %d), batch %s, "
                        "policy %s, alpha %.3f, costs %s%s", self.n_dec + 1, c.mode, c.fixed_k, c.batch,
                        c.policy, c.alpha, c.costs_file,
                        f" (tables reloaded, n={sorted(self.costs.curves)})" if snap["costs"] is not None else "")

    def _share(self, p: torch.Tensor, num_sampled, n: int):
        """Every rank, right after drafting: replace the local p_sel (and sampled counts)
        by rank 0's, with one in-place NCCL broadcast on the current stream (no host sync)."""
        t0 = time.perf_counter()
        st = self.stage_gpu[:n]
        st[:, :7].copy_(p)
        if num_sampled is not None:
            st[:, 7].copy_(num_sampled[:n])
        self.pynccl.broadcast(st, src=0)
        self.drift_gpu += (st[:, :7] != p).any(-1).sum()
        self.drift_cpu.copy_(self.drift_gpu, non_blocking=True)   # complete by the next event sync
        self.drift_rows += n
        self.t_dbcast += time.perf_counter() - t0
        return st[:, :7], (st[:, 7] if num_sampled is not None else None)

    def _fingerprint(self, batch: list, n: int, k: int, ctx: int) -> bytes:
        """The decision and the state it came from, for the cross-rank check."""
        lam = self.lam.get(n)
        parts = [struct.pack("<qiiqdd", self.n_dec, n, k, ctx, self.bias_global,
                             lam if lam is not None else -1.0)]
        for rid, _ in batch:
            st = self.req.get(rid)
            if st is not None and "p" in st:
                parts.append(struct.pack(f"<d{len(st['p'])}d", st["bias"], *st["p"]))
        return b"".join(parts)

    def _sync_point(self) -> None:
        """Every rank, after the verify launch of decision n_dec (a multiple of
        sync_every): compare decision hashes and hand out rank 0's control changes."""
        t0 = time.perf_counter()
        rank0 = self.tp.rank_in_group == 0
        snap = self._control_snapshot() if rank0 else None
        drifted = int(self.drift_cpu[0]) if self.drift_cpu is not None else 0
        row = torch.tensor([self._hash, self.n_dec, self._staged_seq if rank0 else 0, drifted, self.drift_rows],
                           dtype=torch.int64)
        if self.tp.world_size > 1:
            rows = [torch.zeros_like(row) for _ in range(self.tp.world_size)]
            torch.distributed.all_gather(rows, row, group=self.tp.cpu_group)
            rows = [r.tolist() for r in rows]
        else:
            rows = [row.tolist()]
        if rows[0][2] > self.ctl_seq:          # rank 0 staged a change: everyone adopts it now
            box = [snap]
            if self.tp.world_size > 1:
                torch.distributed.broadcast_object_list(box, src=self.tp.ranks[0], group=self.tp.cpu_group)
            self._apply_snapshot(box[0])
        if self.tp.world_size > 1:
            self.drift = [(r[3], r[4]) for r in rows]
        self.checks += 1
        if any(r[:2] != rows[0][:2] for r in rows):
            self.check_fail = (self.n_dec, [r[0] for r in rows])
            self.local = False                 # every rank sees the same rows: all switch together
            logger.error("Verify cap: RANK MISMATCH in the local decision at decision %d (crc32 per TP rank %s, "
                         "decision counts %s); falling back to rank-0 decide + broadcast for the rest of the run",
                         self.n_dec, [r[0] for r in rows], [r[1] for r in rows])
        self._hash = 0
        self.t_syncpt += time.perf_counter() - t0

    @classmethod
    def maybe_create(cls, vllm_config, device):
        return cls(vllm_config, device) if enabled() else None

    # -- request bookkeeping ---------------------------------------------------------
    def on_new_request(self, req_id: str, sp) -> None:
        self.eligible[req_id] = bool(
            sp is not None and sp.temperature == 0
            and getattr(sp, "structured_outputs", None) is None
            and not getattr(sp, "logit_bias", None)
            and not getattr(sp, "allowed_token_ids", None)
            and not getattr(sp, "bad_words", None)
            and getattr(sp, "presence_penalty", 0) == 0
            and getattr(sp, "frequency_penalty", 0) == 0
            and getattr(sp, "repetition_penalty", 1) == 1
        )

    def on_finished(self, req_ids) -> None:
        for r in req_ids:
            self.eligible.pop(r, None)
            self.req.pop(r, None)
            self.hist.pop(r, None)

    # -- after drafting ----------------------------------------------------------------
    def record(self, speculator, input_batch, num_sampled=None) -> None:
        """Snapshot p_sel for this step's drafts and this step's sampled counts
        (async D2H; no host sync)."""
        n = input_batch.num_reqs
        scores = getattr(speculator, "_selector_scores", None)
        if scores is None or n == 0:
            return
        p = scores[:n].softmax(-1).amax(-1)                       # [n, 7]
        idx = input_batch.idx_mapping[:n]
        if self.local and self.pynccl is not None:
            p, num_sampled = self._share(p, num_sampled, n)
        self.p_gpu[idx] = p
        self.p_cpu.copy_(self.p_gpu, non_blocking=True)
        if num_sampled is not None:
            self.acc_gpu[idx] = num_sampled[:n].to(torch.int32)
            self.acc_cpu.copy_(self.acc_gpu, non_blocking=True)
        self.step += 1
        for slot in input_batch.idx_mapping_np[:n].tolist():
            self.slot_step[slot] = self.step
        if self.event is None:
            self.event = torch.cuda.Event()
        self.event.record()
        if self.gap is not None:
            self.gap.record()
        self._t_record = time.perf_counter()

    # -- decision math (rank 0; every rank with local decide) -------------------------
    def _q(self, j: int, pj: float, bias: float) -> float:
        u = min(1.0, -math.log10(max(1e-6, 1.0 - pj)) / 4.0)
        q = min(1 - 1e-4, max(1e-4, self.table[j][min(self.bins - 1, int(u * self.bins))]))
        if bias:
            q = 1.0 / (1.0 + math.exp(-(math.log(q / (1 - q)) + bias)))
        return q

    def _learn(self, st: dict, sampled: int) -> None:
        """Online log-odds correction from the verified prefix of the last draft."""
        accepted = sampled - 1
        if not 0 <= accepted <= st["k"]:
            return
        bias = self.bias_global + st["bias"]
        for j in range(st["k"]):
            y = 1.0 if j < accepted else 0.0
            err = y - self._q(j, st["p"][j], bias)
            st["bias"] = max(-3.0, min(3.0, st["bias"] + self.lr_req * err))
            self.bias_global = max(-3.0, min(3.0, self.bias_global + self.lr_global * err))
            if y == 0.0:
                break

    def _expected(self, p: list[float], bias: float) -> dict[int, float]:
        """Expected committed tokens (accepted + bonus) for every cap."""
        surv, cum, exp = 1.0, 1.0, {}
        for j in range(7):
            surv *= self._q(j, p[j], bias)
            cum += surv
            exp[j + 1] = cum
        return exp

    def _choose(self, exps: list[dict[int, float]], costs: dict[int, float], n: int) -> int:
        e = {k: sum(x[k] for x in exps) for k in CAPS}
        if self.ctl.policy == "ratio":
            return max(CAPS, key=lambda k: (e[k] / costs[k], k))
        lam = self.lam.get(n)
        if lam is None:
            lam = max(e[k] / costs[k] for k in CAPS)
        k = max(CAPS, key=lambda k: (e[k] - lam * costs[k], k))
        a = self.ctl.alpha
        self.lam[n] = (1 - a) * lam + a * e[k] / costs[k]
        return k

    # -- runtime control and live periods (rank 0) ------------------------------------
    def _poll_control(self) -> None:
        if not self.ctl.poll() or not self.ctl.costs_file:
            return
        path = os.path.join(os.path.dirname(self.ctl.path), self.ctl.costs_file)
        try:
            self.costs = CostCurve(json.loads(open(path).read()))
            self.lam.clear()
            logger.info("Verify cap: cost tables reloaded from %s (n=%s)", path, sorted(self.costs.curves))
        except Exception as e:  # noqa: BLE001
            logger.warning("Verify cap: keeping the current cost table; %s: %s", path, e)

    def _note_period(self, now: float, n: int, k: int, ctx: int) -> None:
        src = getattr(self, "ctl_src", None) or self.ctl
        if src.periods_epoch != self._periods_epoch:   # a new A/B arm: forget the old periods
            self._periods_epoch = src.periods_epoch
            self.periods.clear()
        prev, self.prev_decision = self.prev_decision, (now, self.step, n, k, ctx)
        if prev is None:
            return
        t0, step0, n0, k0, ctx0 = prev
        dt = now - t0
        if self.step != step0 + 1 or dt > 1.0:   # another step ran in between, or idle
            return
        band = sum(ctx0 >= b for b in CTX_BANDS)
        self.periods.setdefault((n0, k0, band), deque(maxlen=400)).append(dt * 1000.0)

    def _dump_periods(self) -> None:
        if self.ctl.path is None or not self.periods:
            return
        out = {}
        for (n, k, band), v in sorted(self.periods.items()):
            out[f"n{n}_k{k}_ctx{band}"] = {"n": n, "k": k, "ctx_band": band, "count": len(v),
                                          "median_ms": round(statistics.median(v), 2),
                                          "mean_ms": round(sum(v) / len(v), 2)}
        path = os.path.join(os.path.dirname(self.ctl.path), "periods.json")
        try:
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"time": time.time(), "ctx_bands": CTX_BANDS, "periods": out}, f, indent=1)
            os.replace(tmp, path)
        except OSError as e:
            logger.warning("Verify cap: cannot write %s: %s", path, e)

    # -- before the next target step ---------------------------------------------------
    def _batch(self, so):
        """[(req_id, drafts)] when the step is 1..batch_max greedy decode requests that
        each carry exactly 7 drafts; None otherwise. Depends only on the scheduler
        output, so every rank reaches the same answer."""
        n = len(so.num_scheduled_tokens)
        if not 1 <= n <= self.batch_max or so.has_structured_output_requests:
            return None
        out = []
        for rid, ntok in so.num_scheduled_tokens.items():
            spec = so.scheduled_spec_decode_tokens.get(rid)
            if not spec or len(spec) != 7 or ntok != 8 or not self.eligible.get(rid):
                return None
            out.append((rid, list(spec)))
        return out

    @staticmethod
    def _apply(so, batch: list, k: int) -> None:
        for rid, spec in batch:
            so.scheduled_spec_decode_tokens[rid] = spec[:k]
            so.num_scheduled_tokens[rid] = 1 + k
        so.total_num_scheduled_tokens = len(batch) * (1 + k)

    def _predict(self, batch: list) -> int:
        if len(batch) == 1:
            src = self.hist.get(batch[0][0]) or self.recent
        else:
            src = self.hist_n.get(len(batch))
        if not src:
            return 7
        c = Counter(src)
        return max(CAPS, key=lambda k: (c[k], k))

    def pre_trim(self, so, runner) -> None:
        """Apply the predicted cap (no host sync) so step preparation overlaps the GPU.
        With local decide + early, when the draft is already complete, apply the actual cap."""
        self.pending = None
        batch = self._batch(so)
        if batch is None:
            return
        self._t_pre = time.perf_counter()
        k_hat = self._predict(batch)
        early = None
        if self.local and self.early and self.event is not None and self.event.query():
            # Same inputs and state as at resolve, so the same cap on every rank whichever
            # of the two places a rank decides in.
            early = self._decide(batch, runner)
            k_hat = early[0]
        self.pending = (batch, k_hat, early)
        if k_hat < 7:
            self._apply(so, batch, k_hat)

    def _decide(self, batch: list, runner) -> tuple[int, int]:
        """Rank 0 (every rank with local decide): the cap for this step, and the batch's
        mean context."""
        if not self.local:                     # local decide: control changes come at sync points
            self._poll_control()
        ctl, n = self.ctl, len(batch)
        slots = [runner.req_states.req_id_to_index.get(rid) for rid, _ in batch]
        if any(s is None for s in slots):
            return 7, 0
        ctx = int(sum(int(runner.req_states.num_computed_tokens_np[s]) for s in slots) / n)
        if ctl.mode == "off" or (n > 1 and not ctl.batch):
            return 7, ctx
        if self.event is None or any(self.slot_step[s] != self.step for s in slots):
            return 7, ctx
        ts = time.perf_counter()
        self.event.synchronize()
        self.t_sync += time.perf_counter() - ts
        exps, states = [], []
        for (rid, _), s in zip(batch, slots):
            st = self.req.get(rid)
            if st is not None and st["step"] == self.step - 1:
                self._learn(st, int(self.acc_cpu[s]))
            elif st is None:
                st = {"bias": 0.0}
            p = self.p_cpu[s].tolist()
            exps.append(self._expected(p, self.bias_global + st["bias"]))
            states.append((rid, st, p))
        if ctl.mode == "fixed":
            k = ctl.fixed_k
        else:
            costs = self.costs.at(ctx, n)
            k = 7 if costs is None else self._choose(exps, costs, n)
        for rid, st, p in states:
            st.update(p=p, k=k, step=self.step)
            self.req[rid] = st
        return k, ctx

    def resolve(self, so, runner) -> bool:
        """Decide the actual cap from the current draft. True if `so` changed and the
        step must be prepared again."""
        if self.pending is None:
            return False
        batch, k_hat, early = self.pending
        self.pending = None
        n = len(batch)
        k, ctx = 7, 0
        t0 = time.perf_counter()
        self.t_prep += t0 - self._t_pre
        self._open = True
        if self.local:
            return self._resolve_local(so, runner, batch, k_hat, early, t0)
        rank0 = self.tp.rank_in_group == 0
        if rank0:
            k, ctx = self._decide(batch, runner)
        if self.tp.world_size > 1:
            tb = time.perf_counter()
            self.k_buf[0] = k
            torch.distributed.broadcast(self.k_buf, src=self.tp.ranks[0], group=self.tp.cpu_group)
            k = int(self.k_buf[0])
            self.t_bcast += time.perf_counter() - tb
        if n == 1:
            self.hist.setdefault(batch[0][0], deque(maxlen=8)).append(k)
            self.recent.append(k)
        else:
            self.hist_n.setdefault(n, deque(maxlen=8)).append(k)
        self.counts[(n, k)] += 1
        redo = k != k_hat
        if redo:
            self.mispredicts += 1
            self._apply(so, batch, k)
        self.t_total += time.perf_counter() - t0
        if rank0:
            now = time.monotonic()
            self._note_period(now, n, k, ctx)
            if now - self.last_log > 60:
                self._log_window(now)
            if now - self.last_dump > 30:
                self._dump_periods()
                self.last_dump = now
        return redo

    def _resolve_local(self, so, runner, batch: list, k_hat: int, early, t0: float) -> bool:
        """resolve() with local decide: the same decision on every rank, no host collective.
        Logging, periods and the sync point run in after_launch, off the critical path."""
        n = len(batch)
        if early is not None:
            k, ctx = early
            self.n_early += 1
        else:
            k, ctx = self._decide(batch, runner)
        if n == 1:
            self.hist.setdefault(batch[0][0], deque(maxlen=8)).append(k)
            self.recent.append(k)
        else:
            self.hist_n.setdefault(n, deque(maxlen=8)).append(k)
        self.counts[(n, k)] += 1
        redo = k != k_hat
        if redo:
            self.mispredicts += 1
            self._apply(so, batch, k)
        self.n_dec += 1
        self._hash = zlib.crc32(self._fingerprint(batch, n, k, ctx), self._hash)
        self._last = (n, k, ctx)
        self.t_total += time.perf_counter() - t0
        return redo

    # -- around the target launch (model_runner) -------------------------------------
    def before_launch(self) -> None:
        """Just before the verify-cap step's target launch: host-time instrumentation."""
        if not self._open:
            return
        now = time.perf_counter()
        self.t_host += now - self._t_pre
        if self._t_record is not None:          # host time since the drafter was launched
            self.t_window += now - self._t_record
            self.n_window += 1
            self._t_record = None
        if self.gap is not None:
            self.gap.before_launch()

    def after_launch(self) -> None:
        """Right after the verify-cap step's target launch, while the GPU runs the verify:
        local decide's bookkeeping, logging and sync point."""
        if not self._open:
            return
        self._open = False
        if self.gap is not None:
            self.gap.after_launch()
        if self._last is None:
            return
        n, k, ctx = self._last
        self._last = None
        if self.tp.rank_in_group == 0:
            now = time.monotonic()
            self._note_period(now, n, k, ctx)
            if now - self.last_log > 60:
                self._log_window(now)
            if now - self.last_dump > 30:
                self._dump_periods()
                self.last_dump = now
        if self.local and self.n_dec % self.sync_every == 0:
            self._sync_point()

    def _log_window(self, now: float) -> None:
        """Rank 0, about once a minute: the decision mix and the host costs per decision."""
        total = sum(self.counts.values())
        per = 1000.0 / max(1, total)
        by_n = {m: {c: self.counts[(m, c)] for c in CAPS if self.counts[(m, c)]}
                for m in sorted({m for m, _ in self.counts})}
        post = self.t_host - self.t_prep - self.t_total - self.t_redo
        if self.check_fail is not None:
            decide = "rank0 (FALLBACK)"
            check = f"MISMATCH at decision {self.check_fail[0]} (crc32 {self.check_fail[1]})"
        elif self.local:
            decide = "local"
            check = f"ok {self.checks}/{self.checks} (every {self.sync_every}, decision {self.n_dec})"
        else:
            decide, check = "rank0", "off"
        if self.drift is not None:             # since boot: rows whose own p_sel differed from rank 0's
            check += ", own p_sel differed from rank 0's in " + ", ".join(
                f"{d}/{r}" for d, r in self.drift) + " drafted rows per TP rank"
        logger.info("Verify cap: %d decisions in the last minute by batch size %s, global bias "
                    "%.2f, lambda %s; per decision: wait %.2f ms, broadcast %.2f ms, resolve "
                    "total %.2f ms; re-prepared %.0f%% (%.2f ms per re-preparation); host per "
                    "decision: prep %.2f ms, post %.2f ms, pre_trim->launch %.2f ms, draft launch->"
                    "verify launch %.2f ms; decide %s, early %.0f%%, device broadcast %.3f ms, sync "
                    "%.3f ms; rank check %s; gpu %s",
                    total, by_n, self.bias_global,
                    {m: round(v, 4) for m, v in sorted(self.lam.items())},
                    self.t_sync * per, self.t_bcast * per, self.t_total * per,
                    100.0 * self.mispredicts / max(1, total),
                    1000.0 * self.t_redo / max(1, self.mispredicts),
                    self.t_prep * per, post * per, self.t_host * per,
                    1000.0 * self.t_window / max(1, self.n_window),
                    decide, 100.0 * self.n_early / max(1, total), self.t_dbcast * per,
                    self.t_syncpt * per, check, self.gap.summary() if self.gap is not None else "off")
        self.counts.clear()
        self.t_sync = self.t_bcast = self.t_total = self.t_redo = 0.0
        self.t_prep = self.t_host = self.t_window = self.t_dbcast = self.t_syncpt = 0.0
        self.mispredicts = self.n_window = self.n_early = 0
        if self.gap is not None:
            self.gap.reset()
        self.last_log = now

    def trim(self, so, runner) -> None:
        """pre_trim + resolve (+ the launch hooks) in one call (tests and non-overlapped use)."""
        self.pre_trim(so, runner)
        self.resolve(so, runner)
        self.before_launch()
        self.after_launch()
