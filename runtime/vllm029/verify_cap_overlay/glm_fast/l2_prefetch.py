# SPDX-License-Identifier: Apache-2.0
"""VLLM_L2_PREFETCH: warm the next dense weights into the GB10 L2 while a collective runs. Exact.

During a TP all-reduce (NCCL ring today, ~70-100 us per call at decode sizes; the RoCE ring
later) or the DCP chain (query all-gather -> sparse attention -> LSE gather + reduce-scatter),
the GPU's DRAM is nearly idle. A side stream forks right before the collective and issues
`cp.async.bulk.prefetch.L2` for the first MiB of the weights that are read right after it. The
prefetch kernel runs <= 8 CTAs and finishes in microseconds (the instruction is fire-and-
forget); the data streams into L2 during the collective. Weights are static and a prefetch
is only a cache hint, so every output is bit-identical by construction.

Windows (GLM-5.3 = DeepseekV32Model in vllm/models/deepseek_v32/nvidia/model.py; 78 layers):
  B  the post-attention all-reduce (`fused_allreduce_rms_norm(..., post_attention_layernorm)`)
     -> the layer's router `mlp.gate.weight` (bf16, 3.1 MB), shared expert gate_up (int8
     Marlin, 6.3 MB + scales) and down (3.1 MB); for the dense layers 0-2, `mlp.gate_up_proj`.
  C  the MoE all-reduce, which vLLM fuses into the NEXT layer's input norm
     (`fused_allreduce_rms_norm(..., input_layernorm)`) -> that layer's `fused_qkv_a_proj`
     (int8 Marlin, replicated, 16.1 MB), then `q_b_proj` (8.4 MB) if budget is left.
  D  the DCP chain: at `self_attn.dcp_manager.query_gather` -> `W_UV` (bf16, 4 MiB, a strided
     view: prefetched as its 16 contiguous per-head runs) then `o_proj` (int8 Marlin, 25.2 MB).
Each window has its own budget (default 12 MiB of the 24 MB L2), filled in read order with the
small tensors (scales) of a module first, then its packed weight.

Hook placement. The decoder layer calls the module-level `fused_allreduce_rms_norm` of
vllm.models.deepseek_v32.nvidia.model; this module replaces that name with a wrapper that forks
the window mapped to the `norm` argument and then calls the original, so the hook sits above
whatever all-reduce runs underneath (NCCL, glm_roce's RoCE ring, or flashinfer's fused path).
Window D wraps each layer's `dcp_manager.query_gather` (an instance attribute). The target model
forward (`DeepseekV32Model.forward`) joins the side stream at its end, long after every prefetch
finished, so no main-stream kernel waits on it.

CUDA graphs. Fork = `side.wait_stream(main)` + launch on `side`; join = `main.wait_stream(side)`:
both are legal in stream capture, so the prefetch kernels become a side branch of every
captured graph. Tables (int64 [n, 2] {address, bytes}) and the kernel are prepared eagerly:
the kernel is loaded and launched once when the model is linked (first eager forward), and each
layer's tables are built on the eager warm-up forward that MRv2 runs before every capture. A
graph captured without a table simply prefetches nothing for that window. Eager forwards do not
fork unless VLLM_L2_PREFETCH_EAGER=1 (the Python fork costs ~10-20 us of CPU per window).

Knobs (read at link time):
  VLLM_L2_PREFETCH=0|1              master switch (default 0: the hook is not installed)
  VLLM_L2_PREFETCH_WINDOWS=B,C,D    which windows fork
  VLLM_L2_PREFETCH_MB_B / _MB_C / _MB_D   budgets in MiB (default 12 each)
  VLLM_L2_PREFETCH_MAXTOK=128       only forwards with <= this many tokens (decode/verify)
  VLLM_L2_PREFETCH_IMPL=auto|cuda|triton   kernel: the nvcc-built .so (auto: if present or
                                    buildable), else Triton inline asm
  VLLM_L2_PREFETCH_SO=/opt/glm-fast/libglm_l2pf.so   prebuilt kernel (image build)
  VLLM_L2_PREFETCH_CACHE=/tmp/glm_fast    where a missing .so is compiled with nvcc
  VLLM_L2_PREFETCH_CTAS=8           CTAs per prefetch launch (<= 8)
  VLLM_L2_PREFETCH_EAGER=0          also fork in eager (uncaptured) forwards
  VLLM_L2_PREFETCH_CONTROL=<path>   optional JSON file {"windows": "BD"} polled once a second
                                    from execute_model: turns captured windows on/off at runtime
                                    through a device flag each prefetch kernel reads (in-boot A/B,
                                    no recapture; per node; removing the file restores the boot set)

Credits: the idea, the kernel shape (<= 8 CTAs, 16 KiB bulk prefetches, segment table) and the
side-stream fork/join are knapcio's (knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4 @770d115,
overlay/glm_l2_prefetch.py and glm_l2_prefetch_mla.py, Apache-2.0 per their SPDX headers),
re-implemented here for the full GLM-5.3 (MLA + DSA, int8 Marlin dense, NCCL/RoCE TP
all-reduce) with collective-level windows instead of his KDA/RoCE-hook windows.
"""
from __future__ import annotations

import ctypes
import hashlib
import itertools
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger("vllm.glm_fast")

_OFF = ("", "0", "off", "false", "no")
CHUNK = 16384
WINDOW_INDEX = {"B": 0, "C": 1, "D": 2}
HERE = Path(__file__).resolve().parent
MODEL_MOD = "vllm.models.deepseek_v32.nvidia.model"


def _env(name, default):
    return os.environ.get(name, default)


def _on(name, default="0") -> bool:
    return str(_env(name, default)).strip().lower() not in _OFF


class Cfg:
    def __init__(self):
        self.windows = {w.strip().upper() for w in _env("VLLM_L2_PREFETCH_WINDOWS", "B,C,D").split(",")
                        if w.strip()}
        self.budget = {w: int(float(_env(f"VLLM_L2_PREFETCH_MB_{w}", "12")) * (1 << 20)) for w in "BCD"}
        self.maxtok = int(_env("VLLM_L2_PREFETCH_MAXTOK", "128"))
        self.impl = _env("VLLM_L2_PREFETCH_IMPL", "auto").strip().lower()
        self.so = _env("VLLM_L2_PREFETCH_SO", "/opt/glm-fast/libglm_l2pf.so")
        self.cache = _env("VLLM_L2_PREFETCH_CACHE", "/tmp/glm_fast")
        self.ctas = max(1, min(8, int(_env("VLLM_L2_PREFETCH_CTAS", "8"))))
        self.eager = _on("VLLM_L2_PREFETCH_EAGER")


# ---------------------------------------------------------------------------------------------
# Segment planning (pure; CPU-testable)
# ---------------------------------------------------------------------------------------------
def dense_runs(t, max_runs: int = 256):
    """The contiguous byte runs [(ptr, nbytes)] holding a tensor's elements, in address order,
    or None (negative strides, overlapping views, too many runs). A dense tensor is one run;
    W_UV (a [N, L, V] transpose-view into the dequantized kv_b weight) is N runs."""
    if t is None:
        return None
    n = t.numel()
    if n == 0:
        return []
    es = t.element_size()
    dims = [(int(s), int(st)) for s, st in zip(t.shape, t.stride()) if s != 1]
    if any(st <= 0 for _, st in dims):
        return None
    dims.sort(key=lambda d: d[1], reverse=True)
    expected, k = 1, len(dims)
    for j in range(len(dims) - 1, -1, -1):
        s, st = dims[j]
        if st != expected:
            break
        expected *= s
        k = j
    outer = dims[:k]
    count = 1
    for s, _ in outer:
        count *= s
    if count > max_runs:
        return None
    # outer strides must not make the inner blocks overlap
    if any(st < expected for _, st in outer):
        return None
    base = t.data_ptr()
    runs = sorted((base + es * sum(i * st for i, (_, st) in zip(idx, outer)), expected * es)
                  for idx in itertools.product(*[range(s) for s, _ in outer]))
    merged = []
    for p, b in runs:
        if merged and merged[-1][0] + merged[-1][1] == p:
            merged[-1] = (merged[-1][0], merged[-1][1] + b)
        else:
            merged.append((p, b))
    return merged


def _is_dev(t) -> bool:
    """A device tensor (CPU tensors count when GLM_FAST_TEST_CPU_TABLES=1, for the CPU tests)."""
    if t is None or not hasattr(t, "data_ptr"):
        return False
    return bool(getattr(t, "is_cuda", False)) or os.environ.get("GLM_FAST_TEST_CPU_TABLES") == "1"


def module_tensors(mod, min_bytes: int = 4096):
    """Runs of the device parameters of one linear-like module, in the order a weight-only GEMM
    consumes them: small tensors (scales, zero points) first, then the packed weight."""
    if mod is None:
        return []
    items = []
    for _name, p in mod.named_parameters(recurse=False):
        if not _is_dev(p):
            continue
        nbytes = p.numel() * p.element_size()
        if nbytes < min_bytes:
            continue
        runs = dense_runs(p)
        if runs:
            items.append((nbytes, runs))
    items.sort(key=lambda it: it[0])
    return [r for _, runs in items for r in runs]


def tensor_runs(t):
    if not _is_dev(t):
        return []
    return dense_runs(t) or []


def take(runs, budget: int):
    """[(ptr, nbytes)] in read order -> the first `budget` bytes as 16-byte-aligned pieces."""
    out, left = [], int(budget)
    for ptr, nbytes in runs:
        if left < 16:
            break
        if ptr % 16:
            continue  # cp.async.bulk needs 16-byte alignment; a skipped run is only a lost hint
        n = min(int(nbytes), left) & ~15
        if n > 0:
            out.append((int(ptr), n))
            left -= n
    return out


def find_attr_tensor(module, name):
    """The first tensor attribute `name` on `module` or a submodule (W_UV lives on the MLA layer)."""
    for m in module.modules():
        t = m.__dict__.get(name)
        if t is None:
            t = getattr(m, name, None) if hasattr(m, name) else None
        if t is not None and hasattr(t, "data_ptr"):
            return t
    return None


def queue_b(layer):
    """Window B: what the layer reads right after its post-attention all-reduce."""
    mlp = getattr(layer, "mlp", None)
    q = []
    if mlp is None:
        return q
    gate = getattr(mlp, "gate", None)
    if gate is not None and hasattr(gate, "weight"):
        q += tensor_runs(gate.weight)  # router (bf16), read first on the main stream
        shared = getattr(mlp, "shared_experts", None)
        if shared is not None:
            q += module_tensors(getattr(shared, "gate_up_proj", None))
            q += module_tensors(getattr(shared, "down_proj", None))
    else:  # dense MLP layers
        q += module_tensors(getattr(mlp, "gate_up_proj", None))
        q += module_tensors(getattr(mlp, "down_proj", None))
    return q


def queue_c(layer):
    """Window C: the layer's own attention projections, read right after the MoE all-reduce
    that is fused into its input norm."""
    attn = getattr(layer, "self_attn", None)
    if attn is None:
        return []
    q = module_tensors(getattr(attn, "fused_qkv_a_proj", None))
    idx = getattr(attn, "indexer", None)
    if idx is not None and not getattr(attn, "skip_topk", False):
        q += module_tensors(getattr(idx, "wk_weights_proj", None))
    q += module_tensors(getattr(attn, "q_b_proj", None))
    return q


def queue_d(attn):
    """Window D: read after the DCP chain: W_UV (bmm), then o_proj."""
    q = tensor_runs(find_attr_tensor(attn, "W_UV"))
    q += module_tensors(getattr(attn, "o_proj", None))
    return q


# ---------------------------------------------------------------------------------------------
# Kernel backends
# ---------------------------------------------------------------------------------------------
class _Cuda:
    name = "cuda"

    def __init__(self, cfg: Cfg):
        so = Path(cfg.so)
        if not so.exists():
            so = self._build(Path(cfg.cache))
        lib = ctypes.CDLL(str(so))
        lib.glm_l2pf_launch.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong,
                                        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        lib.glm_l2pf_launch.restype = ctypes.c_int
        self.lib, self.ctas, self.path = lib, cfg.ctas, str(so)

    @staticmethod
    def _build(cache: Path) -> Path:
        src = (HERE / "l2pf.cu").read_bytes()
        digest = hashlib.sha256(src).hexdigest()[:12]
        try:
            cache.mkdir(parents=True, exist_ok=True)
        except OSError:
            cache = Path("/tmp/glm_fast")
            cache.mkdir(parents=True, exist_ok=True)
        so = cache / f"libglm_l2pf-{digest}.so"
        if not so.exists():
            tmp = cache / f"libglm_l2pf-{digest}.{os.getpid()}.so"
            nvcc = os.environ.get("NVCC", "nvcc")
            subprocess.run([nvcc, "-O3", "-arch=sm_121a", "-shared", "-Xcompiler", "-fPIC",
                            "-o", str(tmp), str(HERE / "l2pf.cu")], check=True)
            os.replace(tmp, so)
        return so

    def launch(self, table, n: int, stream, flag=None) -> None:
        rc = self.lib.glm_l2pf_launch(ctypes.c_void_p(table.data_ptr()), n, CHUNK, self.ctas,
                                      ctypes.c_void_p(flag.data_ptr() if flag is not None else 0),
                                      ctypes.c_void_p(stream.cuda_stream))
        if rc:
            raise RuntimeError(f"glm_fast l2pf: launch failed, cuda error {rc}")


_TRITON_KERNEL = None


def _triton_kernel():
    global _TRITON_KERNEL
    if _TRITON_KERNEL is None:
        import triton
        import triton.language as tl

        @triton.jit
        def _glm_l2pf_triton(segs_ptr, n, flag_ptr, CHUNK_B: tl.constexpr, LANES: tl.constexpr):
            pid = tl.program_id(0)
            nprog = tl.num_programs(0)
            lanes = tl.arange(0, LANES).to(tl.int64)
            if tl.load(flag_ptr) != 0:
                for s in range(pid, n, nprog):
                    a = tl.load(segs_ptr + 2 * s)
                    b = tl.load(segs_ptr + 2 * s + 1)
                    for base in range(0, b, LANES * CHUNK_B):
                        off = base + lanes * CHUNK_B
                        sz = tl.minimum(b - off, CHUNK_B) & -16
                        tl.inline_asm_elementwise(
                            "{ .reg .pred p; setp.gt.s32 p, $2, 0; "
                            "@p cp.async.bulk.prefetch.L2.global [$1], $2; mov.u32 $0, 0; }",
                            "=r,l,r", [a + off, sz.to(tl.int32)], dtype=tl.int32, is_pure=False, pack=1)

        _TRITON_KERNEL = _glm_l2pf_triton
    return _TRITON_KERNEL


class _Triton:
    name = "triton"

    def __init__(self, cfg: Cfg):
        import torch
        self.kernel = _triton_kernel()
        self.ctas = cfg.ctas
        self.one = torch.ones(1, dtype=torch.int32, device="cuda")

    def launch(self, table, n: int, stream, flag=None) -> None:
        import torch
        with torch.cuda.stream(stream):
            self.kernel[(max(1, min(n, self.ctas)),)](table, n, flag if flag is not None else self.one,
                                                      CHUNK_B=CHUNK, LANES=64, num_warps=2)


def make_backend(cfg: Cfg):
    if cfg.impl in ("auto", "cuda"):
        try:
            return _Cuda(cfg)
        except Exception as exc:  # noqa: BLE001
            if cfg.impl == "cuda":
                raise
            logger.warning("glm_fast l2pf: CUDA kernel unavailable (%r); using Triton", exc)
    return _Triton(cfg)


# ---------------------------------------------------------------------------------------------
# Runtime state and hooks
# ---------------------------------------------------------------------------------------------
class State:
    cfg: Cfg | None = None
    backend = None
    side = {}            # device index -> side stream
    pending = False
    depth = 0
    norm_map = {}        # id(norm module) -> ("B" | "C", layer)
    keepalive = []
    forks = {"B": 0, "C": 0, "D": 0}
    bytes = {"B": 0, "C": 0, "D": 0}
    missing = {"B": 0, "C": 0, "D": 0}
    logged = set()
    linked_models = set()
    flags = None         # device int32 [3]: runtime on/off of the captured windows B, C, D
    control = None       # path of the runtime control file (VLLM_L2_PREFETCH_CONTROL)
    control_mtime = None
    control_checked = 0.0


def _capturing() -> bool:
    import torch
    return bool(torch.cuda.is_current_stream_capturing())


def _compiling() -> bool:
    try:
        import torch
        return bool(torch.compiler.is_compiling())
    except Exception:  # noqa: BLE001
        return False


def _rows(x) -> int:
    try:
        return int(x.shape[0])
    except Exception:  # noqa: BLE001
        return 0


def _table(segs):
    import torch
    flat = [v for s in segs for v in s]
    t = torch.tensor(flat, dtype=torch.int64, device="cuda")
    State.keepalive.append(t)  # captured graphs keep reading this address
    return (t, len(segs), sum(s[1] for s in segs))


def _plan(owner, window: str, queue_fn, target):
    """The cached table of `window` on `owner` (a module), building it when allowed."""
    key = f"_glm_l2pf_{window}"
    p = owner.__dict__.get(key)
    if p is None and not _capturing():
        segs = take(queue_fn(target), State.cfg.budget[window])
        p = _table(segs) if segs else False
        owner.__dict__[key] = p
        if window not in State.logged:
            State.logged.add(window)
            logger.info("glm_fast l2pf: window %s table: %.2f MiB in %d segments (budget %.1f MiB)",
                        window, (p[2] if p else 0) / 2**20, p[1] if p else 0,
                        State.cfg.budget[window] / 2**20)
    return p


def _fork(window: str, p) -> None:
    import torch
    main = torch.cuda.current_stream()
    side = State.side.get(main.device.index)
    if side is None:
        return
    side.wait_stream(main)
    State.backend.launch(p[0], p[1], side, State.flags[WINDOW_INDEX[window]] if State.flags is not None else None)
    State.pending = True
    State.forks[window] += 1
    State.bytes[window] += p[2]


def _maybe_fork(window: str, owner, queue_fn, target, ntok: int) -> None:
    cfg = State.cfg
    if cfg is None or window not in cfg.windows or State.depth <= 0 or not (0 < ntok <= cfg.maxtok):
        return
    if _compiling():  # never trace stream forks into a Dynamo graph (this model runs eager today)
        return
    capturing = _capturing()
    p = _plan(owner, window, queue_fn, target)
    if p is None:
        State.missing[window] += 1
        if f"miss{window}" not in State.logged:
            State.logged.add(f"miss{window}")
            logger.warning("glm_fast l2pf: window %s captured before its table was built (no prefetch "
                           "in that graph)", window)
        return
    if p and (capturing or cfg.eager):
        _fork(window, p)


def join() -> None:
    import torch
    if State.pending:
        main = torch.cuda.current_stream()
        side = State.side.get(main.device.index)
        if side is not None:
            main.wait_stream(side)
        State.pending = False


def _wrap_query_gather(attn) -> bool:
    dm = getattr(attn, "dcp_manager", None)
    qg = getattr(dm, "query_gather", None) if dm is not None else None
    if qg is None or getattr(qg, "_glm_l2pf", False):
        return False

    def query_gather(query, *args, **kwargs):
        _maybe_fork("D", attn, queue_d, attn, _rows(query))
        return qg(query, *args, **kwargs)

    query_gather._glm_l2pf = True
    query_gather.__wrapped__ = qg
    dm.query_gather = query_gather
    return True


def link(model) -> bool:
    """Map every decoder layer's norms to windows, wrap the DCP query gathers, create the side
    stream and load + warm the kernel. Eager only."""
    import torch
    if id(model) in State.linked_models:
        return True
    if _capturing():
        return False
    if State.backend is None:
        State.backend = make_backend(State.cfg)
    dev = torch.cuda.current_device()
    if dev not in State.side:
        State.side[dev] = torch.cuda.Stream(device=dev)
    if State.flags is None:
        State.flags = torch.tensor([int(w in State.cfg.windows) for w in "BCD"], dtype=torch.int32,
                                   device="cuda")
    n_layers = n_d = 0
    for layer in getattr(model, "layers", []):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue  # PPMissingLayer
        n_layers += 1
        State.norm_map[id(layer.post_attention_layernorm)] = ("B", layer)
        State.norm_map[id(layer.input_layernorm)] = ("C", layer)
        if "D" in State.cfg.windows and _wrap_query_gather(attn):
            n_d += 1
    # Load the kernel and run it once outside capture (module load, Triton JIT).
    warm = torch.zeros(16, dtype=torch.int64, device="cuda")
    tbl = torch.tensor([warm.data_ptr(), 128], dtype=torch.int64, device="cuda")
    State.backend.launch(tbl, 1, torch.cuda.current_stream())
    torch.cuda.current_stream().synchronize()
    State.linked_models.add(id(model))
    logger.info("glm_fast l2pf: linked %d layers (%d DCP query gathers), windows=%s, budgets MiB=%s, "
                "maxtok=%d, kernel=%s, ctas=%d", n_layers, n_d, sorted(State.cfg.windows),
                {w: State.cfg.budget[w] / 2**20 for w in "BCD"}, State.cfg.maxtok,
                getattr(State.backend, "path", State.backend.name), State.cfg.ctas)
    return True


def install(module) -> None:
    """Patch vllm.models.deepseek_v32.nvidia.model (called right after it executes)."""
    if getattr(module, "_glm_fast_l2pf", False):
        return
    State.cfg = Cfg()
    orig_norm = module.fused_allreduce_rms_norm

    def fused_allreduce_rms_norm(hidden_states, residual, norm):
        ent = State.norm_map.get(id(norm))
        if ent is not None:
            window, layer = ent
            _maybe_fork(window, layer, queue_b if window == "B" else queue_c, layer, _rows(hidden_states))
        return orig_norm(hidden_states, residual, norm)

    fused_allreduce_rms_norm.__wrapped__ = orig_norm
    module.fused_allreduce_rms_norm = fused_allreduce_rms_norm

    cls = module.DeepseekV32Model
    orig_forward = cls.forward

    def forward(self, *args, **kwargs):
        if id(self) not in State.linked_models:
            try:
                link(self)
            except Exception:  # noqa: BLE001 - a cache hint must never take the engine down
                logger.exception("glm_fast l2pf: link failed; L2 prefetch disabled for this process")
                State.cfg.windows = set()
                State.linked_models.add(id(self))
        State.depth += 1
        try:
            return orig_forward(self, *args, **kwargs)
        finally:
            State.depth -= 1
            if State.depth == 0:
                join()

    forward.__wrapped__ = orig_forward
    cls.forward = forward
    module._glm_fast_l2pf = True
    logger.info("glm_fast l2pf: hooked %s.fused_allreduce_rms_norm and DeepseekV32Model.forward",
                module.__name__)


def parse_control(text: str, captured) -> set | None:
    """Control-file JSON {"windows": "BD"} -> the captured windows to run ({"B", "D"}), or None
    if the file is unreadable. A window that was not captured cannot be turned on at runtime."""
    import json
    try:
        obj = json.loads(text)
        w = obj.get("windows", "") if isinstance(obj, dict) else None
        if not isinstance(w, str):
            return None
    except Exception:  # noqa: BLE001
        return None
    return {c for c in w.upper() if c in "BCD"} & set(captured)


def poll_control(now: float | None = None) -> None:
    """Re-read VLLM_L2_PREFETCH_CONTROL (at most once a second) and flip the device-side window
    flags the captured prefetch kernels read. Exact either way; per-node (no collectives)."""
    import time
    if State.control is None or State.flags is None or State.cfg is None:
        return
    now = time.monotonic() if now is None else now
    if now - State.control_checked < 1.0:
        return
    State.control_checked = now
    try:
        st = os.stat(State.control)
        mtime = (st.st_mtime_ns, st.st_size)
    except OSError:
        mtime = None
    if mtime == State.control_mtime:
        return
    State.control_mtime = mtime
    want = set(State.cfg.windows)
    if mtime is not None:
        try:
            want = parse_control(Path(State.control).read_text(), State.cfg.windows)
        except OSError:
            want = None
        if want is None:
            logger.warning("glm_fast l2pf: unreadable control file %s; windows unchanged", State.control)
            return
    import torch
    State.flags.copy_(torch.tensor([int(w in want) for w in "BCD"], dtype=torch.int32))
    logger.info("glm_fast l2pf: runtime windows now %s (captured %s; control %s)",
                "".join(sorted(want)) or "none", "".join(sorted(State.cfg.windows)), State.control)


def install_runner(module) -> None:
    """With VLLM_L2_PREFETCH_CONTROL set, poll it from GPUModelRunner.execute_model."""
    path = os.environ.get("VLLM_L2_PREFETCH_CONTROL", "").strip()
    cls = module.GPUModelRunner
    if not path or getattr(cls, "_glm_fast_l2pf_control", False):
        return
    State.control = path
    orig = cls.execute_model

    def execute_model(self, *args, **kwargs):
        try:
            poll_control()
        except Exception:  # noqa: BLE001 - a control-file problem must not stop serving
            logger.exception("glm_fast l2pf: control poll failed")
        return orig(self, *args, **kwargs)

    execute_model.__wrapped__ = orig
    cls.execute_model = execute_model
    cls._glm_fast_l2pf_control = True
    logger.info("glm_fast l2pf: runtime control file %s", path)


def stats() -> dict:
    return {"forks": dict(State.forks), "bytes": dict(State.bytes), "missing": dict(State.missing),
            "backend": getattr(State.backend, "name", None)}
