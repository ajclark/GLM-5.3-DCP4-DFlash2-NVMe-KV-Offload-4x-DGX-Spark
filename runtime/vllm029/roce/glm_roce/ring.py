"""Ring transport for RoCEnante: one-shot TP all-reduce on a four-node ring with no switch.

The cluster's TP ranks are cabled as a ring (06c4 - 365c - ddbf - a218 - 06c4, ring
order = TP rank order); b12x RoCEnante needs a link to every peer, which only the DCP
pairs have. ``RingAllReduce`` keeps b12x's pinned region, doorbell and CuTe kernels
unchanged (one flag per source, i.e. the kernels' ``hca_count = 1`` layout) and swaps
the transport for ``_ring_proxy.c``: every rank writes its payload to both neighbours
and forwards its ccw neighbour's payload to its cw neighbour, so each rank holds all
four inputs after at most two hops. The reduction is b12x's: fp32 accumulation in
fixed rank order, so every rank stores identical bits (not NCCL's bits: NCCL's ring
sums in a different order).

Which RDMA device faces which neighbour is discovered, not configured: each rank
publishes its active devices' IPv4 GIDs, and the link of every ring edge is the device
pair whose addresses share a /24 (point-to-point subnets), preferring devices not in
``GLM_ROCE_RING_EXCLUDE`` (default: ``B12X_ROCE_HCA``, the DCP pairs' device).
``GLM_ROCE_RING_HCAS=cw,ccw`` overrides the discovery on a node.

``GLM_ROCE_RING_SPLIT=1``: split mode. Each payload is cut into b12x's two stripes;
stripe 0 travels clockwise and stripe 1 counter-clockwise (the kernels then wait for
two flags per source, ``hca_count = 2``), so both directions of every link carry 1.5
payloads per op instead of 2 and 1. ``GLM_ROCE_RING_CHUNKS=C`` (default 1) adds cut-through:
each direction's share is cut into C chunks with a flag each, and the forwarding proxy passes
every chunk on as soon as it lands instead of waiting for the whole payload (the kernels then
wait for (split ? 2 : 1) * C flags per source). ``GLM_ROCE_RING_BLOCKS`` sets the kernels'
grid (power of two, default 8 as in b12x).
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import hashlib
import ipaddress
import logging
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

logger = logging.getLogger("vllm.glm_roce")

RING_WORLD = 4
RING_ABI_VERSION = 3
_SOURCE = Path(__file__).with_name("_ring_proxy.c")
_LOCK = threading.Lock()
_LIB: Optional[ctypes.CDLL] = None
_CFLAGS = ("-O2", "-std=gnu11", "-Wall", "-Wextra", "-Werror", "-fstack-protector-strong",
           "-D_FORTIFY_SOURCE=2", "-shared", "-fPIC")
# Diagnostic build with per-op proxy timestamps (-DRING_TRACE); never the production library.
TRACE = os.getenv("GLM_ROCE_RING_TRACE", "0").strip() == "1"


def _cache_dir() -> Path:
    override = os.getenv("GLM_ROCE_RING_CACHE_DIR") or os.getenv("B12X_ROCE_CACHE_DIR")
    if override:
        return Path(override)
    return Path(os.path.expanduser("~")) / ".cache" / "glm_roce"


def library_path() -> Path:
    """The compiled proxy for the current source (built at image build; else built here)."""
    digest = hashlib.sha256(_SOURCE.read_bytes()).hexdigest()[:16]
    name = f"ring_proxy-{digest}{'-trace' if TRACE else ''}.so"
    for root in (_cache_dir(), Path(os.path.expanduser("~")) / ".cache" / "glm_roce"):
        if (root / name).exists():
            return root / name
    cc = next((c for c in (os.getenv("CC"), "gcc", "cc") if c and shutil.which(c)), None)
    if cc is None:
        raise RuntimeError("the ring proxy needs a C compiler and libibverbs headers")
    target_dir = _cache_dir()
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        probe = target_dir / f".w{os.getpid()}"
        probe.touch()
        probe.unlink()
    except OSError:
        target_dir = Path(os.path.expanduser("~")) / ".cache" / "glm_roce"
        target_dir.mkdir(parents=True, exist_ok=True)
    tmp = target_dir / f".{name}.{os.getpid()}"
    cmd = [cc, *_CFLAGS, *(("-DRING_TRACE",) if TRACE else ()), "-o", str(tmp), str(_SOURCE),
           "-libverbs", "-lpthread"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("failed to build the ring proxy: " + " ".join(cmd) + "\n" + proc.stderr)
    os.replace(tmp, target_dir / name)
    return target_dir / name


def load() -> ctypes.CDLL:
    global _LIB
    with _LOCK:
        if _LIB is not None:
            return _LIB
        lib = ctypes.CDLL(str(library_path()), use_errno=True)
        u64, p = ctypes.c_uint64, ctypes.c_void_p
        lib.ring_abi_version.restype = ctypes.c_int
        lib.ring_abi_version.argtypes = []
        lib.ring_layout.restype = ctypes.c_int
        lib.ring_layout.argtypes = [ctypes.c_int, u64, ctypes.POINTER(u64)]
        lib.ring_blob_bytes.restype = u64
        lib.ring_blob_bytes.argtypes = []
        lib.ring_create.restype = p
        lib.ring_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int, p, u64, u64,
                                    ctypes.c_char_p, u64]
        lib.ring_local_blob.restype = ctypes.c_int
        lib.ring_local_blob.argtypes = [p, p, u64]
        lib.ring_connect.restype = ctypes.c_int
        lib.ring_connect.argtypes = [p, p, u64]
        lib.ring_start.restype = ctypes.c_int
        lib.ring_start.argtypes = [p]
        lib.ring_stop.restype = None
        lib.ring_stop.argtypes = [p]
        lib.ring_failed.restype = ctypes.c_int
        lib.ring_failed.argtypes = [p]
        lib.ring_error.restype = ctypes.c_char_p
        lib.ring_error.argtypes = [p]
        lib.ring_stat.restype = u64
        lib.ring_stat.argtypes = [p, ctypes.c_int]
        lib.ring_hca_stat.restype = u64
        lib.ring_hca_stat.argtypes = [p, ctypes.c_int, ctypes.c_int]
        lib.ring_destroy.restype = None
        lib.ring_destroy.argtypes = [p]
        if TRACE:
            lib.ring_trace_record_bytes.restype = u64
            lib.ring_trace_record_bytes.argtypes = []
            lib.ring_trace_dump.restype = u64
            lib.ring_trace_dump.argtypes = [p, p, u64]
        if lib.ring_abi_version() != RING_ABI_VERSION:
            raise RuntimeError("unexpected ring proxy ABI version")
        _LIB = lib
        return lib


class RingLayout:
    """Byte offsets of the pinned region (identical to b12x's ``Layout``)."""

    def __init__(self, world_size: int, slot_bytes: int) -> None:
        out = (ctypes.c_uint64 * 7)()
        if load().ring_layout(int(world_size), int(slot_bytes), out) != 0:
            raise ValueError(f"unsupported ring geometry: world={world_size} slot_bytes={slot_bytes}")
        (self.recv_off, self.flag_off, self.send_off, self.ctrl_off, self.total_bytes,
         self.flag_stride, self.slots) = (int(v) for v in out)


class RingProxy:
    """One rank's ring proxy context (same surface as b12x ``Proxy``)."""

    def __init__(self, *, rank: int, hca_cw: str, hca_ccw: str, gid_index: int, region_ptr: int,
                 region_bytes: int, slot_bytes: int, split: bool = False, chunks: int = 1) -> None:
        self._lib = load()
        err = ctypes.create_string_buffer(512)
        self._ctx = self._lib.ring_create(RING_WORLD, int(rank), hca_cw.encode(), hca_ccw.encode(),
                                          int(gid_index), int(bool(split)), int(chunks),
                                          ctypes.c_void_p(int(region_ptr)),
                                          int(region_bytes), int(slot_bytes), err, len(err))
        if not self._ctx:
            raise RuntimeError(f"ring proxy setup failed: {err.value.decode(errors='replace')}")
        self.rank = int(rank)
        self.hca_names = (hca_cw, hca_ccw)

    def local_blob(self) -> bytes:
        n = int(self._lib.ring_blob_bytes())
        buf = ctypes.create_string_buffer(n)
        if self._lib.ring_local_blob(self._ctx, buf, n) != 0:
            raise RuntimeError("ring proxy blob size mismatch")
        return buf.raw

    def connect(self, blobs: list[bytes]) -> None:
        n = int(self._lib.ring_blob_bytes())
        if len(blobs) != RING_WORLD or any(len(b) != n for b in blobs):
            raise RuntimeError("ring proxy blob size mismatch")
        joined = b"".join(blobs)
        buf = ctypes.create_string_buffer(joined, len(joined))
        if self._lib.ring_connect(self._ctx, buf, len(joined)) != 0:
            raise RuntimeError(f"ring queue-pair connect failed: {self.error()}")

    def start(self) -> None:
        if self._lib.ring_start(self._ctx) != 0:
            raise RuntimeError(f"ring proxy thread failed to start: {self.error()}")

    def stop(self) -> None:
        self._lib.ring_stop(self._ctx)

    def failed(self) -> bool:
        return bool(self._lib.ring_failed(self._ctx))

    def error(self) -> str:
        raw = self._lib.ring_error(self._ctx)
        return raw.decode(errors="replace") if raw else ""

    def stats(self) -> dict[str, Any]:
        s = lambda i: int(self._lib.ring_stat(self._ctx, i))  # noqa: E731
        return {
            "ops_posted": s(0),
            "writes_completed": s(1),
            "last_seq": s(2),
            "forwards": s(3),
            "forwards_done": s(4),
            "split": bool(s(5)),
            "chunks": s(6),
            "ring_hcas": {"cw": self.hca_names[0], "ccw": self.hca_names[1]},
            "bytes_posted_per_hca": [int(self._lib.ring_hca_stat(self._ctx, h, 1)) for h in range(2)],
        }

    def trace(self):
        """Diagnostic build only: the per-op timestamp records as a numpy structured array."""
        import numpy as np

        if not TRACE:
            raise RuntimeError("set GLM_ROCE_RING_TRACE=1 for the trace build")
        dt = np.dtype([("seq", "<u4"), ("nbytes", "<u4"), ("t_db", "<u8"), ("t_own_post", "<u8"),
                       ("t_own_cqe", "<u8", (2,)), ("t_arr", "<u8", (RING_WORLD,)),
                       ("t_first", "<u8", (RING_WORLD,)),
                       ("t_fwd_post", "<u8"), ("t_fwd_cqe", "<u8")])
        if dt.itemsize != int(self._lib.ring_trace_record_bytes()):
            raise RuntimeError("ring trace record size mismatch")
        n = 16384
        buf = ctypes.create_string_buffer(n * dt.itemsize)
        got = int(self._lib.ring_trace_dump(self._ctx, buf, n))
        return np.frombuffer(buf.raw[: got * dt.itemsize], dtype=dt).copy()

    def close(self) -> None:
        ctx, self._ctx = self._ctx, None
        if ctx:
            self._lib.ring_destroy(ctx)

    def __del__(self) -> None:  # pragma: no cover
        with contextlib.suppress(Exception):
            self.close()


# -- link discovery ------------------------------------------------------------------------


def _gid_ipv4(dev: Path, gid_index: int) -> Optional[str]:
    """IPv4 address of an IPv4-mapped RoCE v2 GID, or None."""
    try:
        if "ACTIVE" not in (dev / "ports" / "1" / "state").read_text():
            return None
        raw = (dev / "ports" / "1" / "gids" / str(gid_index)).read_text().strip()
    except OSError:
        return None
    try:
        addr = ipaddress.IPv6Address(raw)
    except ValueError:
        return None
    mapped = addr.ipv4_mapped
    return str(mapped) if mapped is not None and int(mapped) != 0 else None


def local_hcas(gid_index: int, root: str = "/sys/class/infiniband") -> dict[str, str]:
    """Active RDMA devices with an IPv4 GID at ``gid_index``: name -> address."""
    out = {}
    for dev in sorted(Path(root).glob("*")):
        ip = _gid_ipv4(dev, gid_index)
        if ip is not None:
            out[dev.name] = ip
    return out


def choose_links(tables: Sequence[dict[str, str]], exclude: Sequence[frozenset[str]],
                 overrides: Sequence[Optional[tuple[str, str]]] = ()) -> list[tuple[str, str]]:
    """``(cw, ccw)`` device names for every rank, from every rank's device table.

    Edge r -> r+1 uses the device pair (a on r, b on r+1) whose IPv4 addresses share a
    /24; pairs where either side is excluded on its node are used only if nothing else
    links the two. Deterministic, so every rank computes the same answer.
    """
    world = len(tables)
    edge: list[tuple[str, str]] = []
    for r in range(world):
        n = (r + 1) % world
        cands = []
        for a, ia in sorted(tables[r].items()):
            for b, ib in sorted(tables[n].items()):
                if ipaddress.ip_network(f"{ia}/24", strict=False) == ipaddress.ip_network(f"{ib}/24", strict=False):
                    penalty = int(a in exclude[r]) + int(b in exclude[n])
                    cands.append((penalty, a, b))
        if not cands:
            raise RuntimeError(f"no RDMA link between ring ranks {r} and {n}: {tables[r]} / {tables[n]}")
        cands.sort()
        edge.append((cands[0][1], cands[0][2]))
    links = [(edge[r][0], edge[(r - 1) % world][1]) for r in range(world)]
    for r, o in enumerate(overrides):
        if o is not None:
            links[r] = o
    return links


def _env_override() -> Optional[tuple[str, str]]:
    raw = os.getenv("GLM_ROCE_RING_HCAS", "").strip()
    if not raw:
        return None
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    if len(parts) != 2:
        raise ValueError(f"GLM_ROCE_RING_HCAS must be 'cw,ccw', got {raw!r}")
    return parts[0], parts[1]


def _env_exclude() -> frozenset[str]:
    raw = os.getenv("GLM_ROCE_RING_EXCLUDE")
    if raw is None:
        raw = os.getenv("B12X_ROCE_HCA", "")
    return frozenset(x.strip().lstrip("=^").split(":")[0] for x in raw.split(",") if x.strip())


# -- the runtime -------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _ring_runtime_class():
    import torch
    from b12x.comm.roce import roce_oneshot as ro

    class RingAllReduce(ro.RoceOneshotAllReduce):
        """b12x RoCEnante with the ring transport (TP=4 on a switchless ring)."""

        algorithm = "rocenante-ring"

        def __init__(self, *, exchange_group: ProcessGroup, device, max_size: int,
                     max_gather_bytes: int = 0, gid_index: Optional[int] = None,
                     threads: int = ro.DEFAULT_THREADS, blocks: Optional[int] = None,
                     split: Optional[bool] = None, chunks: Optional[int] = None) -> None:
            import torch.distributed as dist

            if blocks is None:
                blocks = int(os.getenv("GLM_ROCE_RING_BLOCKS", str(ro.DEFAULT_BLOCKS)))
            if split is None:
                split = os.getenv("GLM_ROCE_RING_SPLIT", "0").strip() == "1"
            if chunks is None:
                chunks = int(os.getenv("GLM_ROCE_RING_CHUNKS", "1"))
            if chunks < 1 or (2 if split else 1) * chunks > 8:
                raise ValueError("GLM_ROCE_RING_CHUNKS: (split ? 2 : 1) * chunks must be 1..8")
            self.chunks = int(chunks)
            if blocks < 1 or blocks & (blocks - 1):
                raise ValueError("GLM_ROCE_RING_BLOCKS must be a power of two")
            self.split = bool(split)
            self.device = ro._normalize_device(device)
            self.rank = dist.get_rank(group=exchange_group)
            self.world_size = dist.get_world_size(group=exchange_group)
            self._group = exchange_group
            self._closed = False
            self._lock = threading.Lock()
            self._proxy = None
            self._gather_buffers = None
            self._align_buffers = None
            self._stream_event = torch.cuda.Event()
            self._last_stream = None
            self._capture_stream = None
            self._capture_id = 0
            if self.world_size != RING_WORLD:
                raise ValueError(f"the ring transport needs world size {RING_WORLD}, got {self.world_size}")
            if int(max_size) < ro.PACK_BYTES:
                raise ValueError("max_size must hold at least one 16-byte pack")
            self.max_size = int(max_size)
            self.max_gather_bytes = int(max_gather_bytes)
            self._threads = int(threads)
            self._blocks = int(blocks)
            self._counter_classes = self._blocks.bit_length()
            self.gid_index = ro.default_gid_index() if gid_index is None else int(gid_index)
            self.spin_limit = ro._env_int("B12X_ROCE_SPIN_LIMIT", default=ro.DEFAULT_SPIN_LIMIT)

            # Links: every rank publishes its device table, everyone picks the same links.
            error: Optional[str] = None
            try:
                table, excl, override = local_hcas(self.gid_index), _env_exclude(), _env_override()
            except Exception as exc:  # noqa: BLE001
                table, excl, override, error = {}, frozenset(), None, str(exc)
            infos = ro._exchange((error, table, excl, override), exchange_group)
            errs = [f"rank {i}: {e}" for i, (e, *_rest) in enumerate(infos) if e]
            if errs:
                raise RuntimeError("ring link discovery failed: " + "; ".join(errs))
            links = choose_links([i[1] for i in infos], [i[2] for i in infos], [i[3] for i in infos])
            self.links = links
            cw, ccw = links[self.rank]
            # One flag per piece and source: the kernels run with hca_count = pieces.
            pieces = (2 if self.split else 1) * self.chunks
            self.hca_names = tuple(f"ring:{cw}>{ccw}#{p}" for p in range(pieces))

            slot_bytes = ro._align_up(max(self.max_size, self.max_gather_bytes), ro._SLOT_ALIGNMENT)
            self._layout = RingLayout(self.world_size, slot_bytes)
            self._slot_bytes = slot_bytes
            with torch.cuda.device(self.device):
                self._region = torch.zeros(self._layout.total_bytes, dtype=torch.uint8, pin_memory=True)
                self._counters = torch.zeros(2 + 2 * self._counter_classes, dtype=torch.int32,
                                             device=self.device)
            host_ptr = self._region.data_ptr()
            if self._device_pointer(host_ptr) != host_ptr:
                raise RuntimeError("the ring all-reduce needs directly device-accessible host pointers")
            self._recv_base = host_ptr + self._layout.recv_off
            self._flag_base = host_ptr + self._layout.flag_off
            self._send_base = host_ptr + self._layout.send_off
            self._ctrl_base = host_ptr + self._layout.ctrl_off
            self._ctrl_words = self._region[self._layout.ctrl_off: self._layout.ctrl_off + 28].view(torch.int32)
            self._error_word = self._ctrl_words[2:3]
            self._ctrl_np = self._ctrl_words.numpy()
            self._epoch_address = self._counters.data_ptr()
            self._poison_address = self._epoch_address + 4 * (1 + 2 * self._counter_classes)

            blob = b""
            try:
                self._proxy = RingProxy(rank=self.rank, hca_cw=cw, hca_ccw=ccw, gid_index=self.gid_index,
                                        region_ptr=host_ptr, region_bytes=self._layout.total_bytes,
                                        slot_bytes=slot_bytes, split=self.split, chunks=self.chunks)
                blob = self._proxy.local_blob()
            except Exception as exc:  # noqa: BLE001 - reported collectively
                error = str(exc)
            config = {
                "api_version": ro.API_VERSION,
                "ring_abi": load().ring_abi_version() if error is None else None,
                "world_size": self.world_size, "slot_bytes": slot_bytes, "slots": self._layout.slots,
                "flag_stride": self._layout.flag_stride, "max_size": self.max_size,
                "max_gather_bytes": self.max_gather_bytes, "spin_limit": self.spin_limit,
                "threads": self._threads, "blocks": self._blocks, "links": links, "split": self.split, "chunks": self.chunks,
            }
            statuses = ro._exchange((error, blob, config), exchange_group)
            failures = [f"rank {i}: {s[0]}" for i, s in enumerate(statuses) if s[0] is not None]
            if not failures:
                ref = statuses[0][2]
                for i, s in enumerate(statuses):
                    diff = {k: (ref[k], s[2].get(k)) for k in ref if s[2].get(k) != ref[k]}
                    if diff:
                        failures.append(f"rank {i} configuration differs from rank 0: {diff}")
            if failures:
                self.close()
                raise RuntimeError("ring all-reduce setup failed: " + "; ".join(failures))
            try:
                self._proxy.connect([s[1] for s in statuses])
                self._proxy.start()
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
            verdicts = ro._exchange(error, exchange_group)
            failures = [f"rank {i}: {v}" for i, v in enumerate(verdicts) if v is not None]
            if failures:
                self.close()
                raise RuntimeError("ring all-reduce connect failed: " + "; ".join(failures))
            logger.info("GLM_ROCE_RING ready rank=%d cw=%s ccw=%s links=%s max_size=%d split=%d chunks=%d blocks=%d",
                        self.rank, cw, ccw, links, self.max_size, int(self.split), self.chunks, self._blocks)

        def stats(self) -> dict[str, Any]:
            info = super().stats()
            info["algorithm"] = self.algorithm
            info["links"] = self.links
            info["blocks"] = self._blocks
            return info

    return RingAllReduce


def RingAllReduce(**kwargs):  # noqa: N802 - factory with the class's name
    return _ring_runtime_class()(**kwargs)
