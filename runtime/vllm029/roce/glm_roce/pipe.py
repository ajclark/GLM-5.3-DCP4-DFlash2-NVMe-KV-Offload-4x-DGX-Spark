"""Pipelined prefill-size RoCE gathers (results/prefill-item2-20261006/REPORT.md, E2 pipelining; ``GLM_ROCE_PIPE=1``).

A second runtime for one DCP pair, used only for eager gathers of at least ``GLM_ROCE_PIPE_MIN_BYTES`` (default
4 MiB, i.e. prefill); decode gathers stay on the vendored runtime and its kernels, and this runtime refuses CUDA
graph capture. It has its own pinned region, queue pairs and proxy thread (``_pipe_proxy.c``, derived from the
vendored proxy), so the vendored runtime's protocol and state are untouched.

Per op (one sequence number, one send/receive slot, exactly as the vendored protocol):
1. the kernel stages the local shard into the send slot in ``n`` chunks (``chunk_geometry``); the last block to
   finish chunk ``c`` writes ``ready[slot][c] = seq`` to the control record (chunk 0 also rings the doorbell);
2. the proxy posts each chunk as soon as it is ready, striped over the HCAs, each stripe followed by its
   ``flag[src][slot][hca][c] = seq`` on the same reliable queue pair;
3. meanwhile the kernel copies the local shard into its output columns, then, chunk by chunk, waits for that
   chunk's peer flags and copies it out, while later chunks are still on the wire.
Output bytes are the same as the vendored gather's (same source bytes to the same destinations).
"""
from __future__ import annotations

import ctypes
import functools
import hashlib
import logging
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr
from b12x.comm.roce import _allgather_cute
from b12x.comm.roce._cute_intrinsics import (
    atomic_add_relaxed_gpu_u32,
    fence_sc_gpu,
    fence_sc_sys,
    ld_global_v4_u32,
    ld_relaxed_gpu_u32,
    ld_relaxed_sys_u32,
    ld_relaxed_sys_v4_u32,
    spin_until_eq_acquire_sys,
    st_global_v4_u32,
    st_release_gpu_u32,
    st_relaxed_sys_u32,
)

logger = logging.getLogger("vllm.glm_roce")

PIPE_ABI_VERSION = 103
MAX_CHUNKS = 8
CTRL_READY_WORD = 8
PACK_BYTES = 16
SLOTS = 2
_SOURCE = Path(__file__).with_name("_pipe_proxy.c")
_CFLAGS = ("-O2", "-std=gnu11", "-Wall", "-Wextra", "-Werror", "-fstack-protector-strong", "-D_FORTIFY_SOURCE=2",
           "-shared", "-fPIC")
_LOCK = threading.Lock()
_LIB: Optional[ctypes.CDLL] = None
_PREPARED: set = set()

ENV_ON = "GLM_ROCE_PIPE"
ENV_MIN_BYTES = "GLM_ROCE_PIPE_MIN_BYTES"
ENV_CHUNK_BYTES = "GLM_ROCE_PIPE_CHUNK_BYTES"


def read_config(environ=None) -> tuple[bool, int, int]:
    """``(enabled, min_bytes, chunk_bytes)``; raises on unusable values."""
    env = os.environ if environ is None else environ
    on = env.get(ENV_ON, "0").strip().lower() not in ("", "0", "off", "false", "no")
    min_bytes = int(env.get(ENV_MIN_BYTES, str(4 << 20)))
    chunk_bytes = int(env.get(ENV_CHUNK_BYTES, str(8 << 20)))
    if min_bytes < (1 << 20):
        raise ValueError(f"{ENV_MIN_BYTES}={min_bytes}: below 1 MiB would reach decode-sized (graph) gathers")
    if chunk_bytes < (1 << 20) or chunk_bytes % PACK_BYTES:
        raise ValueError(f"{ENV_CHUNK_BYTES}={chunk_bytes}: must be >= 1 MiB and a multiple of 16")
    return on, min_bytes, chunk_bytes


def chunk_geometry(nbytes: int, chunk_bytes: int) -> tuple[int, int]:
    """``(n_chunks, chunk_packs)``; identical to the proxy's ``pipe_chunks`` split."""
    n = max(1, min(MAX_CHUNKS, (int(nbytes) + int(chunk_bytes) - 1) // int(chunk_bytes)))
    total_packs = int(nbytes) // PACK_BYTES
    return n, (total_packs + n - 1) // n


# -- proxy library -------------------------------------------------------------------------


def _cache_dir() -> Path:
    override = os.getenv("GLM_ROCE_RING_CACHE_DIR") or os.getenv("B12X_ROCE_CACHE_DIR")
    return Path(override) if override else Path(os.path.expanduser("~")) / ".cache" / "glm_roce"


def library_path() -> Path:
    """The compiled proxy for the current source (built at image build; else built here)."""
    digest = hashlib.sha256(_SOURCE.read_bytes()).hexdigest()[:16]
    name = f"pipe_proxy-{digest}.so"
    for root in (_cache_dir(), Path(os.path.expanduser("~")) / ".cache" / "glm_roce"):
        if (root / name).exists():
            return root / name
    cc = next((c for c in (os.getenv("CC"), "gcc", "cc") if c and shutil.which(c)), None)
    if cc is None:
        raise RuntimeError("the pipe proxy needs a C compiler and libibverbs headers")
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
    cmd = [cc, *_CFLAGS, "-o", str(tmp), str(_SOURCE), "-libverbs", "-lpthread"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("failed to build the pipe proxy: " + " ".join(cmd) + "\n" + proc.stderr)
    os.replace(tmp, target_dir / name)
    return target_dir / name


def load() -> ctypes.CDLL:
    global _LIB
    with _LOCK:
        if _LIB is not None:
            return _LIB
        lib = ctypes.CDLL(str(library_path()), use_errno=True)
        u64, p = ctypes.c_uint64, ctypes.c_void_p
        lib.pipe_abi_version.restype = ctypes.c_int
        lib.pipe_abi_version.argtypes = []
        lib.pipe_layout.restype = ctypes.c_int
        lib.pipe_layout.argtypes = [ctypes.c_int, u64, ctypes.POINTER(u64)]
        lib.pipe_blob_bytes.restype = u64
        lib.pipe_blob_bytes.argtypes = []
        lib.pipe_create.restype = p
        lib.pipe_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.c_int,
                                    ctypes.c_int, p, u64, u64, ctypes.c_char_p, u64]
        lib.pipe_set_chunk_bytes.restype = None
        lib.pipe_set_chunk_bytes.argtypes = [p, u64]
        lib.pipe_local_blob.restype = ctypes.c_int
        lib.pipe_local_blob.argtypes = [p, p, u64]
        lib.pipe_connect.restype = ctypes.c_int
        lib.pipe_connect.argtypes = [p, p, u64]
        lib.pipe_start.restype = ctypes.c_int
        lib.pipe_start.argtypes = [p]
        lib.pipe_stop.restype = None
        lib.pipe_stop.argtypes = [p]
        lib.pipe_failed.restype = ctypes.c_int
        lib.pipe_failed.argtypes = [p]
        lib.pipe_error.restype = ctypes.c_char_p
        lib.pipe_error.argtypes = [p]
        lib.pipe_stat.restype = u64
        lib.pipe_stat.argtypes = [p, ctypes.c_int]
        lib.pipe_hca_stat.restype = u64
        lib.pipe_hca_stat.argtypes = [p, ctypes.c_int, ctypes.c_int]
        lib.pipe_destroy.restype = None
        lib.pipe_destroy.argtypes = [p]
        if lib.pipe_abi_version() != PIPE_ABI_VERSION:
            raise RuntimeError("unexpected pipe proxy ABI version")
        _LIB = lib
        return lib


class PipeLayout:
    """Byte offsets of the pinned region: recv[src][slot], flag[src][slot][hca][chunk], send[slot], ctrl."""

    def __init__(self, world_size: int, slot_bytes: int) -> None:
        out = (ctypes.c_uint64 * 7)()
        if load().pipe_layout(int(world_size), int(slot_bytes), out) != 0:
            raise ValueError(f"unsupported pipe geometry: world={world_size} slot_bytes={slot_bytes}")
        (self.recv_off, self.flag_off, self.send_off, self.ctrl_off, self.total_bytes,
         self.flag_stride, self.slots) = (int(v) for v in out)


class PipeProxy:
    """One rank's pipe proxy context (same surface as b12x ``Proxy``)."""

    def __init__(self, *, world_size: int, rank: int, hca_names: tuple, gid_index: int, region_ptr: int,
                 region_bytes: int, slot_bytes: int, chunk_bytes: int) -> None:
        self._lib = load()
        names = (ctypes.c_char_p * len(hca_names))(*[n.encode() for n in hca_names])
        err = ctypes.create_string_buffer(512)
        self._ctx = self._lib.pipe_create(int(world_size), int(rank), names, len(hca_names), int(gid_index),
                                          ctypes.c_void_p(int(region_ptr)), int(region_bytes), int(slot_bytes),
                                          err, len(err))
        if not self._ctx:
            raise RuntimeError(f"pipe proxy setup failed: {err.value.decode(errors='replace')}")
        self._lib.pipe_set_chunk_bytes(self._ctx, int(chunk_bytes))
        self.world_size, self.rank, self.hca_names = int(world_size), int(rank), tuple(hca_names)

    def local_blob(self) -> bytes:
        n = int(self._lib.pipe_blob_bytes())
        buf = ctypes.create_string_buffer(n)
        if self._lib.pipe_local_blob(self._ctx, buf, n) != 0:
            raise RuntimeError("pipe proxy blob size mismatch")
        return buf.raw

    def connect(self, blobs: list) -> None:
        n = int(self._lib.pipe_blob_bytes())
        if len(blobs) != self.world_size or any(len(b) != n for b in blobs):
            raise RuntimeError("pipe proxy blob size mismatch")
        joined = b"".join(blobs)
        buf = ctypes.create_string_buffer(joined, len(joined))
        if self._lib.pipe_connect(self._ctx, buf, len(joined)) != 0:
            raise RuntimeError(f"pipe queue-pair connect failed: {self.error()}")

    def start(self) -> None:
        if self._lib.pipe_start(self._ctx) != 0:
            raise RuntimeError(f"pipe proxy thread failed to start: {self.error()}")

    def stop(self) -> None:
        self._lib.pipe_stop(self._ctx)

    def failed(self) -> bool:
        return bool(self._lib.pipe_failed(self._ctx))

    def error(self) -> str:
        raw = self._lib.pipe_error(self._ctx)
        return raw.decode(errors="replace") if raw else ""

    def stats(self) -> dict:
        return {
            "ops_posted": int(self._lib.pipe_stat(self._ctx, 0)),
            "writes_completed": int(self._lib.pipe_stat(self._ctx, 1)),
            "last_seq": int(self._lib.pipe_stat(self._ctx, 2)),
            "bytes_posted_per_hca": [int(self._lib.pipe_hca_stat(self._ctx, h, 1))
                                     for h in range(len(self.hca_names))],
        }

    def close(self) -> None:
        ctx, self._ctx = self._ctx, None
        if ctx:
            self._lib.pipe_destroy(ctx)

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


# -- kernel ----------------------------------------------------------------------------------


class _PipeGatherLaunch:
    """Chunked staging with per-chunk ready words, early local copy, per-chunk wait and copy-out."""

    def __init__(self, world_size, rank, threads, slots, flag_stride, hca_count) -> None:
        if int(threads) < int(world_size) * int(hca_count):
            raise ValueError("RoCE kernels need threads >= world_size * hca_count")
        self._world_size = int(world_size)
        self._rank = int(rank)
        self._threads = int(threads)
        self._slots = int(slots)
        self._flag_stride = int(flag_stride)
        self._hca_count = int(hca_count)

    @cute.jit
    def __call__(self, input_ptr: cute.Pointer, output_ptr: cute.Pointer, shard_packs: Int32, nbytes: Int32,
                 row_packs: Int32, n_chunks: Int32, chunk_packs: Int32, recv_base: Int64, flag_base: Int64,
                 send_base: Int64, ctrl_base: Int64, slot_bytes: Int64, epoch_ptr: Int64, chunk_counter_ptr: Int64,
                 tail_counter_ptr: Int64, poison_ptr: Int64, spin_limit: Uint32, grid_x: Int32,
                 stream: cuda.CUstream) -> None:
        self.kernel(input_ptr, output_ptr, shard_packs, nbytes, row_packs, n_chunks, chunk_packs, recv_base,
                    flag_base, send_base, ctrl_base, slot_bytes, epoch_ptr, chunk_counter_ptr, tail_counter_ptr,
                    poison_ptr, spin_limit).launch(grid=(grid_x, 1, 1), block=[self._threads, 1, 1],
                                                   cluster=(1, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, input_ptr: cute.Pointer, output_ptr: cute.Pointer, shard_packs: Int32, nbytes: Int32,
               row_packs: Int32, n_chunks: Int32, chunk_packs: Int32, recv_base: Int64, flag_base: Int64,
               send_base: Int64, ctrl_base: Int64, slot_bytes: Int64, epoch_ptr: Int64, chunk_counter_ptr: Int64,
               tail_counter_ptr: Int64, poison_ptr: Int64, spin_limit: Uint32) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        input_base = Int64(input_ptr.toint())
        output_base = Int64(output_ptr.toint())
        epoch = ld_relaxed_gpu_u32(epoch_ptr)
        seq = epoch + Uint32(1)
        slot = Int64(seq & Uint32(1))
        send_slot = send_base + slot * slot_bytes
        index = Int32(bidx) * Int32(self._threads) + Int32(tidx)
        stride = Int32(gdim) * Int32(self._threads)
        out_row_packs = Int32(self._world_size) * row_packs

        poisoned = ld_relaxed_gpu_u32(poison_ptr)
        if poisoned == Uint32(0):
            # 1. stage chunk by chunk; the last block to finish chunk c publishes ready[slot][c] = seq
            #    (chunk 0 also rings the doorbell). Each block's system fence before its arrival orders its staged
            #    stores; the last arriver fences again before the store the proxy polls (vendored pattern).
            c = Int32(0)
            while c < n_chunks:
                lo = c * chunk_packs
                hi = lo + chunk_packs
                if hi > shard_packs:
                    hi = shard_packs
                stage_index = lo + index
                while stage_index < hi:
                    words = ld_global_v4_u32(input_base + Int64(stage_index) * Int64(PACK_BYTES))
                    st_global_v4_u32(send_slot + Int64(stage_index) * Int64(PACK_BYTES),
                                     words[0], words[1], words[2], words[3])
                    stage_index += stride
                cute.arch.sync_threads()
                if Int32(tidx) == Int32(0):
                    fence_sc_sys()
                    prior = atomic_add_relaxed_gpu_u32(chunk_counter_ptr + Int64(c) * Int64(4), Uint32(1))
                    if (prior + Uint32(1)) % Uint32(gdim) == Uint32(0):
                        if c == Int32(0):
                            st_relaxed_sys_u32(ctrl_base + Int64(4), Uint32(nbytes))
                            st_relaxed_sys_u32(ctrl_base + Int64(16) + slot * Int64(4), Uint32(nbytes))
                        fence_sc_sys()
                        st_relaxed_sys_u32(ctrl_base + (Int64(CTRL_READY_WORD) + slot * Int64(MAX_CHUNKS)
                                                        + Int64(c)) * Int64(4), seq)
                        if c == Int32(0):
                            fence_sc_sys()
                            st_relaxed_sys_u32(ctrl_base, seq)
                c += Int32(1)

            # 2. the local shard goes to its column block of every output row while the NIC moves the payload
            copy_index = index
            while copy_index < shard_packs:
                row = copy_index // row_packs
                col = copy_index - row * row_packs
                dest = output_base + (Int64(row) * Int64(out_row_packs) + Int64(self._rank) * Int64(row_packs)
                                      + Int64(col)) * Int64(PACK_BYTES)
                words = ld_global_v4_u32(input_base + Int64(copy_index) * Int64(PACK_BYTES))
                st_global_v4_u32(dest, words[0], words[1], words[2], words[3])
                copy_index += stride

            # 3. chunk by chunk: wait for every peer stripe flag of chunk c, then copy chunk c of each peer out
            c = Int32(0)
            while c < n_chunks:
                lo = c * chunk_packs
                hi = lo + chunk_packs
                if hi > shard_packs:
                    hi = shard_packs
                if ld_relaxed_gpu_u32(poison_ptr) == Uint32(0):
                    if Int32(tidx) < Int32(self._world_size * self._hca_count):
                        peer = Int32(tidx) // Int32(self._hca_count)
                        hca = Int32(tidx) - peer * Int32(self._hca_count)
                        if peer != Int32(self._rank):
                            flag_addr = flag_base + ((((Int64(peer) * Int64(self._slots) + slot)
                                                       * Int64(self._hca_count) + Int64(hca)) * Int64(MAX_CHUNKS)
                                                      + Int64(c)) * Int64(self._flag_stride))
                            timed_out = spin_until_eq_acquire_sys(flag_addr, seq, spin_limit)
                            if timed_out != Uint32(0):
                                st_relaxed_sys_u32(ctrl_base + Int64(12), Uint32(peer))
                                st_relaxed_sys_u32(ctrl_base + Int64(24), Uint32(hca))
                                st_relaxed_sys_u32(ctrl_base + Int64(8), seq)
                                st_release_gpu_u32(poison_ptr, seq)
                cute.arch.sync_threads()
                if ld_relaxed_gpu_u32(poison_ptr) == Uint32(0):
                    for source in cutlass.range_constexpr(self._world_size):
                        if cutlass.const_expr(source != self._rank):
                            peer_slot = recv_base + (Int64(source) * Int64(self._slots) + slot) * slot_bytes
                            copy_index = lo + index
                            while copy_index < hi:
                                row = copy_index // row_packs
                                col = copy_index - row * row_packs
                                dest = output_base + (Int64(row) * Int64(out_row_packs)
                                                      + Int64(source) * Int64(row_packs)
                                                      + Int64(col)) * Int64(PACK_BYTES)
                                words = ld_relaxed_sys_v4_u32(peer_slot + Int64(copy_index) * Int64(PACK_BYTES))
                                st_global_v4_u32(dest, words[0], words[1], words[2], words[3])
                                copy_index += stride
                c += Int32(1)

            # 4. the last block to finish publishes the next epoch (vendored)
            fence_sc_gpu()
            cute.arch.sync_threads()
            if Int32(tidx) == Int32(0):
                prior = atomic_add_relaxed_gpu_u32(tail_counter_ptr, Uint32(1))
                if (prior + Uint32(1)) % Uint32(gdim) == Uint32(0):
                    fence_sc_gpu()
                    if ld_relaxed_sys_u32(ctrl_base + Int64(8)) == Uint32(0):
                        st_release_gpu_u32(epoch_ptr, seq)


def _key(world_size, rank, threads, slots, flag_stride, hca_count, device_index):
    return (int(world_size), int(rank), int(threads), int(slots), int(flag_stride), int(hca_count),
            int(device_index))


@functools.cache
def get_launcher(world_size, rank, threads, slots, flag_stride, hca_count, device_index):
    process_key = _key(world_size, rank, threads, slots, flag_stride, hca_count, device_index)
    launch = _PipeGatherLaunch(world_size, rank, threads, slots, flag_stride, hca_count)
    cache_key = process_key[:-1]
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=cache_key)
    dummy = _allgather_cute._dummy
    raw = b12x_compile(
        launch, dummy(cutlass.Uint32, 16), dummy(cutlass.Uint32, 16),
        1, 16, 1, 1, 1, 16, 16, 16, 16, 4096, 16, 16, 16, 16, 1, 1, current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("glm.roce.pipe_gather", 1, cache_key),
    )

    def run(input_address, output_address, shard_packs, nbytes, row_packs, n_chunks, chunk_packs, recv_base,
            flag_base, send_base, ctrl_base, slot_bytes, epoch_address, chunk_counter_address,
            tail_counter_address, poison_address, spin_limit, grid_x) -> None:
        raw(make_ptr(cutlass.Uint32, input_address, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Uint32, output_address, cute.AddressSpace.gmem, assumed_align=16),
            int(shard_packs), int(nbytes), int(row_packs), int(n_chunks), int(chunk_packs), int(recv_base),
            int(flag_base), int(send_base), int(ctrl_base), int(slot_bytes), int(epoch_address),
            int(chunk_counter_address), int(tail_counter_address), int(poison_address), int(spin_limit),
            int(grid_x), current_cuda_stream())

    _PREPARED.add(process_key)
    return run


# -- runtime ---------------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _runtime_class():
    import torch
    import torch.distributed as dist
    from b12x.comm.roce import roce_oneshot as ro

    class PipeGather(ro.RoceOneshotAllReduce):
        """Eager-only pipelined gathers for one DCP pair (vendored runtime surface; own proxy and kernel)."""

        algorithm = "rocenante-pipe"

        def __init__(self, *, exchange_group, device, max_gather_bytes: int, chunk_bytes: int,
                     hca_names: Optional[tuple] = None, gid_index: Optional[int] = None,
                     threads: int = ro.DEFAULT_THREADS, blocks: int = ro.DEFAULT_BLOCKS) -> None:
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
            if blocks < 1 or blocks & (blocks - 1) or blocks > 32:
                raise ValueError("pipe blocks must be a power of two <= 32")
            self.max_size = ro.PACK_BYTES  # gathers only; all_reduce is never routed here
            self.max_gather_bytes = int(max_gather_bytes)
            self.chunk_bytes = int(chunk_bytes)
            self._threads = int(threads)
            self._blocks = int(blocks)
            self._counter_classes = self._blocks.bit_length()
            self.gid_index = ro.default_gid_index() if gid_index is None else int(gid_index)
            self.spin_limit = ro._env_int("B12X_ROCE_SPIN_LIMIT", default=ro.DEFAULT_SPIN_LIMIT)
            names = tuple(hca_names) if hca_names else ro.discover_hcas(self.gid_index)
            if not names:
                raise RuntimeError("no active RDMA device for the pipe runtime")
            self.hca_names = names[:2]
            slot_bytes = ro._align_up(max(self.max_size, self.max_gather_bytes), ro._SLOT_ALIGNMENT)
            self._layout = PipeLayout(self.world_size, slot_bytes)
            self._slot_bytes = slot_bytes
            with torch.cuda.device(self.device):
                self._region = torch.zeros(self._layout.total_bytes, dtype=torch.uint8, pin_memory=True)
                # epoch, stage (unused) and tail counters per grid class, poison: the vendored layout
                self._counters = torch.zeros(2 + 2 * self._counter_classes, dtype=torch.int32, device=self.device)
                # one arrival counter per (grid class, chunk), free-running like the vendored stage counters
                self._chunk_counters = torch.zeros(self._counter_classes * MAX_CHUNKS, dtype=torch.int32,
                                                   device=self.device)
            host_ptr = self._region.data_ptr()
            if self._device_pointer(host_ptr) != host_ptr:
                raise RuntimeError("the pipe runtime needs directly device-accessible host pointers")
            self._recv_base = host_ptr + self._layout.recv_off
            self._flag_base = host_ptr + self._layout.flag_off
            self._send_base = host_ptr + self._layout.send_off
            self._ctrl_base = host_ptr + self._layout.ctrl_off
            self._ctrl_words = self._region[self._layout.ctrl_off: self._layout.ctrl_off + 28].view(torch.int32)
            self._error_word = self._ctrl_words[2:3]
            self._ctrl_np = self._ctrl_words.numpy()
            self._epoch_address = self._counters.data_ptr()
            self._poison_address = self._epoch_address + 4 * (1 + 2 * self._counter_classes)

            error: Optional[str] = None
            blob = b""
            try:
                self._proxy = PipeProxy(world_size=self.world_size, rank=self.rank, hca_names=self.hca_names,
                                        gid_index=self.gid_index, region_ptr=host_ptr,
                                        region_bytes=self._layout.total_bytes, slot_bytes=slot_bytes,
                                        chunk_bytes=self.chunk_bytes)
                blob = self._proxy.local_blob()
            except Exception as exc:  # noqa: BLE001 - reported collectively
                error = str(exc)
            config = {
                "pipe_abi": load().pipe_abi_version() if error is None else None,
                "world_size": self.world_size, "hca_count": len(self.hca_names), "slot_bytes": slot_bytes,
                "slots": self._layout.slots, "flag_stride": self._layout.flag_stride,
                "max_gather_bytes": self.max_gather_bytes, "chunk_bytes": self.chunk_bytes,
                "max_chunks": MAX_CHUNKS, "spin_limit": self.spin_limit, "threads": self._threads,
                "blocks": self._blocks,
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
                raise RuntimeError("pipe runtime setup failed: " + "; ".join(failures))
            try:
                self._proxy.connect([s[1] for s in statuses])
                self._proxy.start()
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
            verdicts = ro._exchange(error, exchange_group)
            failures = [f"rank {i}: {v}" for i, v in enumerate(verdicts) if v is not None]
            if failures:
                self.close()
                raise RuntimeError("pipe runtime connect failed: " + "; ".join(failures))
            logger.info("GLM_ROCE_PIPE ready rank=%d hcas=%s max_gather=%d chunk_bytes=%d blocks=%d", self.rank,
                        ",".join(self.hca_names), self.max_gather_bytes, self.chunk_bytes, self._blocks)

        def _key(self):
            return (self.world_size, self.rank, self._threads, self._layout.slots, self._layout.flag_stride,
                    len(self.hca_names), self.device.index)

        def prepare(self, dtypes=None, *, padded_gather: bool = False) -> None:
            with torch.cuda.device(self.device):
                get_launcher(*self._key())
                if padded_gather:
                    self._gather_scratch(ro.PACK_BYTES)

        def should_allreduce(self, inp) -> bool:
            return False

        def _launch_gather(self, input_address: int, output_address: int, nbytes: int, row_packs: int) -> None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("the pipe runtime is eager-only; a captured gather must use the vendored one")
            launcher = get_launcher(*self._key())
            grid_blocks = ro._grid_blocks(nbytes // ro.PACK_BYTES, self._threads, self._blocks)
            counter_class = int(grid_blocks).bit_length() - 1
            tail = self._epoch_address + 4 * (1 + self._counter_classes + counter_class)
            chunk_counters = self._chunk_counters.data_ptr() + 4 * MAX_CHUNKS * counter_class
            n_chunks, chunk_packs = chunk_geometry(nbytes, self.chunk_bytes)
            launcher(input_address, output_address, nbytes // ro.PACK_BYTES, nbytes, row_packs, n_chunks,
                     chunk_packs, self._recv_base, self._flag_base, self._send_base, self._ctrl_base,
                     self._slot_bytes, self._epoch_address, chunk_counters, tail, self._poison_address,
                     self.spin_limit, grid_blocks)
            self.check_health()

        def stats(self) -> dict[str, Any]:
            info = super().stats()
            info.update({"algorithm": self.algorithm, "chunk_bytes": self.chunk_bytes, "blocks": self._blocks})
            return info

    return PipeGather


def PipeGather(**kwargs):  # noqa: N802 - factory with the class's name
    return _runtime_class()(**kwargs)
