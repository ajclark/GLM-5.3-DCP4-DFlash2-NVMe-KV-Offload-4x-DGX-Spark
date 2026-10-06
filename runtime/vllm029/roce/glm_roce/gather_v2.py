"""Prefill-size RoCEnante gathers with less serial copying (results/prefill-item2-20261006/REPORT.md, E2).

The vendored b12x all-gather kernel runs four phases serially on at most 8 blocks: stage the local shard into
the pinned send slot, ring the proxy doorbell, wait for every peer's stripe flags, then copy *every* shard
(local from the input, peers from their NIC-written slots) into the output. For a 36 MiB prefill shard the wire
takes ~1.6 ms but the whole collective ~3.0-3.4 ms. Two levers, both only for shards of at least
``GLM_ROCE_LARGE_MIN_BYTES`` (default 4 MiB, i.e. prefill; decode gathers keep the stock kernel and grid):

- ``GLM_ROCE_LARGE_BLOCKS`` (8 = stock, 16 or 32): a larger grid for the staging and copy-out phases. Capped at 32
  so every block is co-resident on GB10's 48 SMs: the doorbell is rung by the last block to finish staging, so a
  block that could not be scheduled until a spinning block exits would deadlock.
- ``GLM_ROCE_OWNCOPY_EARLY=1``: the local shard is copied into the output right after the doorbell, while the
  NIC moves the payload, instead of after the flag wait. Same bytes to the same addresses (bit-identical); only
  the peer shards remain after the wait.

The protocol (slots, flags, epoch, counters, proxy) is untouched, so the runtime, the proxy and the decode path
are exactly the vendored ones. Vendored b12x stays byte-identical to its reviewed state (tests/test_vendored_b12x.py).
"""
from __future__ import annotations

import functools
import os
from collections.abc import Callable

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
from b12x.comm.roce.roce_oneshot import PACK_BYTES, RoceOneshotAllReduce, _grid_blocks

ENV_LARGE_BLOCKS = "GLM_ROCE_LARGE_BLOCKS"
ENV_LARGE_MIN_BYTES = "GLM_ROCE_LARGE_MIN_BYTES"
ENV_OWNCOPY_EARLY = "GLM_ROCE_OWNCOPY_EARLY"
STOCK_BLOCKS = 8
_PREPARED: set[tuple[object, ...]] = set()


def read_config(environ=None) -> tuple[int, int, bool]:
    """``(large_blocks, large_min_bytes, own_early)``; raises on unusable values (reported through the vote)."""
    env = os.environ if environ is None else environ
    blocks = int(env.get(ENV_LARGE_BLOCKS, str(STOCK_BLOCKS)))
    if blocks not in (8, 16, 32):
        raise ValueError(f"{ENV_LARGE_BLOCKS}={blocks}: must be 8, 16 or 32")
    min_bytes = int(env.get(ENV_LARGE_MIN_BYTES, str(4 << 20)))
    if min_bytes < (1 << 20):
        raise ValueError(f"{ENV_LARGE_MIN_BYTES}={min_bytes}: below 1 MiB would reach decode-sized gathers")
    early = env.get(ENV_OWNCOPY_EARLY, "0").strip().lower() not in ("", "0", "off", "false", "no")
    return blocks, min_bytes, early


def enabled(environ=None) -> bool:
    blocks, _, early = read_config(environ)
    return blocks != STOCK_BLOCKS or early


class _GatherEarlyLaunch:
    """The vendored all-gather kernel with the local-shard copy moved before the flag wait."""

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
                 row_packs: Int32, recv_base: Int64, flag_base: Int64, send_base: Int64, ctrl_base: Int64,
                 slot_bytes: Int64, epoch_ptr: Int64, stage_counter_ptr: Int64, tail_counter_ptr: Int64,
                 poison_ptr: Int64, spin_limit: Uint32, grid_x: Int32, stream: cuda.CUstream) -> None:
        self.kernel(input_ptr, output_ptr, shard_packs, nbytes, row_packs, recv_base, flag_base, send_base,
                    ctrl_base, slot_bytes, epoch_ptr, stage_counter_ptr, tail_counter_ptr, poison_ptr,
                    spin_limit).launch(grid=(grid_x, 1, 1), block=[self._threads, 1, 1], cluster=(1, 1, 1),
                                       stream=stream)

    @cute.kernel
    def kernel(self, input_ptr: cute.Pointer, output_ptr: cute.Pointer, shard_packs: Int32, nbytes: Int32,
               row_packs: Int32, recv_base: Int64, flag_base: Int64, send_base: Int64, ctrl_base: Int64,
               slot_bytes: Int64, epoch_ptr: Int64, stage_counter_ptr: Int64, tail_counter_ptr: Int64,
               poison_ptr: Int64, spin_limit: Uint32) -> None:
        # Phases 1, 2, 3 and 5 are the vendored kernel's statements unchanged; phase 4's local-shard copy runs as
        # phase 2b, between the doorbell and the flag wait.
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

        poisoned = ld_relaxed_gpu_u32(poison_ptr)
        if poisoned == Uint32(0):
            # 1. stage the local shard into the pinned send slot
            stage_index = index
            while stage_index < shard_packs:
                words = ld_global_v4_u32(input_base + Int64(stage_index) * Int64(PACK_BYTES))
                st_global_v4_u32(send_slot + Int64(stage_index) * Int64(PACK_BYTES),
                                 words[0], words[1], words[2], words[3])
                stage_index += stride
            cute.arch.sync_threads()

            # 2. the last block to finish staging rings the proxy doorbell
            if Int32(tidx) == Int32(0):
                fence_sc_sys()
                prior = atomic_add_relaxed_gpu_u32(stage_counter_ptr, Uint32(1))
                if (prior + Uint32(1)) % Uint32(gdim) == Uint32(0):
                    st_relaxed_sys_u32(ctrl_base + Int64(4), Uint32(nbytes))
                    st_relaxed_sys_u32(ctrl_base + Int64(16) + slot * Int64(4), Uint32(nbytes))
                    fence_sc_sys()
                    st_relaxed_sys_u32(ctrl_base, seq)

            # 2b. the local shard goes to its column block of every output row while the NIC moves the payload
            out_row_packs = Int32(self._world_size) * row_packs
            copy_index = index
            while copy_index < shard_packs:
                row = copy_index // row_packs
                col = copy_index - row * row_packs
                dest = output_base + (Int64(row) * Int64(out_row_packs) + Int64(self._rank) * Int64(row_packs)
                                      + Int64(col)) * Int64(PACK_BYTES)
                words = ld_global_v4_u32(input_base + Int64(copy_index) * Int64(PACK_BYTES))
                st_global_v4_u32(dest, words[0], words[1], words[2], words[3])
                copy_index += stride

            # 3. wait for every peer's payload-stripe flags
            if Int32(tidx) < Int32(self._world_size * self._hca_count):
                peer = Int32(tidx) // Int32(self._hca_count)
                hca = Int32(tidx) - peer * Int32(self._hca_count)
                if peer != Int32(self._rank):
                    flag_addr = flag_base + ((Int64(peer) * Int64(self._slots) + slot) * Int64(self._hca_count)
                                             + Int64(hca)) * Int64(self._flag_stride)
                    timed_out = spin_until_eq_acquire_sys(flag_addr, seq, spin_limit)
                    if timed_out != Uint32(0):
                        st_relaxed_sys_u32(ctrl_base + Int64(12), Uint32(peer))
                        st_relaxed_sys_u32(ctrl_base + Int64(24), Uint32(hca))
                        st_relaxed_sys_u32(ctrl_base + Int64(8), seq)
                        st_release_gpu_u32(poison_ptr, seq)
            cute.arch.sync_threads()
            failed = ld_relaxed_gpu_u32(poison_ptr)
            if failed == Uint32(0):
                # 4. peer shards only (the local one went out in 2b)
                for source in cutlass.range_constexpr(self._world_size):
                    if cutlass.const_expr(source != self._rank):
                        peer_slot = recv_base + (Int64(source) * Int64(self._slots) + slot) * slot_bytes
                        copy_index = index
                        while copy_index < shard_packs:
                            row = copy_index // row_packs
                            col = copy_index - row * row_packs
                            dest = output_base + (Int64(row) * Int64(out_row_packs)
                                                  + Int64(source) * Int64(row_packs)
                                                  + Int64(col)) * Int64(PACK_BYTES)
                            words = ld_relaxed_sys_v4_u32(peer_slot + Int64(copy_index) * Int64(PACK_BYTES))
                            st_global_v4_u32(dest, words[0], words[1], words[2], words[3])
                            copy_index += stride

            # 5. the last block to finish publishes the next epoch
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


def is_prepared(*key) -> bool:
    return _key(*key) in _PREPARED


@functools.cache
def get_early_launcher(world_size, rank, threads, slots, flag_stride, hca_count, device_index) -> Callable[..., None]:
    """Compile the early-local-copy gather once (same argument types as the vendored launcher)."""
    process_key = _key(world_size, rank, threads, slots, flag_stride, hca_count, device_index)
    launch = _GatherEarlyLaunch(world_size, rank, threads, slots, flag_stride, hca_count)
    cache_key = process_key[:-1]
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=cache_key)
    dummy = _allgather_cute._dummy
    raw = b12x_compile(
        launch, dummy(cutlass.Uint32, 16), dummy(cutlass.Uint32, 16),
        1, 16, 1, 16, 16, 16, 16, 4096, 16, 16, 16, 16, 1, 1, current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("glm.roce.allgather_early", 1, cache_key),
    )

    def run(input_address, output_address, shard_packs, nbytes, row_packs, recv_base, flag_base, send_base,
            ctrl_base, slot_bytes, epoch_address, stage_counter_address, tail_counter_address, poison_address,
            spin_limit, grid_x) -> None:
        raw(make_ptr(cutlass.Uint32, input_address, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Uint32, output_address, cute.AddressSpace.gmem, assumed_align=16),
            int(shard_packs), int(nbytes), int(row_packs), int(recv_base), int(flag_base), int(send_base),
            int(ctrl_base), int(slot_bytes), int(epoch_address), int(stage_counter_address),
            int(tail_counter_address), int(poison_address), int(spin_limit), int(grid_x), current_cuda_stream())

    _PREPARED.add(process_key)
    return run


class GatherV2AllReduce(RoceOneshotAllReduce):
    """The vendored runtime; shards >= ``large_min_bytes`` launch with ``large_blocks`` and optionally the
    early-local-copy kernel. All-reduces and smaller gathers keep the stock 8-block geometry and kernel."""

    def __init__(self, *, large_blocks: int, large_min_bytes: int, own_early: bool, **kwargs) -> None:
        # Arrival counters exist per power-of-two grid up to the constructor's ``blocks``; allocate for the large
        # grid, then keep the stock maximum for everything that is not a large gather.
        super().__init__(blocks=max(STOCK_BLOCKS, int(large_blocks)), **kwargs)
        self._blocks = STOCK_BLOCKS
        self._large_blocks = int(large_blocks)
        self._large_min_bytes = int(large_min_bytes)
        self._own_early = bool(own_early)

    def _early_key(self):
        return self._gather_launcher_key()

    def prepare(self, dtypes=None, *, padded_gather: bool = False) -> None:
        import torch

        super().prepare(dtypes if dtypes is not None else (torch.bfloat16,), padded_gather=padded_gather)
        if self._own_early:
            with torch.cuda.device(self.device):
                get_early_launcher(*self._early_key())

    def stats(self):
        s = super().stats()
        s.update({"large_blocks": self._large_blocks, "large_min_bytes": self._large_min_bytes,
                  "own_early": self._own_early})
        return s

    def _launch_gather(self, input_address: int, output_address: int, nbytes: int, row_packs: int) -> None:
        if nbytes < self._large_min_bytes:
            return super()._launch_gather(input_address, output_address, nbytes, row_packs)
        import torch

        capturing = torch.cuda.is_current_stream_capturing()
        if self._own_early:
            key = self._early_key()
            if capturing and not is_prepared(*key):
                raise RuntimeError("RoCE early gather launcher must be prepared before CUDA graph capture")
            launcher = get_early_launcher(*key)
        else:
            key = self._gather_launcher_key()
            if capturing and not _allgather_cute.is_launcher_prepared(*key):
                raise RuntimeError("RoCE all-gather launcher must be prepared before CUDA graph capture")
            launcher = _allgather_cute.get_launcher(*key)
        grid_blocks = _grid_blocks(nbytes // PACK_BYTES, self._threads, self._large_blocks)
        stage_counter, tail_counter = self._counter_addresses(grid_blocks)
        launcher(input_address, output_address, nbytes // PACK_BYTES, nbytes, row_packs, self._recv_base,
                 self._flag_base, self._send_base, self._ctrl_base, self._slot_bytes, self._epoch_address,
                 stage_counter, tail_counter, self._poison_address, self.spin_limit, grid_blocks)
        if not capturing:
            self.check_health()
