# Why our RDMA collectives take about half of NCCL's time (2026-10-01)

**Scope.** This explains why the custom one-shot RDMA collectives beat NCCL on the 4× DGX Spark ring at decode sizes:
- RoCEnante for the DCP pairs (step 3);
- this repo's ring transport for the TP all-reduce (step 7).

It is built from three sources: our measurements, NCCL's own init log on these nodes, and the two code paths. Anything
inferred rather than measured says so. Step numbers refer to [DECODE-SPEEDUPS.md](DECODE-SPEEDUPS.md). The live
cycle times were measured on an abliterated graft of GLM-5.3 with the same quantization envelope as GLM-5.3 Int4-Int8Mix
(Int4 group-128 routed experts, Int8 group-128 elsewhere, identical shapes).

## The measurement

**TP all-reduce** of `[T, 6144]` bf16 over 4 ranks. Each run is 156 back-to-back all-reduces in one CUDA graph, NCCL
with the production environment, same harness, model stopped. Source:
`results/step7-ring-allreduce-20261001/ab/` (variant `s0b8`), mean of the four ranks.

| tokens T | payload | NCCL Ring | our ring | ratio | saved |
|---:|---:|---:|---:|---:|---:|
| 1 | 12 KiB | 51.0 µs | 21.1 µs | 2.4× | 30 µs |
| 2 | 24 KiB | 59.8 | 25.9 | 2.3× | 34 |
| 4 | 48 KiB | 70.6 | 31.4 | 2.2× | 39 |
| 8 | 96 KiB | 97.3 | 49.0 | **2.0×** | 48 |
| 16 | 192 KiB | 135.2 | 81.4 | 1.7× | 54 |
| 32 | 384 KiB | 185.4 | 127.6 | 1.5× | 58 |

These are the original link map's numbers. Since the PCIe-aware link map (see "What limits throughput per byte"), the
ring takes 19.8 / 38.5 / 81.9 µs at T = 1 / 8 / 32.

A straight-line fit of the original numbers splits each into a fixed cost and a per-byte cost:

| | fixed | per KiB | effective payload rate |
|---|---:|---:|---:|
| NCCL | ≈ 55 µs | 0.36 µs | ≈ 2.8 GB/s |
| ours | ≈ 20 µs | 0.29 µs | ≈ 3.5 GB/s |

"Half the latency" is accurate at the sizes decode actually uses (T = 1–8 for one request). The advantage shrinks as
messages grow. In live serving it showed up as −6.3 ms per 1-request K=7 cycle (124.4 vs 130.7 ms), close to the
predicted 156 all-reduces × ~46 µs.

**DCP pairs** (2 ranks, direct cable): query gather, LSE gather and attention-output reduce-scatter, 78 layers × 3 per
graph. Source: `results/step3-roce-20261001/pair-*.log`.

| per collective | NCCL | RoCEnante |
|---|---:|---:|
| T=8 (1 request) | 47 µs | 25 µs |
| T=32 (4 requests) | 87 µs | 54 µs |

## Short answer

Four structural differences, in order of how much of the gap they explain:

1. **Fewer dependent network steps (TP only).** NCCL's ring all-reduce makes 2(N−1) = **6** sequential hops on 4
   ranks: reduce-scatter, then all-gather. Ours makes **2**: everyone writes both neighbours, then each rank forwards one
   neighbour's data once. This is most of the fixed cost.
2. **The receiver's CPU is never involved.** NCCL's InfiniBand transport is two-sided: the receiver's proxy thread
   takes part in every transfer. Ours is a one-sided RDMA write straight into the memory the consuming GPU kernel
   polls, followed by an inline flag.
3. **Every byte is sent once, at full payload efficiency.**
   - At decode sizes NCCL runs the LL protocol: forcing `NCCL_PROTO=LL` reproduces its default timings at T ≤ 8.
   - LL pairs every 4 bytes of data with a 4-byte flag, so it pushes twice the bytes through the NIC's PCIe link, in
     one-channel slices that each pass through both proxies.
   - NCCL also bounces every step through host staging buffers, because it reports GPU Direct RDMA *disabled* here.
     On GB10 the copies themselves are cheap (a 96 KiB copy takes 1.7 µs, measured below), so they cost mainly extra
     steps, not bandwidth.
   - Our kernel stages once; the NIC writes raw payload straight into the consumer's slot; the reduction reads it in
     place.
4. **No CPU nodes in the CUDA graphs** (whole graphs, not single collectives). NCCL inserts a host callback per
   collective in captured graphs. Our proxies poll a doorbell in memory instead.

## What NCCL does on these nodes (from its init log)

`docker logs vllm_glm53big | grep "NCCL INFO"` on spark-06c4:

```
NET/IB : GPU Direct RDMA Disabled for HCA 0 'roceP2p1s0f0'      (and for HCA 1, and the merged HCA 2)
Connected all rings, use ring PXN 0 GDR 0
1 coll channels, 1 collnet channels, 0 nvls channels, 1 p2p channels, 1 p2p channels per peer
Channel 00/01 : 0 1 2 3
ncclIbConnectImpl: ... ctsFifoRkey=0x1de0de ctsFifoLkey=0x1de0de
ncclIbRecvCommInit: Receive work requests will be posted on-demand
[Proxy Progress] Device 0 CPU core 0
```

The launcher pins further settings (`runtime/vllm029/launch.sh`):
- `NCCL_ALGO=Ring`;
- `NCCL_MIN/MAX_NCHANNELS=1` and `NCCL_MIN/MAX_CTAS=1`. These came with the original recipe (as of 10-01; since 10-06
  NCCL runs 2 channels over both PCIe links, see [NETWORK.md](NETWORK.md)).

Putting that together, one NCCL TP all-reduce of S bytes on our 4-node ring is:

- **6 sequential steps**, each moving S/4.
- **Every step makes the GPU kernel wait** for the incoming chunk, reduce it with its own data (reduce-scatter steps),
  then hand it on.
- **Each hop goes through both proxies.**
  - The sending GPU writes into a send buffer and bumps a counter; the sender's proxy thread polls it and posts the
    RDMA write.
  - The receiving side is receiver-driven. The receiver's proxy posts the receive and a "clear to send" entry into the
    sender's FIFO (`ctsFifo`), consumes the receive completion, and recycles the buffer.
  - With the Simple protocol the receiving GPU waits for the proxy's counter update. With LL/LL128 it polls inline
    flags, but the receive proxy still paces the buffers.

  NCCL does not log which protocol it picks per call; we did not trace it.
- **Bounce buffers:** with GDR off, the send and receive buffers are host-pinned staging buffers, not the tensors.
  Data goes from the GPU tensor to the send buffer (GPU copy), across the wire to the receive buffer, then into the
  next step or the output (GPU copy or reduce).
- **One channel and one CTA** (thread block) per collective, so those copies run on a single block.

## What our path does

The pinned region, doorbell and CuTe kernel are b12x RoCEnante's (`runtime/vllm029/roce/b12x/b12x/comm/roce/`). The TP
ring transport is ours (`runtime/vllm029/roce/glm_roce/_ring_proxy.c`, `ring.py`). One TP all-reduce of S bytes goes:

1. **Stage and doorbell.** One kernel launch (8 blocks) copies the input into this rank's pinned send slot. It then
   writes the byte count and sequence number into the doorbell word.
2. **Our proxy posts the writes.** The proxy thread on this node busy-polls the doorbell. It posts two RDMA writes per
   neighbour, on the reliable connection facing that neighbour:
   - the payload, written straight into the neighbour's receive slot for this rank;
   - a 4-byte inline write of the sequence number into the matching flag.

   The connection delivers in order, so the flag cannot land before the data. Region addresses and keys were
   exchanged once at startup, so no handshake happens per operation.
3. **One forward.** When the counter-clockwise neighbour's flag lands in this node's memory, this node's proxy (CPU
   only, no GPU kernel) writes those bytes and flag on to the clockwise neighbour. After that, every rank holds all
   three peers' data.
4. **Wait, then reduce once.** The same kernel spins on the three peers' flags, reading pinned memory in place. It then
   sums all four inputs in fp32 in fixed rank order and writes the output once. The reduction is identical on every
   rank, with one rounding.
5. **Flow control comes from the collective itself.** Each peer has two slots. A slot can only be rewritten after
   every reader has finished the previous use, because nobody can start op s+2 before everyone finished op s+1. The
   proof is in the header of `_ring_proxy.c`. There are no credits, no clear-to-send and no receive work requests.

The DCP pairs use the same machinery with one direct hop (RoCEnante as vendored: the two-rank all-gather, and
reduce-scatter as an exchange of halves).

## Where the time goes

**Fixed cost: ~55 µs vs ~20 µs, mostly hop count.**
- NCCL: ~55 µs over 6 dependent steps is roughly 8–9 µs per step. That covers the GPU-to-proxy handoff, the RDMA post,
  the wire and completion, the receiver-side handling and the GPU kernel's per-step synchronisation.
- Ours: ~20 µs is two hops plus one kernel's stage, doorbell, wait and reduce.
- The per-hop costs are similar in size. The 3× difference in dependent hops (6 vs 2) explains most of the 35 µs.
- This is an inference from the fit and the step counts; neither stack was profiled at that level.

**Per-byte cost: the NIC's PCIe link, not the wire or the GPU.** Measured in "What limits throughput per byte"
below. NCCL's per-byte cost is further inflated by LL's 50% payload efficiency at decode sizes; that is inferred from
the `NCCL_PROTO` A/B, not profiled. In short:
- Each 200 GbE port reaches the GB10 through a PCIe Gen5 x4 link, ~13.6 GB/s each way, shared by the two ports on that
  link.
- Both of our original ring devices sat on the same x4 link.
- The GPU's staging and reduction passes cost under 10% of an all-reduce.

**Graph-level (not in the microbenchmark).**
- In captured CUDA graphs, NCCL adds a CPU host callback per collective to hand the operation to its proxy.
- The September trace showed the first collective of every target and draft graph starting 210–580 µs late (median),
  waiting for CPU threads to wake from deep idle states, then running 300–700 µs
  (`results/step5-l2-argmax-20261001/REPORT.md`).
- Our proxies poll memory, so with the ring and the DCP pairs on RDMA the decode graphs have no NCCL operations at
  ≤1 MiB. The GPU now idles 0.4 ms between draft and verify (step-4 gap timer). Most of that recovery came from the
  local K decision; we did not isolate the graph-level part.

## What we pay for it

- **More bytes on the wire.** Writing everyone's full payload to both neighbours is latency-optimal, not
  bandwidth-optimal. NCCL's ring wins for big messages, so the ring takes only all-reduces ≤ 1 MiB
  (`GLM_ROCE_TP_MAX_SIZE`). Prefill-sized all-reduces (25 MB at 2,048 tokens) stay on NCCL.
- **CPU and power.** Each runtime has a proxy thread that busy-polls while serving. That is two per node now (DCP pair
  and TP ring); both back off to naps when idle. The DCP-only state was checked nominal; the ring's second thread is
  still awaiting the operator's power check.
- **Topology-specific.** The TP transport assumes exactly four ranks in a ring with ring order equal to TP rank order.
  RoCEnante needs a direct link to every peer, which only the DCP pairs have.
- **Numerics.** The TP ring is exact and identical on every rank, but not NCCL's bits: about a third of bf16 elements
  differ by one ulp. Ours is the single-rounding fp32 sum, so it is closer to the exact result. The DCP pairs are
  byte-identical to NCCL.
- **Fail-stop.** A peer that stops answering trips a spin limit; the runtime raises and serving stops. There is no
  fallback mid-run. `ROCE_TP=0` / `ROCE_DCP=0` revert to NCCL with no rebuild.

## What limits throughput per byte (measured; `results/perbyte-20261001/`)

Four experiments, model stopped.

**1. The links: perftest `ib_write_lat` / `ib_write_bw`, host memory, one queue pair.**
- Every port tops out at **109 Gb/s = 13.6 GB/s**, not 200 Gb/s: f0 and f1 on the ring, and the DCP device.
- One write takes about 2 µs + size / 13.6 GB/s:

  | write size | one-way time |
  |---:|---:|
  | 16 KiB | 4.6 µs |
  | 64 KiB | 8.7 µs |
  | 128 KiB | 13.4 µs |
  | 256 KiB | 25.1 µs |

- The RDMA devices sit behind **two** PCIe Gen5 x4 links (`/sys/class/infiniband/*/device`), one per PCI domain:

  | PCI domain | devices | port 0 | port 1 |
  |---|---|---|---|
  | 0000:01:00 | rocep1s0f0, rocep1s0f1 | no IPv4 | DCP-pair link .101/.105 |
  | 0002:01:00 | roceP2p1s0f0, roceP2p1s0f1 | ring links .102/.104 | pair links .100/.103 |

  Gen5 x4 is about 15.75 GB/s raw, so 13.6 GB/s is that link's practical limit. The x4 host link, not the 200 Gb/s
  port, caps the bandwidth.

**2. The GPU passes: `runtime/vllm029/roce/bench/bench_gpu_pinned.py`, one GPU, cold buffers, CUDA graphs.**

| pass | T=8 | T=32 |
|---|---:|---:|
| Stage (device → pinned send slot) | 1.7 µs | 2.9 µs |
| Reduce (own + 3 peers read from pinned memory, fp32, store) | 3.3 µs | 8.9 µs |

- Pinned host memory streams at 150–220 GB/s to the GPU, close to device memory.
- `ld.relaxed.sys` (b12x) costs the same as plain or volatile loads.
- 8, 16 or 48 blocks make no difference.
- **The GPU's share is under 10% of an all-reduce.**

**3. Inside the proxy: the trace build (`GLM_ROCE_RING_TRACE=1`) during real 4-rank all-reduces.**
Times are medians relative to the rank's own doorbell (`runtime/vllm029/roce/bench/bench_ring_breakdown.py`).

| T=8 (96 KiB) | ports on one x4 (original) | pair edge on the other x4 |
|---|---:|---:|
| proxy sees doorbell → writes posted | 0.2 µs | 0.2 µs |
| own write acked (one hop) | **22.9 µs** | **12.9 µs** |
| neighbours' payloads land | 22.4 / 22.7 µs | 13.7 / 12.9 µs |
| proxy reacts to neighbour → forward posted | 0.1 µs | 0.1 µs |
| forwarded payload lands (second hop) | +13.2 µs | +13.7 µs |
| last peer data in (network total) | 36.0 µs | 26.5 µs |
| rest of the per-op period (GPU stage, wait, reduce, next launch) | 11.7 µs | 11.6 µs |
| **all-reduce** | **49.6 µs** | **40.5 µs** |

- **Originally, the first hop ran at half speed.** Each node sends its payload to both neighbours at once and receives
  two payloads plus a forward. With both ring devices behind one x4 link, three payloads cross it in each direction
  per op.
- **The data confirms it.** "Last peer data in" grows by 0.25 µs per KiB of payload, which is 3 payloads at ~12 GB/s.
  The 96 KiB own write took 22.9 µs to be acknowledged, against 11 µs for the same write alone (perftest).
- **The CPU is not a factor:**
  - the proxy reacts in 0.1–0.2 µs;
  - it already runs on the big X925 cores;
  - pinning it to a big core changed nothing.

**4. The fix: one ring edge per PCIe link.**
- `GLM_ROCE_RING_EXCLUDE=roceP2p1s0f1` makes link discovery take the DCP-pair edge over rocep1s0f1 (domain 0000).
  The other edge stays on roceP2p1s0f0 (0002), so every node's two ring edges use different x4 links.
- The DCP collectives share rocep1s0f1, but they never run at the same time as a TP all-reduce: both are sequential
  in the layer.
- Split mode (`GLM_ROCE_RING_SPLIT=1`) then balances the bytes: each edge carries 1.5 payloads per direction instead
  of 2 and 1.
- Results, 4-rank harness, medians over 4 ranks:

  | T | original | pair edge on other x4 | **+ split** |
  |---:|---:|---:|---:|
  | 1 | 21.4 µs | 19.9 | **19.8** |
  | 8 | 49.6 | 40.5 | **38.5** (−22%) |
  | 16 | 82.4 | 60.8 | **53.0** (−36%) |
  | 32 | 126.9 | 94.9 | **81.9** (−35%) |

- The per-byte slope falls from 0.29 to 0.17 µs/KiB.
- Both configurations pass the full exactness test: eager reference, 300 CUDA-graph replays per size with DCP gathers
  interleaved on the same NIC, 0 errors.
- In production since 2026-10-01 19:06 (`ROCE_TP_EXCLUDE`, `ROCE_TP_SPLIT` in `runtime/vllm029/launch.sh`).

**Live cycle times, fixed K** (ms, median; `results/perbyte-20261001/live/`), compared with boot B, the same stack on
the original link map (`results/step456-integration-20261001/bootB/costgrid/`):

| requests × K | tokens | before | after | change |
|---|---:|---:|---:|---:|
| 1 × 1 | 2 | 71.95 | 71.62 | −0.3 |
| 1 × 7 | 8 | 115.96 | 115.11 | −0.9 |
| 2 × 7 | 16 | 170.22 | 169.10 | −1.1 |
| 4 × 1 | 8 | 130.86 | 132.82 | +2.0 |
| 4 × 3 | 16 | 181.89 | 178.61 | −3.3 |
| 3 × 7 | 24 | 222.38 | 216.27 | **−6.1** |
| 4 × 7 | 32 | 259.53 | 254.64 | **−4.9** |

- **At decode sizes up to 16 tokens** the live change is within about ±2 ms of boot-to-boot noise. The
  microbenchmark predicts about −1.7 ms per pass at 8 tokens.
- **At 24–32 tokens it is a clear 2–3%.** It matches the prediction there (−30 to −45 µs × ~156 all-reduces).
- **Expect gains mainly with 3–4 concurrent requests.**

**What limits it now.** Since the fix, split mode keeps both of each node's PCIe x4 links busy for the whole op, at
1.5 payloads per link per direction. At T=32 that is ~43 µs of a ~54 µs network time; the rest is two write latencies.
That leaves ~10 µs of fixed GPU and launch work.

**Cut-through forwarding was built and measured** (`GLM_ROCE_RING_CHUNKS`, default 1): chunks with their own flags,
each forwarded as it lands.
- It is exact, and it gains nothing in split mode (T=8 34.8 → 37.7 µs, T=32 76.5 → 78.9 µs). No serialized hop is
  left to hide.
- It helps only the unsplit map (T=8 40.5 → 37.4 µs), which still trails split mode.
- The only remaining lever is fewer bytes per node: a reduce-scatter + all-gather design moves 2 payloads per node
  instead of 3. It is worth ~10 µs at T=32 and nothing at T ≤ 8, so it is not built.

## Caveats on the comparison

- **NCCL here is NCCL as configured in production:** Ring forced, one channel, one CTA, and no GPU Direct RDMA on this
  platform.
- **Retuned NCCL in the same harness** (`results/perbyte-20261001/ring-nccl-*`), µs per all-reduce:

  | NCCL setting | T=1 | T=8 | T=32 |
  |---|---:|---:|---:|
  | production (auto protocol, 1 channel) | 51 | 97 | 185 |
  | 4 channels / 4 CTAs | 52 | 71 | 136 |
  | Simple | 80 | 99 | 135 |
  | LL | 53 | 96 | 298 |

  Against the best of these, the cross-domain ring is still **2.6× faster at T=1, 1.8× at T=8 and 1.7× at T=32.**
- More NCCL channels would likely help the prefill all-reduces that stay on NCCL. That is not measured at prefill size.
- **Microbenchmark vs serving.** The tables are back-to-back all-reduces. Serving gains are measured separately as live
  cycle times.
