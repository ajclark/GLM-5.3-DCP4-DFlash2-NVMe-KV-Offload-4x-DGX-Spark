# Serving network: RoCE addressing on the four Sparks

Each Spark's ConnectX-7 exposes every 200 GbE port through **two PCIe functions**, one on each of the NIC's two
PCIe Gen5 x4 links (PCI domains `0000` and `0002`, about 13.6 GB/s each way per link). The four Sparks form a
switchless ring, 06c4 → 365c → ddbf → a218 → 06c4: port 1 cables the DCP pairs (06c4–365c, ddbf–a218), port 0
cables the other ring neighbours (365c–ddbf, a218–06c4).

The x4 host link, not the 200 Gb/s port, limits large transfers
([RDMA collectives latency](RDMA-COLLECTIVES-LATENCY.md)). The serving stack therefore spreads traffic over both
links: RoCEnante stripes the DCP pairs' collectives over both twins of port 1, the TP ring puts its two edges on
different links, and NCCL runs one channel per link for the prefill TP all-reduce.

## Static addresses (NetworkManager profiles, MTU 9000, IPv6 off)

All addresses are in `192.168.x.0/24`.

| profile | function (netdev) | PCI domain / port | 06c4 | 365c | ddbf | a218 |
|---|---|---|---|---|---|---|
| `roce-p0` | `roceP2p1s0f0` (`enP2p1s0f0np0`) | 0002 / port 0 | .104.2 | .102.1 | .102.2 | .104.1 |
| `roce-p2p` | `roceP2p1s0f1` (`enP2p1s0f1np1`) | 0002 / port 1 | .100.1 | .100.2 | .103.2 | .103.1 |
| `roce-p2p-unused` | `rocep1s0f1` (`enp1s0f1np1`) | 0000 / port 1 | .101.1 | .101.2 | .105.2 | .105.1 |
| `roce-p0-twin` | `rocep1s0f0` (`enp1s0f0np0`) | 0000 / port 0 | .106.2 | .107.1 | .107.2 | .106.1 |

`roce-p0-twin` is created by [`node/roce-p0-twin.sh apply`](../node/roce-p0-twin.sh) (`rollback` deletes it). Without
it, NetworkManager runs an in-memory DHCP profile on `enp1s0f0np0` that never gets a lease and flushes manually added
addresses on every retry.

## Who uses which function

| traffic | functions | launcher switch |
|---|---|---|
| DCP pair collectives (RoCEnante, decode and prefill) | `rocep1s0f1` + `roceP2p1s0f1` (both twins of port 1, striped) | `ROCE_DCP_HCAS` |
| TP all-reduce ≤ 1 MiB (RDMA ring, decode and verify) | `rocep1s0f1` (pair edge) + `roceP2p1s0f0` (other edge) | `ROCE_TP_EXCLUDE=roceP2p1s0f1,rocep1s0f0` |
| NCCL (prefill TP all-reduce, TP all-gathers) | all four, merged into one virtual NIC per PCIe link, one channel each | `NCCL_HCAS`, `NCCL_CHANNELS=2` |

NCCL ties each channel's ring to one (virtual) NIC for both directions, and a virtual NIC must reach both ring
neighbours. That is why the domain-0000 pair needs the port-0 twin address: with it, NCCL merges
`rocep1s0f0+rocep1s0f1` and `roceP2p1s0f0+roceP2p1s0f1` and runs one channel on each link. The 25 MB prefill
all-reduce drops from 2.88 ms to 1.81 ms with bit-identical output
([prefill results](../results/prefill-item2-20261006/REPORT.md)).

`start-glm53.sh` refuses to start unless every node has the IPv4 RoCE v2 GID at index 3 on `rocep1s0f0`. Without the
`roce-p0-twin` profile, launch with `NCCL_HCAS='=roceP2p1s0f0,roceP2p1s0f1' ROCE_TP_EXCLUDE=roceP2p1s0f1` (the
single-link layout, about 5% slower prefill).

## Node settings

- GPU clocks are locked at about 2000 MHz by `gpu-clock-lock.service` on every node; all measurements use that lock.
- The profiles above, the clock lock and the kernel survive a reboot. The serving containers use restart policy `no`:
  after a reboot, run `bash start-glm53.sh`.
