// RDMA proxy for one-shot collectives on a four-node ring with no switch.
//
// Derived from b12x/comm/roce/_roce_proxy.c (local-inference-lab/b12x b58f34ea,
// Apache-2.0; RoCEnante by Luke Alonso and Jason Cook) as patched in this repository
// (idle backoff, slot bound). The region layout, doorbell and GPU kernels are b12x's;
// what changes is the transport: b12x writes every payload to every peer over an
// all-peer fabric, while here rank r only has queue pairs to its ring neighbours
// r+1 (HCA 0, "cw") and r-1 (HCA 1, "ccw"), and rank r+2 is reached by forwarding.
//
// Per collective with sequence number s (slot s & 1), each rank's proxy strictly
// alternates  own(s) -> forward(s) -> own(s+1) -> ...:
//   own(s):     once the local kernel rang the doorbell for s and forward(s-1) has
//               completed, write send[slot] into recv[r][slot] of r+1 (HCA 0) and of
//               r-1 (HCA 1), each followed on the same RC queue pair by the 4-byte
//               flag s in flag[r][slot].  The byte count is read from the doorbell's
//               per-slot word and remembered for forward(s).
//   forward(s): once own(s) is posted and flag[r-1][slot] in r's own region reads s
//               (rank r-1's payload landed), write recv[r-1][slot] into
//               recv[r-1][slot] of r+1, followed by the flag s.  So x_{r-1} reaches
//               r+1 = (r-1)+2.
// Every rank thus receives all three peers' payloads with one flag per source: the
// layout and flag protocol the b12x one-shot and all-gather kernels use with
// hca_count = 1, so those kernels run unchanged.
//
// Why this is safe with two slots (x_q[s] is rank q's payload for op s):
//   * recv[r-1][slot] at r is rewritten by x_{r-1}[s+2], which r-1 posts after its
//     doorbell s+2, i.e. after its kernel s+1 finished, which needed x_r[s+1], which
//     r posts only after forward(s) completed (its reads of that slot are done).
//   * The forwarded copy in recv[r-1][slot] at r+1 is rewritten by forward(s+2) from
//     r, which follows r's own(s+2), doorbell s+2 and r's kernel s+1, which needed
//     x_{r+1}[s+1], which r+1 posts after its doorbell s+1, i.e. after its kernel s
//     consumed the slot.  Direct payloads follow the b12x argument unchanged.
//   * Doorbells: r's kernel s+1 needs x_{r+1}[s+1]; r+1 posts it after forward(s) at
//     r+1, which needs x_r[s], so doorbell s+2 cannot ring before own(s) is posted:
//     at most two doorbells are pending and the per-slot byte count read by own(s)
//     is still s's.
//   * No cycle: own(s) everywhere depends only on step s-1 and the local doorbell;
//     forward(s) on own(s) of the forwarder and of its ccw neighbour.
//
// Split mode (ring_create(..., split=1); kernels compiled with hca_count = 2, i.e. two
// flags per source): every payload is cut into two stripes (b12x's stripe split),
// stripe 0 travels clockwise and stripe 1 counter-clockwise, so each link direction
// carries 1.5 payloads per op instead of 2 (cw) and 1 (ccw), and each forwarded hop
// moves half a payload:
//   own(s):       to r+1 stripe 0 then stripe 1, to r-1 stripe 1 then stripe 0 (the
//                 stripe the neighbour forwards goes first), each with its own flag
//                 flag[r][slot][stripe].
//   forward_cw:   stripe 0 of x_{r-1} to r+1 once flag[r-1][slot][0] reads s.
//   forward_ccw:  stripe 1 of x_{r+1} to r-1 once flag[r+1][slot][1] reads s.
//   own(s+1) waits for both forwards of s to complete.  The safety argument above
//   holds for each direction separately (mirror it for the ccw stream).
//
// Cut-through (ring_create(..., chunks=C > 1)): every payload is cut into F = D*C pieces
// (D = 2 when split, else 1), each with its own flag flag[src][slot][piece]; the kernels run
// with hca_count = F.  Piece p belongs to direction p / C and is chunk p % C of it.  own(s)
// posts all F pieces to each neighbour as one chained post (the pieces that neighbour
// forwards first), and a forward stream passes each of its C pieces on as soon as that
// piece's flag lands, instead of waiting for the whole payload, so the second hop overlaps
// the first.  A stream's forward of s is complete when the chain carrying its last piece
// completes; own(s+1) waits for that exactly as above, so the slot argument is unchanged
// (each piece is a sub-range of the same slot, forwarded once, after it landed).

// Idle behaviour is the patched b12x one: hot for B12X_ROCE_IDLE_HOT_US after the
// last activity, then naps doubling up to B12X_ROCE_IDLE_MAX_NAP_US.
//
// Compiled at image build (hardened flags); plain C over libibverbs, no CUDA.

#define _GNU_SOURCE
#include <errno.h>
#include <infiniband/verbs.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define RING_WORLD 4
#define RING_HCAS 2            // 0 = to rank+1 (cw), 1 = to rank-1 (ccw)
#define ROCE_SLOTS 2
#define RING_MAX_FLAGS 8       // flags per (source, slot): pieces = (split ? 2 : 1) * chunks
#define ROCE_FLAG_STRIDE 128
#define ROCE_PORT 1
#define ROCE_SEND_DEPTH 1024
#define RING_CHAIN_CAP 32      // outstanding signaled chains per HCA (each <= 2 * RING_MAX_FLAGS WRs)
#define RING_ABI_VERSION 3
#define ROCE_IDLE_HOT_NS_DEFAULT 50000000ull
#define ROCE_IDLE_MIN_NAP_NS 50000ull
#define ROCE_IDLE_MAX_NAP_NS_DEFAULT 2000000ull
#define KIND_OWN 1ull
#define KIND_FWD_CW 2ull
#define KIND_FWD_CCW 3ull
#define KIND_LAST 4ull         // or'ed into a forward chain's kind when it carries the stream's last piece

#ifdef RING_TRACE
// Diagnostic build only (-DRING_TRACE; GLM_ROCE_RING_TRACE=1 in ring.py): per-op CLOCK_MONOTONIC
// timestamps taken by the proxy thread, kept in a ring of RING_TRACE_N records indexed by seq.
#define RING_TRACE_N 16384u
typedef struct {
    uint32_t seq, nbytes;
    uint64_t t_db;               // doorbell for seq first seen
    uint64_t t_own_post;         // own payload + flag posted on both HCAs
    uint64_t t_own_cqe[2];       // own flag write completed (acked) on HCA 0 (cw) / 1 (ccw)
    uint64_t t_arr[RING_WORLD];  // every piece flag of src seen == seq (src != rank)
    uint64_t t_first[RING_WORLD]; // first piece flag of src seen
    uint64_t t_fwd_post;         // first forward chain (cw) posted
    uint64_t t_fwd_cqe;          // forward completed
} ring_trace_t;
#endif

typedef struct {
    uint64_t region_addr;
    uint32_t rkey[RING_HCAS];
    uint16_t lid[RING_HCAS];
    uint8_t gid[RING_HCAS][16];
    uint32_t mtu[RING_HCAS];
    uint32_t qp_num[RING_HCAS];
} ring_blob_t;

typedef struct {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_mr *mr;
    struct ibv_cq *cq;
    struct ibv_qp *qp;          // to this HCA's neighbour
    uint32_t outstanding;       // signaled work requests not yet completed
    uint64_t writes_completed;
    uint64_t bytes_posted;
    union ibv_gid gid;
    uint16_t lid;
    enum ibv_mtu mtu;
    uint64_t peer_addr;         // neighbour's region
    uint32_t peer_rkey;         // neighbour's MR on its HCA facing us
} ring_hca_t;

typedef struct {
    int rank;
    int gid_index;
    ring_hca_t hca[RING_HCAS];
    uint8_t *region;
    size_t region_bytes;
    size_t slot_bytes;
    size_t recv_off;
    size_t flag_off;
    size_t send_off;
    size_t ctrl_off;
    int started;
    pthread_t thread;
    atomic_int running;
    atomic_int failed;
    int split;                  // 1: direction 0 travels cw, direction 1 ccw
    int nchunks;                // C: chunks per direction (cut-through when > 1)
    int nflag;                  // F = (split ? 2 : 1) * C pieces (flags) per source
    uint32_t last_seq;          // own ops posted
    uint32_t fwd_posted[2];     // ops whose forward is fully posted: [0] cw (x_{r-1}), [1] ccw (x_{r+1})
    uint32_t fwd_piece[2];      // next chunk to forward of op fwd_posted[d] + 1
    uint32_t fwd_done[2];       // forwards completed (last chain acknowledged)
    uint32_t own_nbytes[ROCE_SLOTS]; // byte count of the own op in each slot
    uint64_t idle_hot_ns;
    uint64_t idle_max_nap_ns;
    uint64_t ops_posted;
    uint64_t fwd_count;
    uint64_t writes_completed;
    char err[512];
#ifdef RING_TRACE
    ring_trace_t *trace;
#endif
} ring_ctx_t;

void ring_destroy(ring_ctx_t *c);

static void set_err(ring_ctx_t *c, const char *what, int e) {
    snprintf(c->err, sizeof(c->err), "%s: %s", what, e ? strerror(e) : "failed");
}

static uint64_t monotonic_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

#ifdef RING_TRACE
static ring_trace_t *trace_rec(ring_ctx_t *c, uint32_t seq) {
    ring_trace_t *r = &c->trace[seq % RING_TRACE_N];
    if (r->seq != seq) {
        memset(r, 0, sizeof(*r));
        r->seq = seq;
    }
    return r;
}

// Record, for the current and the next op, when each peer's first and last piece flag landed.
static void trace_arrivals(ring_ctx_t *c) {
    uint64_t now = 0;
    for (uint32_t s = c->last_seq; s != c->last_seq + 2u; s++) {
        if (s == 0) {
            continue;
        }
        for (int src = 0; src < RING_WORLD; src++) {
            if (src == c->rank) {
                continue;
            }
            int landed = 0;
            for (int p = 0; p < c->nflag; p++) {
                volatile uint32_t *f = (volatile uint32_t *)(c->region + c->flag_off +
                    (((uint64_t)src * ROCE_SLOTS + (s & 1u)) * (uint64_t)c->nflag + (uint64_t)p) * ROCE_FLAG_STRIDE);
                landed += __atomic_load_n(f, __ATOMIC_ACQUIRE) == s;
            }
            if (landed == 0) {
                continue;
            }
            ring_trace_t *r = trace_rec(c, s);
            if (now == 0) {
                now = monotonic_ns();
            }
            if (r->t_first[src] == 0) {
                r->t_first[src] = now;
            }
            if (landed == c->nflag && r->t_arr[src] == 0) {
                r->t_arr[src] = now;
            }
        }
    }
}

uint64_t ring_trace_record_bytes(void) { return sizeof(ring_trace_t); }

uint64_t ring_trace_dump(ring_ctx_t *c, void *out, uint64_t max_records) {
    uint64_t n = max_records < RING_TRACE_N ? max_records : RING_TRACE_N;
    memcpy(out, c->trace, n * sizeof(ring_trace_t));
    return n;
}
#endif

static uint64_t env_us_as_ns(const char *name, uint64_t default_ns) {
    const char *raw = getenv(name);
    if (raw == NULL || *raw == '\0') {
        return default_ns;
    }
    char *end = NULL;
    unsigned long long us = strtoull(raw, &end, 10);
    if (end == raw || *end != '\0' || us > 60000000ull) {
        return default_ns;
    }
    return (uint64_t)us * 1000ull;
}

int ring_abi_version(void) { return RING_ABI_VERSION; }

// b12x's roce_layout with room for RING_MAX_FLAGS flags per (source, slot); the b12x kernels
// address flags as ((src * slots + slot) * hca_count + piece) * stride, so they run unchanged.
int ring_layout(int world, uint64_t slot_bytes, uint64_t *out) {
    if (world != RING_WORLD || slot_bytes == 0 || (slot_bytes % 4096) != 0) {
        return -1;
    }
    uint64_t recv_bytes, flag_bytes, send_bytes, send_off, ctrl_off, total;
    if (slot_bytes > ((uint64_t)1 << 40) ||
        __builtin_mul_overflow((uint64_t)world * ROCE_SLOTS, slot_bytes, &recv_bytes) ||
        __builtin_mul_overflow((uint64_t)world * ROCE_SLOTS * RING_MAX_FLAGS,
                               (uint64_t)ROCE_FLAG_STRIDE, &flag_bytes) ||
        __builtin_mul_overflow((uint64_t)ROCE_SLOTS, slot_bytes, &send_bytes) ||
        __builtin_add_overflow(recv_bytes, flag_bytes, &send_off) ||
        __builtin_add_overflow(send_off, send_bytes, &ctrl_off) ||
        __builtin_add_overflow(ctrl_off, (uint64_t)ROCE_FLAG_STRIDE, &total)) {
        return -1;
    }
    out[0] = 0;
    out[1] = recv_bytes;
    out[2] = send_off;
    out[3] = ctrl_off;
    out[4] = total;
    out[5] = ROCE_FLAG_STRIDE;
    out[6] = ROCE_SLOTS;
    return 0;
}

uint64_t ring_blob_bytes(void) { return sizeof(ring_blob_t); }

static int open_hca(ring_ctx_t *c, int h, const char *name) {
    int num = 0;
    struct ibv_device **list = ibv_get_device_list(&num);
    if (list == NULL) {
        set_err(c, "ibv_get_device_list", errno);
        return -1;
    }
    struct ibv_device *dev = NULL;
    for (int i = 0; i < num; i++) {
        if (strcmp(ibv_get_device_name(list[i]), name) == 0) {
            dev = list[i];
            break;
        }
    }
    if (dev == NULL) {
        ibv_free_device_list(list);
        snprintf(c->err, sizeof(c->err), "RDMA device %s not found", name);
        return -1;
    }
    ring_hca_t *hca = &c->hca[h];
    hca->ctx = ibv_open_device(dev);
    ibv_free_device_list(list);
    if (hca->ctx == NULL) {
        set_err(c, "ibv_open_device", errno);
        return -1;
    }
    struct ibv_port_attr port;
    if (ibv_query_port(hca->ctx, ROCE_PORT, &port) != 0) {
        set_err(c, "ibv_query_port", errno);
        return -1;
    }
    if (port.state != IBV_PORT_ACTIVE) {
        snprintf(c->err, sizeof(c->err), "RDMA device %s port %d is not active", name, ROCE_PORT);
        return -1;
    }
    hca->lid = port.lid;
    hca->mtu = port.active_mtu;
    if (ibv_query_gid(hca->ctx, ROCE_PORT, c->gid_index, &hca->gid) != 0) {
        set_err(c, "ibv_query_gid", errno);
        return -1;
    }
    hca->pd = ibv_alloc_pd(hca->ctx);
    if (hca->pd == NULL) {
        set_err(c, "ibv_alloc_pd", errno);
        return -1;
    }
    hca->mr = ibv_reg_mr(hca->pd, c->region, c->region_bytes,
                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (hca->mr == NULL) {
        set_err(c, "ibv_reg_mr(pinned region)", errno);
        return -1;
    }
    hca->cq = ibv_create_cq(hca->ctx, ROCE_SEND_DEPTH, NULL, NULL, 0);
    if (hca->cq == NULL) {
        set_err(c, "ibv_create_cq", errno);
        return -1;
    }
    struct ibv_qp_init_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.send_cq = hca->cq;
    attr.recv_cq = hca->cq;
    attr.qp_type = IBV_QPT_RC;
    attr.cap.max_send_wr = ROCE_SEND_DEPTH;
    attr.cap.max_recv_wr = 1;
    attr.cap.max_send_sge = 1;
    attr.cap.max_recv_sge = 1;
    attr.cap.max_inline_data = 16;
    hca->qp = ibv_create_qp(hca->pd, &attr);
    if (hca->qp == NULL) {
        set_err(c, "ibv_create_qp", errno);
        return -1;
    }
    struct ibv_qp_attr init;
    memset(&init, 0, sizeof(init));
    init.qp_state = IBV_QPS_INIT;
    init.pkey_index = 0;
    init.port_num = ROCE_PORT;
    init.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
    int rc = ibv_modify_qp(hca->qp, &init,
                           IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    if (rc != 0) {
        set_err(c, "ibv_modify_qp(INIT)", rc);
        return -1;
    }
    return 0;
}

ring_ctx_t *ring_create(int world, int rank, const char *hca_cw, const char *hca_ccw,
                        int gid_index, int split, int chunks, void *region, uint64_t region_bytes,
                        uint64_t slot_bytes, char *err, uint64_t err_len) {
    if (chunks < 1 || (split ? 2 : 1) * chunks > RING_MAX_FLAGS) {
        snprintf(err, err_len, "ring chunks %d out of range (pieces per source <= %d)", chunks, RING_MAX_FLAGS);
        return NULL;
    }
    uint64_t layout[7];
    if (ring_layout(world, slot_bytes, layout) != 0 || layout[4] > region_bytes ||
        rank < 0 || rank >= world) {
        snprintf(err, err_len, "invalid ring runtime geometry");
        return NULL;
    }
    ring_ctx_t *c = calloc(1, sizeof(*c));
    if (c == NULL) {
        snprintf(err, err_len, "out of memory");
        return NULL;
    }
    c->rank = rank;
    c->gid_index = gid_index;
    c->split = split ? 1 : 0;
    c->nchunks = chunks;
    c->nflag = (split ? 2 : 1) * chunks;
#ifdef RING_TRACE
    c->trace = calloc(RING_TRACE_N, sizeof(ring_trace_t));
    if (c->trace == NULL) {
        snprintf(err, err_len, "out of memory (trace)");
        free(c);
        return NULL;
    }
#endif
    c->region = region;
    c->region_bytes = region_bytes;
    c->slot_bytes = slot_bytes;
    c->idle_hot_ns = env_us_as_ns("B12X_ROCE_IDLE_HOT_US", ROCE_IDLE_HOT_NS_DEFAULT);
    c->idle_max_nap_ns = env_us_as_ns("B12X_ROCE_IDLE_MAX_NAP_US", ROCE_IDLE_MAX_NAP_NS_DEFAULT);
    if (c->idle_max_nap_ns < ROCE_IDLE_MIN_NAP_NS) {
        c->idle_max_nap_ns = ROCE_IDLE_MIN_NAP_NS;
    }
    c->recv_off = layout[0];
    c->flag_off = layout[1];
    c->send_off = layout[2];
    c->ctrl_off = layout[3];
    const char *names[RING_HCAS] = {hca_cw, hca_ccw};
    for (int h = 0; h < RING_HCAS; h++) {
        if (open_hca(c, h, names[h]) != 0) {
            snprintf(err, err_len, "%s", c->err);
            ring_destroy(c);
            return NULL;
        }
    }
    return c;
}

int ring_local_blob(ring_ctx_t *c, void *out, uint64_t out_len) {
    if (out_len < sizeof(ring_blob_t)) {
        return -1;
    }
    ring_blob_t blob;
    memset(&blob, 0, sizeof(blob));
    blob.region_addr = (uint64_t)(uintptr_t)c->region;
    for (int h = 0; h < RING_HCAS; h++) {
        blob.rkey[h] = c->hca[h].mr->rkey;
        blob.lid[h] = c->hca[h].lid;
        blob.mtu[h] = (uint32_t)c->hca[h].mtu;
        memcpy(blob.gid[h], c->hca[h].gid.raw, 16);
        blob.qp_num[h] = c->hca[h].qp->qp_num;
    }
    memcpy(out, &blob, sizeof(blob));
    return 0;
}

// Connect our HCA h to the neighbour's HCA `ph` (our cw meets the next rank's ccw).
static int connect_qp(ring_ctx_t *c, int h, const ring_blob_t *peer, int ph) {
    ring_hca_t *hca = &c->hca[h];
    struct ibv_qp_attr rtr;
    memset(&rtr, 0, sizeof(rtr));
    rtr.qp_state = IBV_QPS_RTR;
    rtr.path_mtu = (enum ibv_mtu)(peer->mtu[ph] < (uint32_t)hca->mtu ? peer->mtu[ph] : (uint32_t)hca->mtu);
    rtr.dest_qp_num = peer->qp_num[ph];
    rtr.rq_psn = 0;
    rtr.max_dest_rd_atomic = 1;
    rtr.min_rnr_timer = 12;
    rtr.ah_attr.is_global = 1;
    rtr.ah_attr.dlid = peer->lid[ph];
    rtr.ah_attr.sl = 0;
    rtr.ah_attr.src_path_bits = 0;
    rtr.ah_attr.port_num = ROCE_PORT;
    memcpy(rtr.ah_attr.grh.dgid.raw, peer->gid[ph], 16);
    rtr.ah_attr.grh.sgid_index = (uint8_t)c->gid_index;
    rtr.ah_attr.grh.hop_limit = 64;
    rtr.ah_attr.grh.traffic_class = 0;
    rtr.ah_attr.grh.flow_label = 0;
    int rc = ibv_modify_qp(hca->qp, &rtr,
                           IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                               IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    if (rc != 0) {
        set_err(c, "ibv_modify_qp(RTR)", rc);
        return -1;
    }
    struct ibv_qp_attr rts;
    memset(&rts, 0, sizeof(rts));
    rts.qp_state = IBV_QPS_RTS;
    rts.timeout = 14;
    rts.retry_cnt = 7;
    rts.rnr_retry = 7;
    rts.sq_psn = 0;
    rts.max_rd_atomic = 1;
    rc = ibv_modify_qp(hca->qp, &rts,
                       IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                           IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
    if (rc != 0) {
        set_err(c, "ibv_modify_qp(RTS)", rc);
        return -1;
    }
    hca->peer_addr = peer->region_addr;
    hca->peer_rkey = peer->rkey[ph];
    return 0;
}

// blobs: RING_WORLD blobs in rank order.
int ring_connect(ring_ctx_t *c, const void *blobs, uint64_t blobs_len) {
    if (blobs_len < sizeof(ring_blob_t) * (uint64_t)RING_WORLD) {
        snprintf(c->err, sizeof(c->err), "peer blob buffer too small");
        return -1;
    }
    const ring_blob_t *all = (const ring_blob_t *)blobs;
    int next = (c->rank + 1) % RING_WORLD;
    int prev = (c->rank + RING_WORLD - 1) % RING_WORLD;
    if (connect_qp(c, 0, &all[next], 1) != 0 || connect_qp(c, 1, &all[prev], 0) != 0) {
        return -1;
    }
    return 0;
}

static int drain_cq(ring_ctx_t *c, int h) {
    struct ibv_wc wc[32];
    int n = ibv_poll_cq(c->hca[h].cq, 32, wc);
    if (n < 0) {
        set_err(c, "ibv_poll_cq", errno);
        return -1;
    }
    for (int i = 0; i < n; i++) {
        uint64_t kind = wc[i].wr_id >> 32;
        uint64_t base = kind & 3ull;
        uint32_t seq = (uint32_t)wc[i].wr_id;
        if (wc[i].status != IBV_WC_SUCCESS) {
            snprintf(c->err, sizeof(c->err),
                     "RDMA %s write on HCA %d (seq %u) failed: %s (vendor_err 0x%x)",
                     base == KIND_OWN ? "payload" : "forward", h, seq,
                     ibv_wc_status_str(wc[i].status), wc[i].vendor_err);
            return -1;
        }
        c->hca[h].outstanding -= 1;
        c->hca[h].writes_completed += 1;
        c->writes_completed += 1;
#ifdef RING_TRACE
        if (base == KIND_OWN) {
            trace_rec(c, seq)->t_own_cqe[h] = monotonic_ns();
        } else if (kind == (KIND_FWD_CW | KIND_LAST)) {
            trace_rec(c, seq)->t_fwd_cqe = monotonic_ns();
        }
#endif
        if ((kind & KIND_LAST) && (base == KIND_FWD_CW || base == KIND_FWD_CCW)) {
            uint32_t *done = &c->fwd_done[base == KIND_FWD_CCW];
            if ((int32_t)(seq - *done) > 0) {
                *done = seq;
            }
        }
    }
    return 0;
}

static int drain_all(ring_ctx_t *c) {
    for (int h = 0; h < RING_HCAS; h++) {
        if (drain_cq(c, h) != 0) {
            return -1;
        }
    }
    return 0;
}

// Byte range of piece p of an nbytes payload cut into nflag pieces (whole 16-byte packs; a piece
// of a tiny payload may be empty, then only its flag is written).
static void piece_range(const ring_ctx_t *c, uint32_t nbytes, int p, uint64_t *off, uint32_t *len) {
    uint64_t packs = nbytes / 16u;
    uint64_t a = packs * (uint64_t)p / (uint64_t)c->nflag;
    uint64_t b = packs * (uint64_t)(p + 1) / (uint64_t)c->nflag;
    *off = a * 16u;
    *len = (uint32_t)((b - a) * 16u);
}

// Post pieces[0..n) of op `seq` from source rank `src_rank` on HCA h as one chain: per piece
// the bytes (local `base + off` -> the neighbour's recv[src_rank][slot] + off) followed by the
// 4-byte flag seq in its flag[src_rank][slot][piece].  RC ordering keeps every flag behind its
// bytes; only the chain's last flag is signaled, so its completion implies the whole chain.
static int post_chain(ring_ctx_t *c, int h, int src_rank, uint32_t seq, uint32_t nbytes,
                      const int *pieces, int n, uint8_t *base, uint64_t kind) {
    ring_hca_t *hca = &c->hca[h];
    while (hca->outstanding >= RING_CHAIN_CAP) {
        if (drain_cq(c, h) != 0) {
            return -1;
        }
        if (!atomic_load_explicit(&c->running, memory_order_relaxed)) {
            snprintf(c->err, sizeof(c->err), "ring proxy stopped with %u chains outstanding on HCA %d",
                     hca->outstanding, h);
            return -1;
        }
    }
    uint32_t slot = seq & 1u;
    uint32_t seq_copy = seq;
    struct ibv_sge sge[2 * RING_MAX_FLAGS];
    struct ibv_send_wr wr[2 * RING_MAX_FLAGS];
    int w = 0;
    uint64_t bytes = 0;
    for (int i = 0; i < n; i++) {
        int p = pieces[i];
        uint64_t off;
        uint32_t len;
        piece_range(c, nbytes, p, &off, &len);
        if (len != 0) {
            sge[w] = (struct ibv_sge){.addr = (uint64_t)(uintptr_t)(base + off), .length = len, .lkey = hca->mr->lkey};
            memset(&wr[w], 0, sizeof(wr[w]));
            wr[w].wr_id = (kind << 32) | seq;
            wr[w].sg_list = &sge[w];
            wr[w].num_sge = 1;
            wr[w].opcode = IBV_WR_RDMA_WRITE;
            wr[w].wr.rdma.remote_addr = hca->peer_addr + c->recv_off +
                                        ((uint64_t)src_rank * ROCE_SLOTS + slot) * c->slot_bytes + off;
            wr[w].wr.rdma.rkey = hca->peer_rkey;
            bytes += len;
            w++;
        }
        sge[w] = (struct ibv_sge){.addr = (uint64_t)(uintptr_t)&seq_copy, .length = 4, .lkey = 0};
        memset(&wr[w], 0, sizeof(wr[w]));
        wr[w].wr_id = (kind << 32) | seq;
        wr[w].sg_list = &sge[w];
        wr[w].num_sge = 1;
        wr[w].opcode = IBV_WR_RDMA_WRITE;
        wr[w].send_flags = IBV_SEND_INLINE;
        wr[w].wr.rdma.remote_addr = hca->peer_addr + c->flag_off +
            (((uint64_t)src_rank * ROCE_SLOTS + slot) * (uint64_t)c->nflag + (uint64_t)p) * ROCE_FLAG_STRIDE;
        wr[w].wr.rdma.rkey = hca->peer_rkey;
        w++;
    }
    for (int i = 0; i + 1 < w; i++) {
        wr[i].next = &wr[i + 1];
    }
    wr[w - 1].next = NULL;
    wr[w - 1].send_flags |= IBV_SEND_SIGNALED;
    struct ibv_send_wr *bad = NULL;
    int rc = ibv_post_send(hca->qp, &wr[0], &bad);
    if (rc != 0) {
        set_err(c, "ibv_post_send", rc);
        return -1;
    }
    hca->outstanding += 1;
    hca->bytes_posted += bytes;
    return 0;
}

static int check_nbytes(ring_ctx_t *c, uint32_t nbytes, const char *what) {
    if (nbytes == 0 || nbytes % 16 != 0 || (uint64_t)nbytes > (uint64_t)c->slot_bytes) {
        snprintf(c->err, sizeof(c->err), "ring %s payload of %u bytes is not a positive multiple "
                 "of 16 within the %zu-byte slot", what, nbytes, c->slot_bytes);
        return -1;
    }
    return 0;
}

// Forward stream d (0 = cw: x_{r-1}'s direction-0 pieces to r+1; 1 = ccw, split only: x_{r+1}'s
// direction-1 pieces to r-1) of op `want`: post every further piece whose flag has landed, in
// chunk order, as one chain.  Returns 1 when something was posted, -1 on error.
static int try_forward(ring_ctx_t *c, int d, uint32_t want) {
    int src = d == 0 ? (c->rank + RING_WORLD - 1) % RING_WORLD : (c->rank + 1) % RING_WORLD;
    uint32_t slot = want & 1u;
    int pieces[RING_MAX_FLAGS];
    int n = 0;
    uint32_t k = c->fwd_piece[d];
    while (k < (uint32_t)c->nchunks) {
        int p = d * c->nchunks + (int)k;
        volatile uint32_t *in = (volatile uint32_t *)(c->region + c->flag_off +
            (((uint64_t)src * ROCE_SLOTS + slot) * (uint64_t)c->nflag + (uint64_t)p) * ROCE_FLAG_STRIDE);
        if (__atomic_load_n(in, __ATOMIC_ACQUIRE) != want) {
            break;
        }
        pieces[n++] = p;
        k++;
    }
    if (n == 0) {
        return 0;
    }
    int last = k == (uint32_t)c->nchunks;
    uint64_t kind = (d == 0 ? KIND_FWD_CW : KIND_FWD_CCW) | (last ? KIND_LAST : 0);
    uint8_t *base = c->region + c->recv_off + ((uint64_t)src * ROCE_SLOTS + slot) * c->slot_bytes;
#ifdef RING_TRACE
    if (d == 0 && c->fwd_piece[0] == 0) {
        trace_rec(c, want)->t_fwd_post = monotonic_ns();
    }
#endif
    if (post_chain(c, d, src, want, c->own_nbytes[slot], pieces, n, base, kind) != 0) {
        return -1;
    }
    c->fwd_count += (uint64_t)n;
    if (last) {
        c->fwd_posted[d] = want;
        c->fwd_piece[d] = 0;
    } else {
        c->fwd_piece[d] = k;
    }
    return 1;
}

static void *proxy_main(void *arg) {
    ring_ctx_t *c = (ring_ctx_t *)arg;
    volatile uint32_t *ctrl = (volatile uint32_t *)(c->region + c->ctrl_off);
    int streams = c->split ? 2 : 1;
    uint64_t idle = 0;
    uint64_t nap_ns = 0;
    uint64_t last_active_ns = monotonic_ns();
    while (atomic_load_explicit(&c->running, memory_order_relaxed)) {
        int progress = 0;
        // 1. own(next): doorbell rung for it, and every forward of the previous op completed.
        uint32_t seq = __atomic_load_n(&ctrl[0], __ATOMIC_ACQUIRE);
#ifdef RING_TRACE
        if (seq != c->last_seq) {
            ring_trace_t *r = trace_rec(c, c->last_seq + 1);
            if (r->t_db == 0) {
                r->t_db = monotonic_ns();
                r->nbytes = ctrl[4 + ((c->last_seq + 1) & 1u)];
            }
        }
        trace_arrivals(c);
#endif
        if (seq != c->last_seq && c->fwd_done[0] == c->last_seq &&
            (!c->split || c->fwd_done[1] == c->last_seq)) {
            uint32_t pending = seq - c->last_seq;
            if (pending > ROCE_SLOTS) {
                snprintf(c->err, sizeof(c->err), "doorbell skipped %u ops (last %u, now %u)",
                         pending, c->last_seq, seq);
                atomic_store(&c->failed, 1);
                return NULL;
            }
            uint32_t next = c->last_seq + 1;
            uint32_t nbytes = ctrl[4 + (next & 1u)];
            if (check_nbytes(c, nbytes, "own") != 0) {
                atomic_store(&c->failed, 1);
                return NULL;
            }
            uint8_t *send = c->region + c->send_off + (uint64_t)(next & 1u) * c->slot_bytes;
            // To each neighbour, the pieces it forwards go first: the cw neighbour forwards
            // direction 0, the ccw neighbour direction 1 (split only).
            int order[RING_HCAS][RING_MAX_FLAGS];
            for (int h = 0; h < RING_HCAS; h++) {
                int first = (c->split && h == 1) ? 1 : 0;
                for (int i = 0; i < c->nflag; i++) {
                    int dir = i < c->nchunks ? first : 1 - first;
                    order[h][i] = c->split ? dir * c->nchunks + i % c->nchunks : i;
                }
            }
            if (post_chain(c, 0, c->rank, next, nbytes, order[0], c->nflag, send, KIND_OWN) != 0 ||
                post_chain(c, 1, c->rank, next, nbytes, order[1], c->nflag, send, KIND_OWN) != 0) {
                atomic_store(&c->failed, 1);
                return NULL;
            }
            c->own_nbytes[next & 1u] = nbytes;
            c->last_seq = next;
            c->ops_posted += 1;
#ifdef RING_TRACE
            trace_rec(c, next)->t_own_post = monotonic_ns();
#endif
            progress = 1;
        }
        // 2. forwards of the op we joined last, once their stripes landed.
        for (int d = 0; d < streams; d++) {
            uint32_t want = c->fwd_posted[d] + 1;
            if (c->last_seq == want) {
                int rc = try_forward(c, d, want);
                if (rc < 0) {
                    atomic_store(&c->failed, 1);
                    return NULL;
                }
                progress |= rc;
            }
        }
        // 3. completions, every pass while anything is in flight.
        if (c->hca[0].outstanding || c->hca[1].outstanding || (++idle % 64) == 0) {  // chains in flight
            if (drain_all(c) != 0) {
                atomic_store(&c->failed, 1);
                return NULL;
            }
        }
        if (progress || c->hca[0].outstanding || c->hca[1].outstanding ||
            c->fwd_posted[0] != c->last_seq || (c->split && c->fwd_posted[1] != c->last_seq)) {
            // Busy, or waiting for a neighbour's stripe of an op we joined.
            idle = progress ? 0 : idle;
            nap_ns = 0;
            last_active_ns = monotonic_ns();
            continue;
        }
        if (nap_ns != 0) {
            struct timespec nap = {(time_t)(nap_ns / 1000000000ull), (long)(nap_ns % 1000000000ull)};
            nanosleep(&nap, NULL);
            nap_ns = nap_ns * 2 < c->idle_max_nap_ns ? nap_ns * 2 : c->idle_max_nap_ns;
        } else if ((idle & 1023u) == 0 && monotonic_ns() - last_active_ns >= c->idle_hot_ns) {
            nap_ns = ROCE_IDLE_MIN_NAP_NS;
        }
    }
    return NULL;
}

int ring_start(ring_ctx_t *c) {
    if (atomic_load(&c->running)) {
        return 0;
    }
    if (!c->started) {
        volatile uint32_t *ctrl = (volatile uint32_t *)(c->region + c->ctrl_off);
        c->last_seq = ctrl[0];
        for (int d = 0; d < 2; d++) {
            c->fwd_posted[d] = c->last_seq;
            c->fwd_piece[d] = 0;
            c->fwd_done[d] = c->last_seq;
        }
        c->started = 1;
    }
    atomic_store(&c->failed, 0);
    atomic_store(&c->running, 1);
    int rc = pthread_create(&c->thread, NULL, proxy_main, c);
    if (rc != 0) {
        atomic_store(&c->running, 0);
        set_err(c, "pthread_create", rc);
        return -1;
    }
    return 0;
}

void ring_stop(ring_ctx_t *c) {
    if (atomic_exchange(&c->running, 0)) {
        pthread_join(c->thread, NULL);
    }
}

int ring_failed(ring_ctx_t *c) { return atomic_load(&c->failed); }

const char *ring_error(ring_ctx_t *c) { return c->err; }

uint64_t ring_stat(ring_ctx_t *c, int which) {
    switch (which) {
    case 0:
        return c->ops_posted;
    case 1:
        return c->writes_completed;
    case 2:
        return c->last_seq;
    case 3:
        return c->fwd_count;
    case 4:
        return c->fwd_done[0];
    case 5:
        return (uint64_t)c->split;
    case 6:
        return (uint64_t)c->nchunks;
    default:
        return 0;
    }
}

uint64_t ring_hca_stat(ring_ctx_t *c, int hca, int which) {
    if (hca < 0 || hca >= RING_HCAS) {
        return 0;
    }
    return which == 0 ? c->hca[hca].writes_completed : c->hca[hca].bytes_posted;
}

void ring_destroy(ring_ctx_t *c) {
    if (c == NULL) {
        return;
    }
    ring_stop(c);
    for (int h = 0; h < RING_HCAS; h++) {
        ring_hca_t *hca = &c->hca[h];
        if (hca->qp != NULL) {
            ibv_destroy_qp(hca->qp);
        }
        if (hca->cq != NULL) {
            ibv_destroy_cq(hca->cq);
        }
        if (hca->mr != NULL) {
            ibv_dereg_mr(hca->mr);
        }
        if (hca->pd != NULL) {
            ibv_dealloc_pd(hca->pd);
        }
        if (hca->ctx != NULL) {
            ibv_close_device(hca->ctx);
        }
    }
#ifdef RING_TRACE
    free(c->trace);
#endif
    free(c);
}
