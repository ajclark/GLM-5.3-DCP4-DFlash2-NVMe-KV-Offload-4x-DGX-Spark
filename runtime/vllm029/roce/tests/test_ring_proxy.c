// Behavioural test of the ring proxy (roce/glm_roce/_ring_proxy.c) with no RDMA device.
//
// Four ranks live in one process. Each rank has a real proxy thread running the
// production proxy_main, a fake NIC (one thread per queue pair, executing that QP's
// RDMA writes in order after random delays, so different QPs race each other), and a
// "kernel" thread that does what the b12x one-shot kernel does with the pinned region:
// stage the payload into send[s & 1], publish nbytes and ring the doorbell, wait for
// flag[p][s & 1] == s from every peer, then (after a random delay) check every peer's
// payload byte for byte. A slot overwritten too early, a lost forward, a wrong byte
// count or a deadlock fails the test.
//
// RING_SPLIT=1 runs the split mode (direction 0 clockwise, 1 counter-clockwise) and
// RING_CHUNKS=C cut-through with C chunks per direction; the kernel then waits for
// (split ? 2 : 1) * C flags per source.
//
//   cc -O2 -std=gnu11 -Wall -Wextra -o /tmp/t test_ring_proxy.c -libverbs -lpthread && /tmp/t
//   RING_SPLIT=1 /tmp/t
#include <infiniband/verbs.h>

static int fake_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad);
static int fake_poll_cq(struct ibv_cq *cq, int n, struct ibv_wc *wc);
#define ibv_post_send fake_post_send
#define ibv_poll_cq fake_poll_cq
#include "../glm_roce/_ring_proxy.c"

#include <assert.h>

#define OPS 20000
#define SLOT 65536
#define QDEPTH 4096

typedef struct {
    uint64_t dst, src, len, wr_id;
    int signaled;
    uint8_t inline_data[16];
    int is_inline;
} fake_wr_t;

typedef struct {
    pthread_mutex_t mu;
    fake_wr_t q[QDEPTH];
    unsigned head, tail;
    int cq;  // index of the completion queue this QP reports to
} fake_qp_t;

typedef struct {
    pthread_mutex_t mu;
    uint64_t ids[QDEPTH];
    unsigned head, tail;
} fake_cq_t;

static fake_qp_t qps[RING_WORLD * RING_HCAS];
static fake_cq_t cqs[RING_WORLD * RING_HCAS];
static struct ibv_qp qp_objs[RING_WORLD * RING_HCAS];
static struct ibv_cq cq_objs[RING_WORLD * RING_HCAS];
static struct ibv_mr mr_obj;
static atomic_int stop_nics;
static uint8_t *regions[RING_WORLD];
static ring_ctx_t *ctxs[RING_WORLD];
static uint64_t layout[7];
static atomic_int errors;
static int split_mode;
static int chunks = 1;

static unsigned rnd(unsigned *state) {
    *state = *state * 1103515245u + 12345u;
    return (*state >> 8) & 0xffffff;
}

static void spin_us(unsigned us) {
    uint64_t end = monotonic_ns() + (uint64_t)us * 1000ull;
    while (monotonic_ns() < end) {
    }
}

static int fake_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad) {
    (void)bad;
    fake_qp_t *f = &qps[qp - qp_objs];
    pthread_mutex_lock(&f->mu);
    for (; wr != NULL; wr = wr->next) {
        assert(f->tail - f->head < QDEPTH);
        fake_wr_t *w = &f->q[f->tail % QDEPTH];
        w->dst = wr->wr.rdma.remote_addr;
        w->len = wr->sg_list[0].length;
        w->wr_id = wr->wr_id;
        w->signaled = (wr->send_flags & IBV_SEND_SIGNALED) != 0;
        w->is_inline = (wr->send_flags & IBV_SEND_INLINE) != 0;
        if (w->is_inline) {
            assert(w->len <= sizeof(w->inline_data));
            memcpy(w->inline_data, (void *)(uintptr_t)wr->sg_list[0].addr, w->len);
        } else {
            w->src = wr->sg_list[0].addr;
        }
        f->tail++;
    }
    pthread_mutex_unlock(&f->mu);
    return 0;
}

static int fake_poll_cq(struct ibv_cq *cq, int n, struct ibv_wc *wc) {
    fake_cq_t *f = &cqs[cq - cq_objs];
    int got = 0;
    pthread_mutex_lock(&f->mu);
    while (got < n && f->head != f->tail) {
        memset(&wc[got], 0, sizeof(wc[got]));
        wc[got].wr_id = f->ids[f->head % QDEPTH];
        wc[got].status = IBV_WC_SUCCESS;
        f->head++;
        got++;
    }
    pthread_mutex_unlock(&f->mu);
    return got;
}

// One NIC per QP: execute writes in order; a write's bytes become visible before the
// next write of the same QP starts (RC ordering), with random gaps between writes.
static void *nic_main(void *arg) {
    int i = (int)(intptr_t)arg;
    fake_qp_t *f = &qps[i];
    unsigned seed = 77u + (unsigned)i;
    while (!atomic_load(&stop_nics)) {
        pthread_mutex_lock(&f->mu);
        if (f->head == f->tail) {
            pthread_mutex_unlock(&f->mu);
            continue;
        }
        fake_wr_t w = f->q[f->head % QDEPTH];
        pthread_mutex_unlock(&f->mu);
        unsigned roll = rnd(&seed);
        if (roll % 1500 == 0) {
            spin_us(2000 + roll % 2000);  // a stalled queue pair: the others run ahead
        } else if (roll % 4 == 0) {
            spin_us(roll % 40);
        }
        // Payload in two halves with a gap: a reader that looks before the flag
        // would see a half-written slot.
        const uint8_t *src = w.is_inline ? w.inline_data : (const uint8_t *)(uintptr_t)w.src;
        uint64_t half = (w.len / 2) & ~15ull;
        memcpy((void *)(uintptr_t)w.dst, src, half);
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
        memcpy((void *)(uintptr_t)(w.dst + half), src + half, w.len - half);
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
        pthread_mutex_lock(&f->mu);
        f->head++;
        pthread_mutex_unlock(&f->mu);
        if (w.signaled) {
            fake_cq_t *c = &cqs[f->cq];
            pthread_mutex_lock(&c->mu);
            c->ids[c->tail % QDEPTH] = w.wr_id;
            c->tail++;
            pthread_mutex_unlock(&c->mu);
        }
    }
    return NULL;
}

static uint32_t op_bytes(uint32_t seq) { return 16u * (1u + (seq * 2654435761u) % (SLOT / 16)); }

static uint32_t word(int src, uint32_t seq, uint32_t idx) {
    return ((uint32_t)src << 28) ^ (seq * 7919u) ^ (idx * 2246822519u);
}

static void *kernel_main(void *arg) {
    int r = (int)(intptr_t)arg;
    uint8_t *region = regions[r];
    volatile uint32_t *ctrl = (volatile uint32_t *)(region + layout[3]);
    unsigned seed = 1000u + (unsigned)r;
    for (uint32_t s = 1; s <= OPS; s++) {
        uint32_t slot = s & 1u, nbytes = op_bytes(s);
        uint32_t *send = (uint32_t *)(region + layout[2] + (uint64_t)slot * SLOT);
        for (uint32_t i = 0; i < nbytes / 4; i++) {
            send[i] = word(r, s, i);
        }
        ctrl[4 + slot] = nbytes;
        __atomic_store_n(&ctrl[0], s, __ATOMIC_RELEASE);
        uint64_t n_flag = (uint64_t)((split_mode ? 2 : 1) * chunks);
        for (int p = 0; p < RING_WORLD; p++) {
            for (uint64_t h = 0; h < n_flag && p != r; h++) {
                volatile uint32_t *flag = (volatile uint32_t *)(region + layout[1] +
                    (((uint64_t)p * ROCE_SLOTS + slot) * n_flag + h) * ROCE_FLAG_STRIDE);
                uint64_t t0 = monotonic_ns();
                while (__atomic_load_n(flag, __ATOMIC_ACQUIRE) != s) {
                    if (monotonic_ns() - t0 > 5000000000ull) {
                        fprintf(stderr, "rank %d: timeout waiting for peer %d stripe %d op %u (proxy err: %s)\n",
                                r, p, (int)h, s, ctxs[r]->err);
                        atomic_fetch_add(&errors, 1);
                        return NULL;
                    }
                }
            }
        }
        if (rnd(&seed) % 8 == 0) {
            spin_us(rnd(&seed) % 60);  // a slow kernel keeps reading its slots
        }
        for (int p = 0; p < RING_WORLD; p++) {
            if (p == r) {
                continue;
            }
            const uint32_t *recv = (const uint32_t *)(region + layout[0] + ((uint64_t)p * ROCE_SLOTS + slot) * SLOT);
            for (uint32_t i = 0; i < nbytes / 4; i++) {
                if (recv[i] != word(p, s, i)) {
                    fprintf(stderr, "rank %d: op %u peer %d word %u is %08x, want %08x\n", r, s, p, i, recv[i],
                            word(p, s, i));
                    atomic_fetch_add(&errors, 1);
                    return NULL;
                }
            }
        }
    }
    return NULL;
}

int main(void) {
    setenv("B12X_ROCE_IDLE_HOT_US", "200", 1);  // short hot window: naps happen mid-run
    setenv("B12X_ROCE_IDLE_MAX_NAP_US", "300", 1);
    split_mode = getenv("RING_SPLIT") != NULL && atoi(getenv("RING_SPLIT")) == 1;
    chunks = getenv("RING_CHUNKS") != NULL ? atoi(getenv("RING_CHUNKS")) : 1;
    assert(ring_layout(RING_WORLD, SLOT, layout) == 0);
    uint64_t ref[7];
    for (int r = 0; r < RING_WORLD; r++) {
        regions[r] = calloc(1, layout[4]);
        ring_ctx_t *c = calloc(1, sizeof(*c));
        c->rank = r;
        c->split = split_mode;
        c->nchunks = chunks;
        c->nflag = (split_mode ? 2 : 1) * chunks;
#ifdef RING_TRACE
        c->trace = calloc(RING_TRACE_N, sizeof(ring_trace_t));
#endif
        c->region = regions[r];
        c->region_bytes = layout[4];
        c->slot_bytes = SLOT;
        c->recv_off = layout[0];
        c->flag_off = layout[1];
        c->send_off = layout[2];
        c->ctrl_off = layout[3];
        c->idle_hot_ns = env_us_as_ns("B12X_ROCE_IDLE_HOT_US", ROCE_IDLE_HOT_NS_DEFAULT);
        c->idle_max_nap_ns = env_us_as_ns("B12X_ROCE_IDLE_MAX_NAP_US", ROCE_IDLE_MAX_NAP_NS_DEFAULT);
        ctxs[r] = c;
    }
    (void)ref;
    for (int r = 0; r < RING_WORLD; r++) {
        for (int h = 0; h < RING_HCAS; h++) {
            int i = r * RING_HCAS + h;
            pthread_mutex_init(&qps[i].mu, NULL);
            pthread_mutex_init(&cqs[i].mu, NULL);
            qps[i].cq = i;
            int peer = h == 0 ? (r + 1) % RING_WORLD : (r + RING_WORLD - 1) % RING_WORLD;
            ctxs[r]->hca[h].qp = &qp_objs[i];
            ctxs[r]->hca[h].cq = &cq_objs[i];
            ctxs[r]->hca[h].mr = &mr_obj;
            ctxs[r]->hca[h].peer_addr = (uint64_t)(uintptr_t)regions[peer];
        }
    }
    pthread_t nics[RING_WORLD * RING_HCAS], kernels[RING_WORLD];
    for (int i = 0; i < RING_WORLD * RING_HCAS; i++) {
        pthread_create(&nics[i], NULL, nic_main, (void *)(intptr_t)i);
    }
    for (int r = 0; r < RING_WORLD; r++) {
        assert(ring_start(ctxs[r]) == 0);
    }
    uint64_t t0 = monotonic_ns();
    for (int r = 0; r < RING_WORLD; r++) {
        pthread_create(&kernels[r], NULL, kernel_main, (void *)(intptr_t)r);
    }
    for (int r = 0; r < RING_WORLD; r++) {
        pthread_join(kernels[r], NULL);
    }
    double secs = (monotonic_ns() - t0) / 1e9;
    // Let the last forwards' completions drain before reading the counters.
    struct timespec ms = {0, 50000000};
    nanosleep(&ms, NULL);
    int ok = atomic_load(&errors) == 0;
    for (int r = 0; r < RING_WORLD; r++) {
        ring_ctx_t *c = ctxs[r];
        printf("rank %d: own %llu fwd %llu last_seq %u fwd_done %u/%u failed %d %s\n", r,
               (unsigned long long)c->ops_posted, (unsigned long long)c->fwd_count, c->last_seq, c->fwd_done[0],
               c->fwd_done[1], atomic_load(&c->failed), c->err);
        ok = ok && c->ops_posted == OPS && c->fwd_count == (uint64_t)OPS * (split_mode ? 2 : 1) * chunks &&
             c->fwd_done[0] == OPS && (!split_mode || c->fwd_done[1] == OPS) && !atomic_load(&c->failed);
    }
    // Slot bound: an own payload bigger than a slot fails the proxy.
    volatile uint32_t *ctrl0 = (volatile uint32_t *)(regions[0] + layout[3]);
    ctrl0[4 + ((OPS + 1) & 1u)] = 2 * SLOT;
    __atomic_store_n(&ctrl0[0], OPS + 1, __ATOMIC_RELEASE);
    uint64_t t1 = monotonic_ns();
    while (!atomic_load(&ctxs[0]->failed) && monotonic_ns() - t1 < 1000000000ull) {
    }
    int bound_ok = atomic_load(&ctxs[0]->failed) && strstr(ctxs[0]->err, "within the") != NULL;
    printf("slot bound: %s (%s)\n", bound_ok ? "ok" : "MISSING", ctxs[0]->err);
    for (int r = 0; r < RING_WORLD; r++) {
        ring_stop(ctxs[r]);
    }
    atomic_store(&stop_nics, 1);
    for (int i = 0; i < RING_WORLD * RING_HCAS; i++) {
        pthread_join(nics[i], NULL);
    }
    printf("%s mode, %d chunk(s): %d ops x 4 ranks in %.2f s (%.1f us/op), errors %d\n", split_mode ? "split" : "single", chunks, OPS, secs, secs * 1e6 / OPS,
           atomic_load(&errors));
    ok = ok && bound_ok;
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
