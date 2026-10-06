// Behavioural test of the RoCE proxy's idle backoff (local patch 1) and slot bound
// (local patch 2), with no RDMA device: a context with no HCAs posts nothing, so the
// proxy loop, doorbell catch-up and idle policy run exactly as in production.
//
//   cc -O2 -std=gnu11 -Wall -Wextra -o /tmp/t test_proxy_idle.c -libverbs -lpthread && /tmp/t
#include "../b12x/b12x/comm/roce/_roce_proxy.c"

#include <assert.h>

static double thread_cpu_s(pthread_t t) {
    clockid_t id;
    struct timespec ts;
    pthread_getcpuclockid(t, &id);
    clock_gettime(id, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static double now_s(void) { return monotonic_ns() / 1e9; }

static void ring(roce_ctx_t *c, uint32_t seq, uint32_t nbytes) {
    volatile uint32_t *ctrl = (volatile uint32_t *)(c->region + c->ctrl_off);
    ctrl[4 + (seq & 1u)] = nbytes;
    __atomic_store_n(&ctrl[0], seq, __ATOMIC_RELEASE);
}

static double wait_posted(roce_ctx_t *c, uint32_t seq) {
    double t0 = now_s();
    while (__atomic_load_n(&c->last_seq, __ATOMIC_ACQUIRE) != seq && !atomic_load(&c->failed)) {
    }
    return now_s() - t0;
}

int main(void) {
    setenv("B12X_ROCE_IDLE_HOT_US", "50000", 1);
    setenv("B12X_ROCE_IDLE_MAX_NAP_US", "2000", 1);
    uint64_t layout[7];
    uint64_t slot = 4096;
    assert(roce_layout(2, slot, layout) == 0);
    void *region = calloc(1, layout[4]);
    char err[512];
    // n_hca must be >= 1 for roce_create, so build the context by hand with no HCA.
    roce_ctx_t *c = calloc(1, sizeof(*c));
    c->world = 2; c->rank = 0; c->n_hca = 0; c->region = region; c->region_bytes = layout[4];
    c->slot_bytes = slot; c->recv_off = layout[0]; c->flag_off = layout[1];
    c->send_off = layout[2]; c->ctrl_off = layout[3];
    c->idle_hot_ns = env_us_as_ns("B12X_ROCE_IDLE_HOT_US", ROCE_IDLE_HOT_NS_DEFAULT);
    c->idle_max_nap_ns = env_us_as_ns("B12X_ROCE_IDLE_MAX_NAP_US", ROCE_IDLE_MAX_NAP_NS_DEFAULT);
    (void)err;
    assert(c->idle_hot_ns == 50000000ull && c->idle_max_nap_ns == 2000000ull);
    assert(roce_start(c) == 0);

    // Hot: back-to-back doorbells are posted promptly.
    double worst_hot = 0;
    for (uint32_t s = 1; s <= 200; s++) {
        ring(c, s, 1024);
        double d = wait_posted(c, s);
        if (d > worst_hot) worst_hot = d;
    }
    // Idle: after the hot window the thread must stop spinning. Measured from 0.5 s
    // after the last doorbell over IDLE_SECONDS (default 2) to exclude the hot window.
    struct timespec half_s = {0, 500000000};
    nanosleep(&half_s, NULL);
    double cpu0 = thread_cpu_s(c->thread), t0 = now_s();
    const char *idle_env = getenv("IDLE_SECONDS");
    struct timespec idle_for = {idle_env ? atoi(idle_env) : 2, 0};
    struct timespec two_s = {2, 0};
    nanosleep(&idle_for, NULL);
    double idle_cpu = (thread_cpu_s(c->thread) - cpu0) / (now_s() - t0);
    // Wake-up latency of the first doorbell after an idle stretch: at most ~one nap.
    ring(c, 201, 1024);
    double wake = wait_posted(c, 201);
    // Two doorbells while the proxy naps: both posted in order (catch-up).
    nanosleep(&two_s, NULL);
    ring(c, 202, 512);
    ring(c, 203, 256);
    double catchup = wait_posted(c, 203);
    assert(!atomic_load(&c->failed));
    // Slot bound: a payload larger than a slot fails the proxy.
    ring(c, 204, (uint32_t)(slot * 2));
    double t1 = now_s();
    while (!atomic_load(&c->failed) && now_s() - t1 < 1.0) {
    }
    int bounded = atomic_load(&c->failed) && strstr(c->err, "exceeds") != NULL;
    printf("hot worst post latency %.1f us; steady idle CPU %.3f%% of a core; wake after idle %.0f us; "
           "catch-up of 2 doorbells %.0f us; slot bound %s (%s)\n",
           worst_hot * 1e6, idle_cpu * 100, wake * 1e6, catchup * 1e6, bounded ? "enforced" : "MISSING",
           c->err);
    roce_stop(c);
    int ok = worst_hot < 0.005 && idle_cpu < 0.05 && wake < 0.010 && catchup < 0.010 && bounded;
    printf(ok ? "proxy idle tests passed\n" : "proxy idle tests FAILED\n");
    free(c);
    free(region);
    return ok ? 0 : 1;
}
