# Production runtime + draft-aware verification cap for single requests and small batches
# (off unless VLLM_VERIFY_CAP=1), the voice-first scheduler class with optional prefill
# cadence (used only when passed as --scheduler-cls), the prefix-hit fix
# (VLLM_PREFIX_HIT_FIX=1), the padded-vocab argmax clamp (always on; bit-exact) and the
# drafter diet (an int8 draft checkpoint loads as-is; VLLM_DFLASH_FC_SPLIT=1 shards fc).
# Not in manifest.json, so the NVMe slab salt (and the warm slab) is unchanged.
ARG BASE=spark-vllm:0.29.0-nvme4-stridefix-kvtier-20260928
FROM ${BASE}
ARG VLLM=/usr/local/lib/python3.12/dist-packages/vllm
COPY vllm/v1/worker/gpu/model_runner.py ${VLLM}/v1/worker/gpu/model_runner.py
COPY vllm/v1/worker/gpu/cudagraph_utils.py ${VLLM}/v1/worker/gpu/cudagraph_utils.py
COPY vllm/v1/worker/gpu/spec_decode/verify_cap.py ${VLLM}/v1/worker/gpu/spec_decode/verify_cap.py
COPY vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py ${VLLM}/v1/worker/gpu/spec_decode/rejection_sampler_utils.py
COPY vllm/v1/worker/gpu/sample/gumbel.py ${VLLM}/v1/worker/gpu/sample/gumbel.py
COPY vllm/v1/core/sched/voice_first.py ${VLLM}/v1/core/sched/voice_first.py
COPY vllm/v1/core/kv_cache_coordinator.py ${VLLM}/v1/core/kv_cache_coordinator.py
COPY vllm/v1/core/single_type_kv_cache_manager.py ${VLLM}/v1/core/single_type_kv_cache_manager.py
COPY vllm/model_executor/models/qwen3_dflash.py ${VLLM}/model_executor/models/qwen3_dflash.py
COPY vllm/model_executor/models/qwen3_dflash2.py ${VLLM}/model_executor/models/qwen3_dflash2.py
# Item 2 E1: the base overlay's SM120 sparse-MLA backend, with its redundant full-output masked_fill_ skipped
# when VLLM_DCP_GLUE=1 (byte-identical to the base file otherwise).
COPY vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py ${VLLM}/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py
COPY cal.json costs.json /opt/verify-cap/
RUN cd ${VLLM} && python3 -m py_compile v1/worker/gpu/model_runner.py v1/worker/gpu/cudagraph_utils.py \
    v1/worker/gpu/spec_decode/verify_cap.py v1/worker/gpu/spec_decode/rejection_sampler_utils.py \
    v1/worker/gpu/sample/gumbel.py v1/core/sched/voice_first.py v1/core/kv_cache_coordinator.py \
    v1/core/single_type_kv_cache_manager.py model_executor/models/qwen3_dflash.py \
    model_executor/models/qwen3_dflash2.py v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py

# RoCEnante one-shot collectives for the DCP pairs (opt-in at runtime: GLM_ROCE_ALLREDUCE=1
# GLM_ROCE_GROUPS=dcp; launcher switch ROCE_DCP=1), and the TP ring transport built on it
# (glm_roce/ring.py + _ring_proxy.c; launcher switch ROCE_TP=1). Vendored b12x is byte-identical to
# local-inference-lab/b12x b58f34ea except the two reviewed proxy patches; the shim is
# knapcio's (MIT), adapted. Review: docs/ROCENANTE-REVIEW.md. The RDMA proxy is compiled
# here, hardened, into a root-owned directory, so containers never compile or load
# anything else at runtime.
COPY roce/b12x/b12x /opt/glm-roce-b12x/b12x
COPY roce/b12x/LICENSE /opt/glm-roce-b12x/LICENSE
COPY roce/glm_roce /opt/glm-roce/glm_roce
COPY roce/glm_roce.pth roce/test_dcp_pair.py roce/test_tp_ring.py /opt/glm-roce/
RUN set -eu; \
    python3 -c "import importlib.util as u, sys; sys.exit('base image already has b12x') if u.find_spec('b12x') else None"; \
    site="$(python3 -c 'import importlib.util as u, os; print(os.path.dirname(os.path.dirname(u.find_spec("vllm").origin)))')"; \
    cp /opt/glm-roce/glm_roce.pth "$site/glm_roce.pth"; \
    src=/opt/glm-roce-b12x/b12x/comm/roce/_roce_proxy.c; digest="$(sha256sum "$src" | cut -c1-16)"; \
    mkdir -p /opt/glm-roce-b12x/proxy-cache; \
    gcc -O2 -std=gnu11 -Wall -Wextra -Werror -fstack-protector-strong -D_FORTIFY_SOURCE=2 -shared -fPIC \
        -o "/opt/glm-roce-b12x/proxy-cache/roce_proxy-$digest.so" "$src" -libverbs -lpthread; \
    src=/opt/glm-roce/glm_roce/_ring_proxy.c; digest="$(sha256sum "$src" | cut -c1-16)"; \
    gcc -O2 -std=gnu11 -Wall -Wextra -Werror -fstack-protector-strong -D_FORTIFY_SOURCE=2 -shared -fPIC \
        -o "/opt/glm-roce-b12x/proxy-cache/ring_proxy-$digest.so" "$src" -libverbs -lpthread; \
    src=/opt/glm-roce/glm_roce/_pipe_proxy.c; digest="$(sha256sum "$src" | cut -c1-16)"; \
    gcc -O2 -std=gnu11 -Wall -Wextra -Werror -fstack-protector-strong -D_FORTIFY_SOURCE=2 -shared -fPIC \
        -o "/opt/glm-roce-b12x/proxy-cache/pipe_proxy-$digest.so" "$src" -libverbs -lpthread; \
    chmod -R a+rX,go-w /opt/glm-roce /opt/glm-roce-b12x; \
    sha256sum /opt/glm-roce-b12x/proxy-cache/*.so
ENV B12X_ROCE_CACHE_DIR=/opt/glm-roce-b12x/proxy-cache
RUN CUDA_VISIBLE_DEVICES= GLM_ROCE_ALLREDUCE=0 PYTHONPATH=/opt/glm-roce:/opt/glm-roce-b12x python3 -c \
    "from glm_roce.install import verify_targets; p = verify_targets(); print('verify_targets', p or 'ok'); raise SystemExit(1 if p else 0)"

# glm_fast (step 5, results/step5-l2-argmax-20261001): vocab-parallel target argmax + L2 prefetch in
# collective windows; inert unless VLLM_VOCAB_PARALLEL_ARGMAX / VLLM_L2_PREFETCH are set. Also the DCP
# combine glue (item 2 E1, VLLM_DCP_GLUE). Re-implements
# knapcio @770d115 glm_l2_prefetch*.py / glm_target_argmax.py (Apache-2.0).
COPY glm_fast /opt/glm-fast/glm_fast
RUN set -eu; \
    site="$(python3 -c 'import importlib.util as u, os; print(os.path.dirname(os.path.dirname(u.find_spec("vllm").origin)))')"; \
    cp /opt/glm-fast/glm_fast/glm_fast.pth "$site/glm_fast.pth"; \
    nvcc -O3 -arch=sm_121a -shared -Xcompiler -fPIC -o /opt/glm-fast/libglm_l2pf.so /opt/glm-fast/glm_fast/l2pf.cu; \
    python3 -m py_compile /opt/glm-fast/glm_fast/*.py; chmod -R a+rX,go-w /opt/glm-fast; sha256sum /opt/glm-fast/libglm_l2pf.so
RUN CUDA_VISIBLE_DEVICES= PYTHONPATH=/opt/glm-fast python3 -m glm_fast.install
