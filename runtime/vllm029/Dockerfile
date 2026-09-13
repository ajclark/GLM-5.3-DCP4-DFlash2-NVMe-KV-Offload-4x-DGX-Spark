FROM vllm/vllm-openai@sha256:18372a7224938643461b846fb64c5c9d3d6e9727e82caf2dc3043e620c9d4d7a
COPY manifest.json /opt/spark-vllm-manifest.json
COPY verify_image.py /opt/verify-spark-vllm.py
RUN python3 /opt/verify-spark-vllm.py --base
COPY overlay/vllm/ /usr/local/lib/python3.12/dist-packages/vllm/
COPY overlay/flashinfer/ /usr/local/lib/python3.12/dist-packages/flashinfer/
RUN python3 /opt/verify-spark-vllm.py
# A distinct module name bypasses the upstream wheel's unpatched JIT binary.
# Build for GB10 without exposing a GPU or loading model weights.
RUN FLASHINFER_CUDA_ARCH_LIST=12.1a MAX_JOBS=1 python3 -c "from flashinfer.jit.mla import gen_sparse_mla_sm120_module; gen_sparse_mla_sm120_module().build()"
ENV VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 DO_NOT_TRACK=1
