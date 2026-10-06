# SPDX-License-Identifier: Apache-2.0
"""glm_fast: step-5 decode speedups for GLM-5.3 on vLLM 0.29 (MRv2), each behind its own switch.

- vocab_argmax  VLLM_VOCAB_PARALLEL_ARGMAX=0|1|check   greedy target argmax without the
                                                       full-vocab logits gather (exact)
- l2_prefetch   VLLM_L2_PREFETCH=0|1                   L2 prefetch of the next dense weights
                                                       inside collective windows (exact)

Installed by glm_fast.pth -> glm_fast.boot (import-time monkeypatches; no vLLM file edited).
Parts are re-implemented from knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4 @770d115 (Apache-2.0 per
the SPDX headers of overlay/glm_l2_prefetch*.py and overlay/glm_target_argmax.py) and vLLM
(Apache-2.0); see each module's docstring.
"""
