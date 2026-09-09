# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated MTP experiment overlay for the pinned GLM serving image.

The captured DeepSeekMTPModel returns (logits hidden, recycled hidden).
Its V2 caller must preserve both: compute_logits normalizes the first,
while the next draft step consumes the already normalized second tensor.
This file is not mounted by the ordinary launcher.
"""

import torch.nn as nn

from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model


class MTPSpeculator(AutoRegressiveSpeculator):
    @property
    def model_returns_tuple(self) -> bool:
        # This is a version-specific contract, pinned by the fixture manifest.
        # Other MTP architectures retain the captured caller's behavior.
        return self.draft_model_config.hf_config.architectures == [
            "DeepSeekMTPModel"
        ]

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        return load_eagle_model(target_model, self.vllm_config)
