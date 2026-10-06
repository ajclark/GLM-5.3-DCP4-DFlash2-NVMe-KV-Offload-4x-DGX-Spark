# SPDX-License-Identifier: Apache-2.0
"""Env-gated start-up hook, imported from ``glm_fast.pth`` in every Python process.

With VLLM_VOCAB_PARALLEL_ARGMAX, VLLM_L2_PREFETCH and VLLM_DCP_GLUE all unset/0 this module does nothing
else: no import hook, the image behaves exactly like its base.

Otherwise it installs a post-import hook that patches, right after it executes and in
whichever process imports it (API server, engine core, the spawned TP workers):
- vllm.v1.worker.gpu.model_runner       -> glm_fast.vocab_argmax.install (GPUModelRunner.sample)
- vllm.models.deepseek_v32.nvidia.model -> glm_fast.l2_prefetch.install
- vllm.v1.attention.ops.dcp             -> glm_fast.dcp_glue.install (VLLM_DCP_GLUE)
No vLLM file is edited (same pattern as glm_roce.boot). A patch that fails raises: the operator
asked for it, so a half-patched engine must not come up; `install.verify_targets` in the image
build catches upstream drift earlier.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import os
import sys
import threading

_OFF = ("", "0", "off", "false", "no")

MOD_RUNNER = "vllm.v1.worker.gpu.model_runner"
MOD_MODEL = "vllm.models.deepseek_v32.nvidia.model"
MOD_DCP = "vllm.v1.attention.ops.dcp"


def _on(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() not in _OFF


def targets() -> dict:
    """module name -> [(glm_fast module, patch function), ...] for the switches that are on."""
    t: dict = {}
    if _on("VLLM_VOCAB_PARALLEL_ARGMAX"):
        t.setdefault(MOD_RUNNER, []).append(("glm_fast.vocab_argmax", "install"))
    if _on("VLLM_L2_PREFETCH"):
        t.setdefault(MOD_MODEL, []).append(("glm_fast.l2_prefetch", "install"))
        if os.environ.get("VLLM_L2_PREFETCH_CONTROL", "").strip():
            t.setdefault(MOD_RUNNER, []).append(("glm_fast.l2_prefetch", "install_runner"))
    if _on("VLLM_DCP_GLUE"):
        t.setdefault(MOD_DCP, []).append(("glm_fast.dcp_glue", "install"))
    return t


def _apply(entries, module) -> None:
    if isinstance(entries, tuple):
        entries = [entries]
    for mod, fn in entries:
        getattr(importlib.import_module(mod), fn)(module)


class PostImportPatcher(importlib.abc.MetaPathFinder):
    """Run ``patch(module)`` right after each target module executes (once per name)."""

    def __init__(self, targets):
        self._targets = dict(targets)
        self._busy = threading.local()

    def pending(self):
        return sorted(self._targets)

    def find_spec(self, name, path=None, target=None):
        if name not in self._targets:
            return None
        busy = getattr(self._busy, "names", None)
        if busy is None:
            busy = self._busy.names = set()
        if name in busy:
            return None
        busy.add(name)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            busy.discard(name)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return None
        entry = self._targets.pop(name)
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            try:
                _apply(entry, module)
            except Exception as exc:
                raise RuntimeError(f"glm_fast: patching {name} failed: {exc!r}") from exc

        spec.loader.exec_module = exec_and_patch
        return spec

    def patch_already_imported(self):
        for name in list(self._targets):
            module = sys.modules.get(name)
            if module is not None:
                _apply(self._targets.pop(name), module)


def main():
    t = targets()
    if not t:
        return None
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if here not in sys.path:
        sys.path.insert(0, here)
    finder = PostImportPatcher(t)
    sys.meta_path.insert(0, finder)
    finder.patch_already_imported()
    return finder


FINDER = main()
