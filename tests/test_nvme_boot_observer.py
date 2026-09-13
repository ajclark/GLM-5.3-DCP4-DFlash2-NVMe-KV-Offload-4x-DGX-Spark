import io
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime/nvme_loader"))
import boot_observer


@pytest.mark.parametrize("done", [True, False])
def test_timed_smoke_requires_correct_completed_stream(monkeypatch, done):
    text = "\n".join(str(i) for i in range(1, 101))
    chunk = {"choices": [{"delta": {"content": text}}]}
    wire = b"data: " + json.dumps(chunk).encode() + b"\n\n"
    if done:
        wire += b"data: [DONE]\n\n"
    monkeypatch.setattr(boot_observer.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(wire))
    if done:
        result = boot_observer.timed_smoke("http://unused")
        assert result["content"] == text
        assert 0 <= result["first_content_seconds"] <= result["completion_seconds"]
    else:
        with pytest.raises(RuntimeError, match="regression"):
            boot_observer.timed_smoke("http://unused")


def test_stream_error_is_not_readiness(monkeypatch):
    monkeypatch.setattr(boot_observer.urllib.request, "urlopen", lambda *a, **kw:
                        io.BytesIO(b'data: {"error": "engine failed"}\n\n'))
    with pytest.raises(RuntimeError, match="engine failed"):
        boot_observer.timed_smoke("http://unused")
