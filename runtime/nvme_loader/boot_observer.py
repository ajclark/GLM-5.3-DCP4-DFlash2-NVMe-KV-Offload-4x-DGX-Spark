"""Controller-owned host samplers and a timed, validated streaming smoke request."""
import json
import re
import subprocess
import time
import urllib.request


class BootSamples:
    def __init__(self, hosts, output, generation):
        self.jobs = []
        try:
            for host in hosts:
                file = (output / f"{generation}-{host}-resources.jsonl").open("w")
                try:
                    proc = subprocess.Popen(["ssh", "-T", "-o", "BatchMode=yes", host,
                                             "python3 -u ~/spark-nvme-build/boot_samples.py"],
                                            stdin=subprocess.PIPE, stdout=file, stderr=subprocess.DEVNULL)
                except BaseException:
                    file.close()
                    raise
                self.jobs.append((proc, file))
        except BaseException:
            self.close()
            raise

    def close(self):
        for proc, _ in self.jobs:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
        for proc, file in self.jobs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            finally:
                file.close()
        self.jobs.clear()


def timed_smoke(base):
    body = {"model": "glm-5.3", "messages": [{"role": "user", "content":
            "Count from 1 to 100, one number per line. Output only the numbers."}],
            "max_tokens": 400, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": True, "stream_options": {"include_usage": True}}
    request = urllib.request.Request(base + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    started = time.monotonic()
    first = None
    text = ""
    usage = None
    complete = False
    with urllib.request.urlopen(request, timeout=300) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                complete = True
                break
            chunk = json.loads(data)
            if "error" in chunk:
                raise RuntimeError("streaming smoke request failed: " + str(chunk["error"]))
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                content = choice.get("delta", {}).get("content") or ""
                if content and first is None:
                    first = time.monotonic() - started
                text += content
    if not complete or first is None or [int(x) for x in re.findall(r"\d+", text)] != list(range(1, 101)):
        raise RuntimeError("streaming count100 regression")
    return {"content": text, "usage": usage, "first_content_seconds": first,
            "completion_seconds": time.monotonic() - started}
