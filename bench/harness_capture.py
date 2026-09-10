"""Private capture input and CPU tokenization. Reports must never serialize rows.

Only aggregate counters, pseudonymous task IDs and measurement provenance leave
this module. Full texts and tokenizer IDs live in memory only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

CATEGORIES = ('reasoning', 'content', 'tool_args')
FIELDS = ('reasoning_text', 'content_text', 'tool_args_text')


def ratio(a, b):
    return a / b if b else None


def load_capture(path):
    path = Path(path)
    data = path.read_bytes()
    rows = []
    for number, line in enumerate(data.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError()
        except (ValueError, TypeError):
            raise ValueError(f'invalid capture record {number}; retry a completed snapshot') from None
        rows.append(row)
    if not rows:
        raise ValueError('capture is empty')
    return rows, {'sha256': hashlib.sha256(data).hexdigest(), 'records': len(rows),
                  'bytes': len(data)}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def flatten_context(row):
    """Approximate chat order: tools, then chronological messages and tool calls.

    No real template delimiters, escaping, hidden reasoning or token IDs are
    reconstructed. Tool argument strings are kept literal, not JSON-escaped.
    """
    chunks = []
    if row.get('tools'):
        chunks.append('tools\n' + canonical(row['tools']))
    for message in row.get('messages', []):
        chunks.append('\n' + message.get('role', '') + '\n')
        content = message.get('content')
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            chunks.extend(part.get('text', '') for part in content if isinstance(part, dict))
        for call in message.get('tool_calls') or []:
            fn = call.get('function', {})
            chunks.append('\n' + fn.get('name', '') + '\n' + fn.get('arguments', ''))
    return ''.join(chunks)


def category_counts(row, tokenizer=None):
    texts = [row.get(field) or '' for field in FIELDS]
    if tokenizer is not None:
        return [len(tokenizer.encode(value)) for value in texts]
    # Character proportions reproduce call_attrib.py. Do not call chars tokens.
    return [len(value) for value in texts]


def call_type(row):
    """Same mutually exclusive buckets as experiments/call_attrib.py:30-41."""
    reasoning, content, _ = category_counts(row)
    if row.get('generation_tokens', 0) < 40 and not row.get('tool_names'):
        return 'side/title (heuristic)'
    if row.get('tool_names') and reasoning == 0 and content == 0:
        return 'tool only'
    if row.get('tool_names') and reasoning > 0:
        return 'reasoning+tool'
    if not row.get('tool_names') and reasoning > 0:
        return 'reasoning+prose'
    return 'prose only'


def identify_calls(rows, source_dir=None):
    """Use local runner time windows; otherwise infer task and conversation reset.

    Caller-supplied labels and user content never appear in the returned IDs.
    Artifacts are read only for timing/check status; no artifact text is copied.
    """
    artifacts = []
    if source_dir:
        for path in sorted(Path(source_dir).glob('run*/**/result.json')):
            data = json.loads(path.read_text())
            if all(isinstance(data.get(k), (float, int)) for k in ('start', 'end')):
                artifacts.append((path, data))
    tasks, sessions, result, previous = {}, {}, [], {}
    for row in rows:
        first_user = next((m.get('content') for m in row.get('messages', [])
                           if m.get('role') == 'user'), None)
        fingerprint = canonical(first_user) if first_user is not None else canonical(row.get('task', ''))
        task = tasks.setdefault(fingerprint, f'task-{len(tasks) + 1:03d}')
        matches = [(p, a) for p, a in artifacts if a['start'] <= row.get('t', -1) <= a['end'] + .1]
        artifact = matches[0] if len(matches) == 1 else None
        if artifact:
            key = str(artifact[0])  # internal only
            method = 'runner timestamp window'
        else:
            prior = previous.get(task)
            count = len(row.get('messages', []))
            reset = prior is None or count <= prior[1]
            key = f'{task}:inferred:{row.get("t", len(result))}' if reset else prior[0]
            method = 'first user fingerprint and message-count reset (inferred)'
            previous[task] = (key, count)
        session = sessions.setdefault(key, {'id': f'session-{len(sessions) + 1:03d}', 'calls': 0})
        session['calls'] += 1
        meta = {'task': task, 'session': session['id'], 'call_in_task': session['calls'],
                'identity_method': method}
        if artifact:
            a = artifact[1]
            # Publish only the numeric run suffix, not an arbitrary directory name.
            run = next((re.fullmatch(r'run(\d+)', p) for p in artifact[0].parts
                        if re.fullmatch(r'run(\d+)', p)), None)
            meta.update(run=int(run.group(1)) if run else None,
                        task_wall_s=a.get('seconds', a['end'] - a['start']),
                        official_pass=a.get('check', {}).get('passed'))
        result.append(meta)
    return result


class TokenizationUnavailable(RuntimeError):
    pass


class CpuTokenizer:
    """Explicit no-hold attestation, health before every small POST, memory cache.

    Chunk boundaries are approximate: independently encoded 2048-character
    pieces may differ from encoding one whole message. No generation API exists
    here. A failed health/tokenize request aborts the analysis, never substitutes
    character IDs for real tokens in a copy screen.
    """
    def __init__(self, endpoint, *, no_hold=False, model='glm-5.3', opener=None,
                 chunk_chars=2048):
        if not no_hold:
            raise TokenizationUnavailable('tokenization skipped: no-hold confirmation is required')
        parsed = urlparse(endpoint)
        if (parsed.scheme, parsed.hostname, parsed.port, parsed.path.rstrip('/')) != (
                'http', 'spark-06c4.local', 8000, '') or parsed.query or parsed.fragment or parsed.username:
            raise ValueError('only http://spark-06c4.local:8000 is allowed')
        if not 1 <= chunk_chars <= 2048:
            raise ValueError('chunk_chars must be 1..2048')
        self.endpoint, self.model = endpoint.rstrip('/'), model
        self.opener, self.chunk_chars = opener or urlopen, chunk_chars
        self.cache, self.requests = {}, 0
        self.health()

    def health(self):
        try:
            with self.opener(Request(self.endpoint + '/health'), timeout=5) as response:
                if response.status != 200:
                    raise TokenizationUnavailable('tokenization skipped: health check failed')
        except (URLError, OSError, TimeoutError):
            raise TokenizationUnavailable('tokenization skipped: health check failed') from None

    def encode(self, value):
        tokens = []
        for offset in range(0, len(value), self.chunk_chars):
            piece = value[offset:offset + self.chunk_chars]
            if piece not in self.cache:
                self.health()
                body = json.dumps({'model': self.model, 'prompt': piece,
                                   'add_special_tokens': False}).encode()
                try:
                    request = Request(self.endpoint + '/tokenize', data=body,
                                      headers={'Content-Type': 'application/json'}, method='POST')
                    with self.opener(request, timeout=10) as response:
                        payload = json.load(response)
                    ids = payload['tokens']
                    if not isinstance(ids, list) or any(type(x) is not int for x in ids):
                        raise ValueError()
                except (URLError, OSError, TimeoutError, ValueError, KeyError, TypeError):
                    raise TokenizationUnavailable('tokenization skipped: invalid or unavailable CPU tokenizer') from None
                self.cache[piece] = ids
                self.requests += 1
            tokens.extend(self.cache[piece])
        return tokens


def tokenizer_options(parser):
    parser.add_argument('--tokenize', action='store_true', help='Use the permitted CPU tokenizer only')
    parser.add_argument('--no-hold', action='store_true', help='Operator confirms no guarded hold for this run')
    parser.add_argument('--model', default='glm-5.3')


def make_tokenizer(args):
    if not args.tokenize:
        return None
    return CpuTokenizer('http://spark-06c4.local:8000', no_hold=args.no_hold, model=args.model)


def write_report(output, report, markdown):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    output.with_suffix('.md').write_text(markdown)
