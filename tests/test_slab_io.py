#!/usr/bin/env python3
"""Sandbox test of the real SlabIO from stage/glm-dcp/multinode.py: crash-safety + reboot
recovery, no GPU/vLLM/CUDA/torch. Extracts the pure slab core (constants, helpers, SlabIO)
from the actual source via ast and execs it with faithful OffloadKey/get_offload_group_idx
(36-byte key = 32-byte hash + 4-byte LE group). Fault injection per the Codex/GLM review."""
import os, sys, ast, struct, zlib, json, tempfile

class OffloadKey(bytes):
    def __new__(cls, b): assert len(b) == 36; return super().__new__(cls, b)
def make_key(hash32: bytes, group: int) -> OffloadKey:
    assert len(hash32) == 32
    return OffloadKey(hash32 + struct.pack("<I", group))
def get_offload_group_idx(key) -> int:
    return struct.unpack("<I", bytes(key)[32:36])[0]

SRC = open("stage/glm-dcp/multinode.py").read()
tree = ast.parse(SRC)
WANT = {"_round_up", "_atomic_write_json", "_drain_iov", "_crc", "SLAB_MAGIC", "SLAB_HEADER_BYTES",
        "SLAB_VERSION", "SLAB_META_VERSION", "_SLAB_HDR", "SLAB_META", "SLAB_EPOCH",
        "DRAFTER_PER_TARGET", "slab_geometry", "SlabIO", "_slab_should_wipe", "_slab_persist_default"}
segs = []
for node in tree.body:
    take = False
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in WANT: take = True
    elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in WANT for t in node.targets): take = True
    elif isinstance(node, ast.Assert): take = True   # the _SLAB_HDR size assert
    if take: segs.append(ast.get_source_segment(SRC, node))
ns = {"os": os, "struct": struct, "zlib": zlib, "json": json, "memoryview": memoryview,
      "OffloadKey": OffloadKey, "get_offload_group_idx": get_offload_group_idx}
exec("\n".join(segs), ns)
SlabIO = ns["SlabIO"]; _round_up = ns["_round_up"]; HDR = ns["SLAB_HEADER_BYTES"]
print("extracted SlabIO; SLAB_VERSION =", ns["SLAB_VERSION"], "META", ns["SLAB_META_VERSION"], "; segments:", len(segs))

fails = 0
def check(name, cond):
    global fails
    print(("PASS" if cond else "FAIL"), name); fails += 0 if cond else 1
def mv(b): return [memoryview(bytearray(b))]

with tempfile.TemporaryDirectory() as d:
    payload0 = 1000
    slot_bytes = [_round_up(HDR + payload0, 4096), _round_up(HDR + 400, 4096)]
    counts = [8, 8]
    io = SlabIO(d, slot_bytes, counts, epoch=1)
    k0 = make_key(b"\x11"*32, 0); k1 = make_key(b"\x22"*32, 0)
    data0 = b"A"*payload0; data1 = b"B"*payload0
    io.write(k0, 0, mv(data0), seq=1)
    io.write(k1, 1, mv(data1), seq=2)
    buf = mv(data0); io.read(k0, 0, buf); check("read round-trips payload", bytes(buf[0]) == data0)
    try: io.read(k1, 0, mv(data1)); check("wrong-key read rejected", False)
    except OSError: check("wrong-key read rejected", True)
    fd = io.fds[0]
    os.pwrite(fd, b"Z", io._off(0,0)+HDR+10)                       # torn payload
    try: io.read(k0, 0, mv(data0)); check("torn payload caught by CRC", False)
    except OSError as e: check("torn payload caught by CRC", "CRC" in str(e))
    os.pwrite(fd, b"\x99", io._off(0,1)+12)                        # torn header
    try: io.read(k1, 1, mv(data1)); check("torn header rejected", False)
    except OSError: check("torn header rejected", True)
    try: io.read(k0, 5, mv(data0)); check("blank slot rejected", False)   # never written
    except OSError: check("blank slot rejected", True)
    io.close()

    io2 = SlabIO(d, slot_bytes, counts, epoch=1)                   # REBOOT: fresh handle, same dir+epoch
    found = {slot: bytes(key) for slot, seq, key in io2.scan(0, epoch=1)}
    check("scan keeps valid-header slot 0", found.get(0) == bytes(k0))   # torn payload, header still valid -> listed (verified at read)
    check("scan drops torn-header slot 1", 1 not in found)
    io2.write(k1, 3, mv(data1), seq=9); io2.close()
    io3 = SlabIO(d, slot_bytes, counts, epoch=1)
    f2 = {slot: bytes(key) for slot, seq, key in io3.scan(0, epoch=1)}
    b = mv(data1); io3.read(k1, 3, b)
    check("reboot: new write survives scan+read", f2.get(3) == bytes(k1) and bytes(b[0]) == data1)
    check("epoch-2 scan sees no epoch-1 slots", len(list(io3.scan(0, epoch=2))) == 0)
    # atomic meta write round-trips
    mp = os.path.join(d, "m.json"); ns["_atomic_write_json"](mp, {"a": 1, "run_config": {"x": 2}})
    check("atomic meta write round-trips", json.load(open(mp)) == {"a": 1, "run_config": {"x": 2}})
    io3.close()

# --- wipe-decision toggle (default on) ---
W = ns["_slab_should_wipe"]
rc = {"model": "glm-5.3", "tp": 4}
good = {"version": ns["SLAB_META_VERSION"], "slot_bytes": [4096, 4096], "slot_counts": [8, 8],
        "engine_version": "v1", "run_config": rc, "boot_id": "AAA"}
sb, sc, ev = [4096, 4096], [8, 8], "v1"
check("gate: fresh (no meta) -> wipe", W(None, sb, sc, rc, ev, "BBB", True) is True)
check("gate: version mismatch -> wipe", W({**good, "version": 1}, sb, sc, rc, ev, "AAA", True) is True)
check("gate: run_config diff -> wipe", W({**good, "run_config": {"tp": 2}}, sb, sc, rc, ev, "AAA", True) is True)
check("gate: slot_counts diff -> wipe", W({**good, "slot_counts": [9, 8]}, sb, sc, rc, ev, "AAA", True) is True)
check("gate: engine_version diff -> wipe", W({**good, "engine_version": "v2"}, sb, sc, rc, ev, "AAA", True) is True)
check("gate: persist ON, new boot -> KEEP", W(good, sb, sc, rc, ev, "ZZZ", True) is False)
check("gate: persist OFF, new boot -> wipe", W(good, sb, sc, rc, ev, "ZZZ", False) is True)
check("gate: persist OFF, same boot -> KEEP", W(good, sb, sc, rc, ev, "AAA", False) is False)
import os as _os
_os.environ.pop("SLAB_PERSIST_ACROSS_REBOOT", None)
check("toggle default is ON", ns["_slab_persist_default"]() is True)
_os.environ["SLAB_PERSIST_ACROSS_REBOOT"] = "0"; check("toggle env=0 -> OFF", ns["_slab_persist_default"]() is False)
_os.environ["SLAB_PERSIST_ACROSS_REBOOT"] = "off"; check("toggle env=off -> OFF", ns["_slab_persist_default"]() is False)
_os.environ.pop("SLAB_PERSIST_ACROSS_REBOOT", None)

print(f"\n=== {'ALL PASS' if fails==0 else str(fails)+' FAILURE(S)'}")
sys.exit(1 if fails else 0)
