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
        "DRAFTER_PER_TARGET", "slab_geometry", "SlabIO", "_slab_should_wipe", "_slab_persist_default",
        "_slab_persist_value", "_dir_fingerprint", "_content_identity"}
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

# --- review fixes: toggle coercion, read-epoch, epoch authority, widened gate, weight fingerprint ---
PV = ns["_slab_persist_value"]
check("persist coercion: 'false' -> OFF (bool('false') would be True)", PV("false") is False)
check("persist coercion: '0'/'off'/'' -> OFF", PV("0") is False and PV("off") is False and PV("") is False)
check("persist coercion: 'true'/'1'/True -> ON", PV("true") is True and PV("1") is True and PV(True) is True)
with tempfile.TemporaryDirectory() as d:
    sb=[_round_up(HDR+1000,4096), _round_up(HDR+400,4096)]; cnt=[8,8]
    kE=make_key(b"\xEE"*32,0); dat=b"E"*1000
    io=SlabIO(d,sb,cnt,epoch=5); io.write(kE,0,mv(dat),seq=1,epoch=5)
    b=mv(dat); io.read(kE,0,b,epoch=5); check("read with matching epoch serves", bytes(b[0])==dat)
    try: io.read(kE,0,mv(dat),epoch=6); check("read with a NEWER load epoch rejects the stale slot", False)
    except OSError as e: check("read with a NEWER load epoch rejects the stale slot", "epoch" in str(e))
    b=mv(dat); io.read(kE,0,b); check("read with no epoch given (legacy call) still serves", bytes(b[0])==dat)
    # a store sealed under an OLDER epoch must not be re-labelled by a concurrent newer epoch (header takes epoch explicitly)
    io.write(make_key(b"\xEF"*32,0),1,mv(dat),seq=2,epoch=3)
    hdrs={slot:ep for slot,ep in ((s_, io.parse_header(os.pread(io.fds[0],HDR,io._off(0,s_)))[1]) for s_ in (0,1))}
    check("each slot sealed with ITS OWN store epoch (5 and 3), self.epoch untouched", hdrs=={0:5,1:3} and io.epoch==5)
    # epoch authority: with the epoch file gone, the next epoch must be ABOVE the max on disk
    check("max_epoch() finds the highest header epoch on disk (5)", io.max_epoch()==5)
    io.close()
    e=SlabIO(os.path.join(d,"empty"),sb,cnt,epoch=0); check("max_epoch() is None on empty slabs", e.max_epoch() is None); e.close()
    # weight fingerprint: stat-based, changes when a checkpoint is swapped in place
    wd=os.path.join(d,"weights"); os.makedirs(wd); open(os.path.join(wd,"model.safetensors"),"wb").write(b"x"*100); open(os.path.join(wd,"config.json"),"w").write("{}")
    FP=ns["_dir_fingerprint"]; f1=FP(wd)
    open(os.path.join(wd,"model.safetensors"),"wb").write(b"y"*101)   # different size -> different weights
    f2=FP(wd); check("weight fingerprint changes when a checkpoint is swapped in place", f1 and f2 and f1!=f2)
    check("weight fingerprint is None for a non-directory (HF repo id)", FP("org/model-id") is None)
# widened gate: content identity change wipes even with identical run_config
W=ns["_slab_should_wipe"]; rc={"model":"glm"}; sb2,sc2,ev="sb","sc","v1"
ident={"weights_target":"abc","dtype":"bf16"}
good={"version":ns["SLAB_META_VERSION"],"slot_bytes":sb2,"slot_counts":sc2,"engine_version":ev,"run_config":rc,"boot_id":"A","content_identity":ident}
check("gate: same everything incl content_identity -> KEEP", W(good,sb2,sc2,rc,ev,"B",True,ident) is False)
check("gate: weights swapped in place (fingerprint differs) -> WIPE", W(good,sb2,sc2,rc,ev,"A",True,{**ident,"weights_target":"zzz"}) is True)
check("gate: dtype change -> WIPE", W(good,sb2,sc2,rc,ev,"A",True,{**ident,"dtype":"fp8"}) is True)
check("gate: old meta lacks content_identity (v2) -> WIPE (version bump handles it too)", W({**good,"version":2},sb2,sc2,rc,ev,"A",True,ident) is True)

print(f"\n=== {'ALL PASS' if fails==0 else str(fails)+' FAILURE(S)'}")
if __name__ == "__main__":
    sys.exit(1 if fails else 0)


def test_standalone_checks():
    assert fails == 0
