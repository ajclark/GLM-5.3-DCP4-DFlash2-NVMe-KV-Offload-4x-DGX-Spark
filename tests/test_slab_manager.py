#!/usr/bin/env python3
"""Manager-level test (the reviewer's #1 missing test): SlabOffloadingManager._attach rebuild
from a rank-0 slab with duplicate keys, seq gaps and a torn slot; epoch-file resurrection; the
reset_cache durable epoch bump. Extracts the real class via ast with a stubbed base."""
import os, sys, ast, struct, zlib, json, tempfile, collections, types

class OffloadKey(bytes):
    def __new__(cls,b): assert len(b)==36; return super().__new__(cls,b)
def make_key(h,g): return OffloadKey(((h*32)[:32])+struct.pack("<I",g))
def get_offload_group_idx(key): return struct.unpack("<I",bytes(key)[32:36])[0]
SRC=open("stage/glm-dcp/multinode.py").read()
WANT={"_round_up","_crc","SLAB_MAGIC","SLAB_HEADER_BYTES","SLAB_VERSION","SLAB_META_VERSION","_SLAB_HDR",
      "SLAB_META","SLAB_EPOCH","DRAFTER_PER_TARGET","slab_geometry","SlabIO","_drain_iov","_atomic_write_json",
      "_atomic_write_text","pathlib_read","SlabOffloadingManager"}
segs=[ast.get_source_segment(SRC,n) for n in ast.parse(SRC).body
      if (isinstance(n,(ast.FunctionDef,ast.ClassDef)) and getattr(n,"name","") in WANT)
      or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in WANT for t in n.targets))
      or isinstance(n,ast.Assert)]
class OffloadingManager: pass            # stub base (vllm)
class _Log:
    def __getattr__(self,n): return lambda *a,**k: None
RANK0={"path":None}
def discover_rank0_base_path(root, model): return RANK0["path"]
ns={"os":os,"struct":struct,"zlib":zlib,"json":json,"collections":collections,"memoryview":memoryview,
    "OffloadKey":OffloadKey,"get_offload_group_idx":get_offload_group_idx,"OffloadingManager":OffloadingManager,
    "logger":_Log(),"discover_rank0_base_path":discover_rank0_base_path,"Any":object,"Iterable":object}
try:
    exec("\n".join(segs),ns)
except Exception as e:
    print("EXTRACT FAILED:",repr(e)); raise
M=ns["SlabOffloadingManager"]; SlabIO=ns["SlabIO"]; HDR=ns["SLAB_HEADER_BYTES"]; _round_up=ns["_round_up"]
fails=0
def check(name,c):
    global fails; print(("PASS" if c else "FAIL"),name); fails+=0 if c else 1
def mv(b): return [memoryview(bytearray(b))]

with tempfile.TemporaryDirectory() as root:
    base=os.path.join(root,"_models_x_deadbeef"); r0=base+"_r0"; os.makedirs(r0); RANK0["path"]=base
    sb=[_round_up(HDR+1000,4096),_round_up(HDR+400,4096)]; cnt=[10,10]
    json.dump({"version":ns["SLAB_META_VERSION"],"slot_counts":cnt,"slot_bytes":sb},open(os.path.join(r0,"slab-meta.json"),"w"))
    kA=make_key(b"\xAA",0); kB=make_key(b"\xBB",0); kC=make_key(b"\xCC",0); d=b"Z"*1000
    io=SlabIO(r0,sb,cnt,epoch=2)
    io.write(kA,0,mv(d),seq=10,epoch=2)          # A old
    io.write(kA,3,mv(d),seq=50,epoch=2)          # A newer (duplicate key, higher seq) -> must win
    io.write(kB,1,mv(d),seq=20,epoch=2)
    io.write(kC,4,mv(d),seq=7,epoch=1)           # stale epoch -> must NOT be indexed
    io.write(make_key(b"\xDD",0),5,mv(d),seq=99,epoch=2)
    os.pwrite(io.fds[0],b"\x99",io._off(0,5)+12) # torn header on slot 5 -> must be free, not indexed
    io.close()
    ns["_atomic_write_text"](os.path.join(r0,"slab-epoch"),"2")
    m=M(root,"x"); ok=m._attach()
    idx=m.index[0]; free=set(m.free[0]); indexed=set(idx.values())
    check("attach succeeds and rebuilds from headers", ok and len(idx)>0)
    check("duplicate key: newest seq wins (kA -> slot 3, not 0)", idx.get(kA)==3)
    check("stale-epoch slot (kC, epoch 1) NOT indexed", kC not in idx)
    check("torn-header slot 5 NOT indexed", 5 not in indexed)
    check("index and free list are DISJOINT", not (indexed & free))
    check("index U free == all slots (nothing lost)", (indexed|free)==set(range(cnt[0])))
    check("losing duplicate slot 0 returned to free", 0 in free)
    check("seq resumes ABOVE the max recovered (>=50)", m.seq>=50)
    # reset_cache: durable epoch bump, index cleared, everything free
    m.reset_cache()
    ep_on_disk=int(open(os.path.join(r0,"slab-epoch")).read())
    check("reset_cache bumps the epoch DURABLY on disk (2 -> 3) before use", ep_on_disk==3 and m.epoch==3)
    check("reset_cache clears the index and frees every slot", len(m.index[0])==0 and set(m.free[0])==set(range(cnt[0])))
    # epoch-file resurrection: delete the epoch file, re-attach -> must NOT resurrect epoch-2/3 slots at epoch 0
    os.remove(os.path.join(r0,"slab-epoch"))
    m2=M(root,"x"); m2._attach()
    check("missing epoch file -> epoch chosen ABOVE max on disk (>=3), never 0", m2.epoch>=3)
    check("missing epoch file -> no old-epoch slot resurrected into the index", len(m2.index[0])==0)
    check("missing epoch file -> re-published durably", os.path.exists(os.path.join(r0,"slab-epoch")) and int(open(os.path.join(r0,"slab-epoch")).read())==m2.epoch)
    # corrupt epoch file -> same rule, no crash
    open(os.path.join(r0,"slab-epoch"),"w").write("garbage")
    m3=M(root,"x"); m3._attach()
    check("corrupt epoch file -> no crash, epoch above max on disk, nothing resurrected", m3.epoch>=3 and len(m3.index[0])==0)

print(f"\n=== {'ALL PASS' if fails==0 else str(fails)+' FAILURE(S)'}")
if __name__ == "__main__":
    sys.exit(1 if fails else 0)


def test_standalone_checks():
    assert fails == 0
