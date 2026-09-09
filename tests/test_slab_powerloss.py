#!/usr/bin/env python3
"""EXHAUSTIVE power-loss model for one reused slot (Codex methodology critique: an abort harness
keeps dirty page-cache data; a real power loss drops any write not made durable, in any order).
The slab write path issues, with NO fsync between them: (1) blank header, (2) payload, (3) seal
header. Before the per-chunk fdatasync, a power loss can leave the HEADER in {old, blank, new,
torn-new} and the PAYLOAD in {old, new, torn-new}, independently. We enumerate ALL 4x3 = 12
on-disk images for reusing slot 0 (valid A -> B) and assert the invariant on each:
    read() never returns bytes that are neither a completely-stored A nor a completely-stored B,
    and any slot scan() yields must read as exactly the block its header names (or raise)."""
import os, sys, ast, struct, zlib, tempfile, itertools

class OffloadKey(bytes):
    def __new__(cls, b): assert len(b)==36; return super().__new__(cls,b)
def make_key(h,g): return OffloadKey(((h*32)[:32])+struct.pack("<I",g))
def get_offload_group_idx(key): return struct.unpack("<I",bytes(key)[32:36])[0]
SRC=open("stage/glm-dcp/multinode.py").read()
WANT={"_round_up","_crc","SLAB_MAGIC","SLAB_HEADER_BYTES","SLAB_VERSION","SLAB_META_VERSION",
      "_SLAB_HDR","SLAB_META","SLAB_EPOCH","DRAFTER_PER_TARGET","slab_geometry","SlabIO","_drain_iov"}
segs=[ast.get_source_segment(SRC,n) for n in ast.parse(SRC).body
      if (isinstance(n,(ast.FunctionDef,ast.ClassDef)) and getattr(n,"name","") in WANT)
      or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in WANT for t in n.targets))
      or isinstance(n,ast.Assert)]
ns={"os":os,"struct":struct,"zlib":zlib,"memoryview":memoryview,"OffloadKey":OffloadKey,"get_offload_group_idx":get_offload_group_idx}
exec("\n".join(segs),ns)
SlabIO,HDR,_round_up,_crc=ns["SlabIO"],ns["SLAB_HEADER_BYTES"],ns["_round_up"],ns["_crc"]
fails=0
def check(name,cond):
    global fails; print(("PASS" if cond else "FAIL"),name); fails+=0 if cond else 1
def mv(b): return [memoryview(bytearray(b))]

sb=[_round_up(HDR+1000,4096),_round_up(HDR+400,4096)]; counts=[4,4]
kA=make_key(b"\xAA",0); kB=make_key(b"\xBB",0); dataA=b"A"*1000; dataB=b"B"*1000

def build_baseline(d):
    io=SlabIO(d,sb,counts,1); io.write(kA,0,mv(dataA),seq=1); io.close()
def hdr_bytes(io_for_header, key, data, seq):
    return io_for_header.header(key,len(data),seq,_crc(data))

with tempfile.TemporaryDirectory() as base:
    seedd=os.path.join(base,"seed"); build_baseline(seedd)
    ioh=SlabIO(seedd,sb,counts,1)  # just to build header bytes
    hA=hdr_bytes(ioh,kA,dataA,1); hB=hdr_bytes(ioh,kB,dataB,2)
    off=ioh._off(0,0); ioh.close()
    seed_g0=open(os.path.join(seedd,"g0.slab"),"rb").read()

    HEADER_STATES={"old_A":hA, "blank":b"\0"*HDR, "new_B":hB, "torn_B":hB[:70]+b"\0"*(HDR-70)}
    PAYLOAD_STATES={"old_A":dataA, "new_B":dataB, "torn_B":dataB[:500]+dataA[500:]}
    total=0; bad=0; served={}
    for hname,hbytes in HEADER_STATES.items():
        for pname,pbytes in PAYLOAD_STATES.items():
            total+=1
            d=os.path.join(base,f"img_{hname}_{pname}")
            os.makedirs(d)
            img=bytearray(seed_g0)
            img[off:off+HDR]=hbytes.ljust(HDR,b"\0")[:HDR]
            img[off+HDR:off+HDR+1000]=pbytes.ljust(1000,b"\0")[:1000]
            open(os.path.join(d,"g0.slab"),"wb").write(img)
            open(os.path.join(d,"g1.slab"),"wb").write(open(os.path.join(seedd,"g1.slab"),"rb").read())
            io=SlabIO(d,sb,counts,1)
            # read under BOTH keys; whatever comes back must equal that key's fully-stored bytes, or raise
            for k,data in ((kA,dataA),(kB,dataB)):
                try:
                    b=mv(data); io.read(k,0,b); got=bytes(b[0])
                    if got!=data: bad+=1; print(f"  CORRUPTION img({hname},{pname}) key {k[:1]} served {got[:8]!r}")
                    else: served.setdefault((hname,pname),[]).append(bytes(k[:1]))
                except OSError: pass
            # scan self-consistency: any yielded slot must read as its header's key exactly
            for slot,seq,key in io.scan(0,epoch=1):
                kk=OffloadKey(key); exp=dataA if kk==kA else (dataB if kk==kB else None)
                if exp is None: bad+=1; print(f"  scan yielded unknown key img({hname},{pname})")
                else:
                    try:
                        b=mv(exp); io.read(kk,slot,b)
                        if bytes(b[0])!=exp: bad+=1; print(f"  CORRUPTION scan/read mismatch img({hname},{pname})")
                    except OSError:
                        pass   # scan may yield a candidate whose payload later fails validation -> load-failure -> recompute (SAFE)
            io.close()
    check(f"exhaustive power-loss: {total} header x payload images, zero corruption", bad==0)
    # sanity: the only images that serve B are (new_B header, new_B payload); the only that serve A are (old_A,old_A)
    only_full = (served == {("old_A","old_A"):[b"\xAA"], ("new_B","new_B"):[b"\xBB"]})
    if not only_full: print("  served map:", served)
    check("only the two fully-consistent images serve a block; all 10 mixed states serve nothing", only_full)

print(f"\n=== {'ALL PASS' if fails==0 else str(fails)+' FAILURE(S)'}")
if __name__ == "__main__":
    sys.exit(1 if fails else 0)


def test_standalone_checks():
    assert fails == 0
