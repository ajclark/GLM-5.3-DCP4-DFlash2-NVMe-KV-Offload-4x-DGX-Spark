#!/usr/bin/env python3
"""Crash-injection + reorder + fuzz tests for the slab durability core (stage/glm-dcp/multinode.py).
The one unacceptable outcome is SILENT CORRUPTION: read() returning bytes that don't match a
completed write, or scan() yielding a slot that read would mis-serve. Every case asserts that
invariant under partial commits, device write-reordering, truncation, and random bit-flips."""
import os, sys, ast, struct, zlib, json, tempfile, random, shutil, contextlib

class OffloadKey(bytes):
    def __new__(cls, b): assert len(b) == 36; return super().__new__(cls, b)
def make_key(h, g): return OffloadKey(((h*32)[:32]) + struct.pack("<I", g))
def get_offload_group_idx(key): return struct.unpack("<I", bytes(key)[32:36])[0]

SRC = open("stage/glm-dcp/multinode.py").read()
WANT = {"_round_up","_crc","SLAB_MAGIC","SLAB_HEADER_BYTES","SLAB_VERSION","SLAB_META_VERSION",
        "_SLAB_HDR","SLAB_META","SLAB_EPOCH","DRAFTER_PER_TARGET","slab_geometry","SlabIO",
        "_atomic_write_json","_atomic_write_text","_drain_iov"}
segs=[ast.get_source_segment(SRC,n) for n in ast.parse(SRC).body
      if (isinstance(n,(ast.FunctionDef,ast.ClassDef)) and getattr(n,"name","") in WANT)
      or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in WANT for t in n.targets))
      or isinstance(n,ast.Assert)]
ns={"os":os,"struct":struct,"zlib":zlib,"json":json,"memoryview":memoryview,
    "OffloadKey":OffloadKey,"get_offload_group_idx":get_offload_group_idx}
exec("\n".join(segs), ns)
SlabIO, HDR, _round_up, _crc = ns["SlabIO"], ns["SLAB_HEADER_BYTES"], ns["_round_up"], ns["_crc"]
_SLAB_HDR, MAGIC, VER = ns["_SLAB_HDR"], ns["SLAB_MAGIC"], ns["SLAB_VERSION"]

fails=0
def check(name, cond):
    global fails; print(("PASS" if cond else "FAIL"), name); fails += 0 if cond else 1
def mv(b): return [memoryview(bytearray(b))]

class Crash(Exception): pass

@contextlib.contextmanager
def crash_after(k):
    """Abort after k low-level write calls. write() = _pwrite_all(blank), os.pwritev(payload),
    _pwrite_all(seal) -> 3 calls for a single-iov payload."""
    calls={"n":0}
    orig_static = SlabIO.__dict__["_pwrite_all"].__func__   # the raw function under the staticmethod
    orig_pwritev = os.pwritev
    def pw(fd,data,off):
        calls["n"]+=1
        if calls["n"]>k: raise Crash()
        return orig_static(fd,data,off)
    def pwv(fd,bufs,off):
        calls["n"]+=1
        if calls["n"]>k: raise Crash()
        return orig_pwritev(fd,bufs,off)
    SlabIO._pwrite_all = staticmethod(pw); os.pwritev = pwv
    try: yield
    finally:
        SlabIO._pwrite_all = staticmethod(orig_static); os.pwritev = orig_pwritev

def read_ok(io, key, slot, expect):
    try:
        b=mv(expect); io.read(key,slot,b); return bytes(b[0])==expect
    except OSError:
        return None   # rejected (safe)

def invariant_holds(d, sb, counts, epoch, kA, dataA, kB, dataB):
    """A fresh reader must never serve wrong bytes; scan must be self-consistent with read."""
    io = SlabIO(d, sb, counts, epoch)
    ok = True
    for k,data in ((kA,dataA),(kB,dataB)):
        r = read_ok(io, k, 0, data)
        if r is False: ok = False        # served WRONG bytes -> corruption
    scanned = {slot:bytes(key) for slot,seq,key in io.scan(0,epoch=epoch)}
    if 0 in scanned:
        key = OffloadKey(scanned[0]); exp = dataA if key==kA else dataB
        if read_ok(io, key, 0, exp) is not True: ok = False   # scan yielded a slot read can't serve
    io.close(); return ok

with tempfile.TemporaryDirectory() as base:
    sb=[_round_up(HDR+1000,4096), _round_up(HDR+400,4096)]; counts=[16,16]
    kA=make_key(b"\xAA",0); kB=make_key(b"\xBB",0); dataA=b"A"*1000; dataB=b"B"*1000
    seed = os.path.join(base,"seed")
    io=SlabIO(seed,sb,counts,1); io.write(kA,0,mv(dataA),seq=1); io.close()  # slot 0 = valid A

    # 1-3. crash while REUSING slot 0 with B, aborting after 0/1/2 low-level writes
    for ab in range(0,3):
        d=os.path.join(base,f"crash{ab}"); shutil.copytree(seed,d)
        io=SlabIO(d,sb,counts,1)
        with crash_after(ab):
            try: io.write(kB,0,mv(dataB),seq=2)
            except Crash: pass
        io.close()
        check(f"crash after {ab} writes (reuse A->B): no corruption, scan consistent",
              invariant_holds(d,sb,counts,1,kA,dataA,kB,dataB))

    # 4. full write (abort after 3 = no crash) -> B is readable
    d=os.path.join(base,"full"); shutil.copytree(seed,d); io=SlabIO(d,sb,counts,1)
    with crash_after(3):
        io.write(kB,0,mv(dataB),seq=2)
    check("full reuse write: B readable, A gone", read_ok(io,kB,0,dataB) is True and read_ok(io,kA,0,dataA) is None); io.close()

    # 5. DEVICE REORDER: a valid sealed header for B lands, but the payload is still A's bytes
    #    (device persisted the seal before the payload). read(B) MUST fail the payload CRC.
    d=os.path.join(base,"reorder"); shutil.copytree(seed,d); io=SlabIO(d,sb,counts,1)
    fd=io.fds[0]; off=io._off(0,0)
    good_hdr = io.header(kB, len(dataB), seq=2, payload_crc=_crc(dataB))  # header claims B's crc
    os.pwrite(fd, good_hdr, off)                                          # but payload is still A
    check("device reorder (B header over A payload): read rejects via payload CRC", read_ok(io,kB,0,dataB) is None); io.close()

    # 6. partial 128B header (torn header): first 64 bytes of a valid header, rest stale -> reject
    d=os.path.join(base,"parthdr"); shutil.copytree(seed,d); io=SlabIO(d,sb,counts,1)
    os.pwrite(io.fds[0], good_hdr[:64], io._off(0,0))
    check("partial header write rejected", read_ok(io,kA,0,dataA) is None and read_ok(io,kB,0,dataB) is None); io.close()

    # 7a. PAYLOAD truncated, header intact: scan may list slot 3 (header-only), but read MUST reject
    #     it, and the invariant (a scanned slot is read-correct-or-raises) must hold.
    kC=make_key(b"\xCC",0)
    d=os.path.join(base,"truncpay"); shutil.copytree(seed,d); io=SlabIO(d,sb,counts,1)
    io.write(kC,3,mv(dataA),seq=3); io.close()
    with open(os.path.join(d,"g0.slab"),"r+b") as f: f.truncate(io._off(0,3)+HDR+10)  # cut mid-payload
    io=SlabIO(d,sb,counts,1)
    scanned=[s for s,_,_ in io.scan(0,epoch=1)]
    scan_safe=all(read_ok(io,kC,s,dataA) is not False for s in scanned)  # never a wrong-bytes serve
    check("payload-truncated: read rejects slot 3 AND scan stays consistent",
          read_ok(io,kC,3,dataA) is None and scan_safe); io.close()
    # 7b. HEADER truncated (file ends mid-header): scan stops there, that slot is not yielded, read raises
    d=os.path.join(base,"trunchdr"); shutil.copytree(seed,d); io=SlabIO(d,sb,counts,1)
    io.write(kC,3,mv(dataA),seq=3); io.close()
    with open(os.path.join(d,"g0.slab"),"r+b") as f: f.truncate(io._off(0,3)+40)  # cut mid-header
    io=SlabIO(d,sb,counts,1)
    check("header-truncated: scan stops (slot 3 not yielded) and read raises",
          all(s!=3 for s,_,_ in io.scan(0,epoch=1)) and read_ok(io,kC,3,dataA) is None); io.close()

    # 8. cross-group contamination: a g1 header written into g0's file -> scan(group 0) drops it
    d=os.path.join(base,"xgroup"); shutil.copytree(seed,d); io=SlabIO(d,sb,counts,1)
    kg1=make_key(b"\xDD",1); h_g1=io.header(kg1,len(dataB),seq=4,payload_crc=_crc(dataB))
    os.pwrite(io.fds[0], h_g1, io._off(0,2)); os.pwrite(io.fds[0], dataB, io._off(0,2)+HDR)
    scanned=[s for s,_,_ in io.scan(0,epoch=1)]
    check("cross-group header in g0 dropped by scan (group mismatch)", 2 not in scanned); io.close()

    # 9. duplicate key across slots: newer seq must win in a rebuild-style pick
    d=os.path.join(base,"dup"); shutil.copytree(seed,d); io=SlabIO(d,sb,counts,1)
    io.write(kA,4,mv(dataA),seq=10); io.write(kA,5,mv(dataB),seq=20); io.close()  # same key, slots 4 (old) & 5 (new)
    io=SlabIO(d,sb,counts,1)
    present={key:(seq,slot) for slot,seq,key in io.scan(0,epoch=1) if OffloadKey(key)==kA}
    # emulate the scheduler's "sorted by seq, later wins": newest seq for kA is slot 5
    newest = max((seq,slot) for slot,seq,key in io.scan(0,epoch=1) if OffloadKey(key)==kA)
    check("duplicate key: newest seq is slot 5", newest[1]==5); io.close()

    # 10. FUZZER: random writes/reuses + random bit-flips; read must never serve wrong bytes
    rng=random.Random(1234); d=os.path.join(base,"fuzz"); io=SlabIO(d,sb,counts,1)
    truth={}   # slot -> (key, data)
    keys=[make_key(bytes([i]),0) for i in range(20)]
    corrupt=0
    for step in range(400):
        slot=rng.randrange(counts[0]); key=rng.choice(keys); data=bytes([rng.randrange(256)])*1000
        io.write(key,slot,mv(data),seq=step+1); truth[slot]=(key,data)
        if rng.random()<0.30:   # flip a random byte somewhere in this slot (payload or header)
            fd=io.fds[0]; pos=io._off(0,slot)+rng.randrange(HDR+1000)
            b=os.pread(fd,1,pos); os.pwrite(fd, bytes([b[0]^0xFF]), pos)
            truth[slot]=(key,None)   # now possibly corrupt: read must return correct-or-error, never wrong
    io.close(); io=SlabIO(d,sb,counts,1)  # simulate reopen
    for slot,(key,data) in truth.items():
        r=read_ok(io,key,slot,data if data is not None else b"?"*1000)
        if data is not None and r is False: corrupt+=1        # clean slot served wrong bytes
        if data is None and r is True: pass                    # a flip that CRC happened to still pass on identical data is impossible here
        if data is None and r is False: corrupt+=1             # returned wrong bytes for a corrupted slot
    # every scanned slot must be readable-correct or the scan shouldn't have yielded it
    for slot,seq,key in io.scan(0,epoch=1):
        k=OffloadKey(key)
        # we don't know the exact bytes for a scanned slot unless it's a clean slot in truth
        t=truth.get(slot)
        if t and t[1] is not None and OffloadKey(bytes(k))==t[0]:
            if read_ok(io,k,slot,t[1]) is not True: corrupt+=1
    io.close()
    check(f"fuzz 400 writes + 30% bit-flips: zero wrong-bytes served (corrupt={corrupt})", corrupt==0)

    # 11. seq recovery across reboot: the max stored seq must be recoverable so the scheduler
    #     resumes ABOVE it (a post-reboot write can never carry a lower seq than a recovered slot;
    #     with SHA-256 content-addressed keys a duplicate key is duplicate content anyway).
    d=os.path.join(base,"seqrec"); io=SlabIO(d,sb,counts,1)
    for i,slot in enumerate(range(6)): io.write(make_key(bytes([i]),0), slot, mv(dataA), seq=100+i*7)
    io.close(); io=SlabIO(d,sb,counts,1)   # reboot
    maxseq=max((seq for _,seq,_ in io.scan(0,epoch=1)), default=0)
    check("seq recovery: max stored seq (142) recoverable from headers after reboot", maxseq==100+5*7)
    # 12. same key twice, different slots, different seq: both slots carry a self-describing header;
    #     read of either returns its own stored bytes (never the other's) -> no cross-slot bleed.
    d=os.path.join(base,"dupkey2"); io=SlabIO(d,sb,counts,1)
    io.write(kA,0,mv(dataA),seq=1); io.write(kA,1,mv(dataB),seq=2)   # same key, DIFFERENT payloads (only possible if key were non-content; tests read isolation)
    io.close(); io=SlabIO(d,sb,counts,1)
    r0=read_ok(io,kA,0,dataA); r1=read_ok(io,kA,1,dataB)
    check("same-key two slots: each slot reads its OWN payload (no cross-slot bleed)", r0 is True and r1 is True); io.close()

print(f"\n=== {'ALL PASS' if fails==0 else str(fails)+' FAILURE(S)'}")
sys.exit(1 if fails else 0)
