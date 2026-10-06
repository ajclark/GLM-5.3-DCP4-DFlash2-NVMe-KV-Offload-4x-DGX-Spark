"""CPU test of the ring link discovery (glm_roce.ring.choose_links / local_hcas)."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glm_roce.ring import choose_links, local_hcas  # noqa: E402

# GID 3 of every device on the four Sparks (2026-10-01), ring order 06c4, 365c, ddbf, a218.
GIDS = [
    {"rocep1s0f0": "fe80:0000:0000:0000:92a7:53f8:2a2c:dc13", "rocep1s0f1": "::ffff:192.168.101.1",
     "roceP2p1s0f0": "::ffff:192.168.104.2", "roceP2p1s0f1": "::ffff:192.168.100.1"},
    {"rocep1s0f0": "fe80::ed2:1126:68d0:de36", "rocep1s0f1": "::ffff:192.168.101.2",
     "roceP2p1s0f0": "::ffff:192.168.102.1", "roceP2p1s0f1": "::ffff:192.168.100.2"},
    {"rocep1s0f0": "fe80::30f3:1d5d:4b21:6c68", "rocep1s0f1": "::ffff:192.168.105.2",
     "roceP2p1s0f0": "::ffff:192.168.102.2", "roceP2p1s0f1": "::ffff:192.168.103.2"},
    {"rocep1s0f0": "fe80::d6e1:feaa:9cf1:6ed5", "rocep1s0f1": "::ffff:192.168.105.1",
     "roceP2p1s0f0": "::ffff:192.168.104.1", "roceP2p1s0f1": "::ffff:192.168.103.1"},
]


def fake_sysfs(root: Path, gids: dict) -> None:
    for dev, gid in gids.items():
        d = root / dev / "ports" / "1"
        (d / "gids").mkdir(parents=True)
        (d / "state").write_text("4: ACTIVE\n")
        (d / "gids" / "3").write_text(gid + "\n")


def tables():
    out = []
    for gids in GIDS:
        with tempfile.TemporaryDirectory() as tmp:
            fake_sysfs(Path(tmp), gids)
            out.append(local_hcas(3, root=tmp))
    return out


def test_local_hcas_skips_link_local():
    t = tables()
    assert "rocep1s0f0" not in t[0] and t[0]["roceP2p1s0f1"] == "192.168.100.1"


def test_links_avoid_the_dcp_device():
    excl = [frozenset({"rocep1s0f1"})] * 4
    links = choose_links(tables(), excl)
    assert links == [
        ("roceP2p1s0f1", "roceP2p1s0f0"),  # 06c4: cw 365c via .100, ccw a218 via .104
        ("roceP2p1s0f0", "roceP2p1s0f1"),  # 365c: cw ddbf via .102, ccw 06c4 via .100
        ("roceP2p1s0f1", "roceP2p1s0f0"),  # ddbf: cw a218 via .103, ccw 365c via .102
        ("roceP2p1s0f0", "roceP2p1s0f1"),  # a218: cw 06c4 via .104, ccw ddbf via .103
    ], links


def test_links_without_exclusion_still_pair_up():
    links = choose_links(tables(), [frozenset()] * 4)
    for r in range(4):
        n = (r + 1) % 4
        assert links[r][0] in tables()[r] and links[n][1] in tables()[n]


def test_missing_link_raises():
    t = tables()
    del t[2]["roceP2p1s0f0"]
    try:
        choose_links(t, [frozenset()] * 4)
    except RuntimeError as e:
        assert "ranks 1 and 2" in str(e)
    else:
        raise AssertionError("expected a missing-link error")


def test_override():
    links = choose_links(tables(), [frozenset()] * 4, [None, ("a", "b"), None, None])
    assert links[1] == ("a", "b")
