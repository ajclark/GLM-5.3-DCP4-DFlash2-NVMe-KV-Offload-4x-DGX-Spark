 Ranked concerns
 1. Stuck-prepared hole (line 9). Wake triggers only on suspended && wanted=1. If
    spark-idle.sh (or the watcher's systemd cgroup) dies between prepare and
    commit, nodes sit prepared, requests are gated, wanted=1, and no rule ever
    fires — hold forever. Resume-from-prepared already drops the gate, so line 9
    should read "any non-active and wanted=1 → --up".
 2. No --up backoff. A failing --up (ring verification) is retried every 10 s →
    near-continuous CX-7 power cycling. Keep one line: after a failed --up, don't
    retry for N minutes (or at all; failed resume needs a human anyway).
 3. Line 4 underspecified. Commit fails on a subset → that process is in failed.
    Does power-off proceed? Say so.
 4. Inline serving makes status blind during resume (up to ~60 s device wait).
    Safe (hold), but the loop must log hold reasons or a failed node is silent.
 Per line: 1 agree — loopback also removes commit exposure on the mgmt net;
 no-walk fine, the walk never gave the watcher visibility anyway. 2 agree. 3
 agree. 4 agree, see 3. 5 agree. 6 agree. 7 agree. 8 agree (idle=-1 correctly
 never qualifies). 9 disagree as written, see concern 1. 10 agree. 11 agree —
 this is exactly what makes restore-on-vanish deletable. 12 agree, but it's
 docs-only enforcement; a stray second process just loses its control endpoint,
 same as before.
 Deletions: all justified. /metrics and /load: correct, the 90-min-idle spin-down
 mid-generation costs one bounded ~30 s stall at the collective — acceptable.
 Abort: resume-from-prepared is equivalent. Restore-on-vanish: covered by line
 11. Keep only the --up backoff (concern 2).
 glm done
