# Adaptive speculation execution status

## Current state:2026-09-09 05:34 UTC — experimental phase complete

- All R6/R7 experiments, declared follow-ups, correctness, restart, overhead and
  final source audits are complete. No controller, benchmark or sampler remains.
- Exact original container IDs/images/config/hostconfig/mounts/mounted-source hashes
  restored on all4nodes; final generation passed. TP4/DCP2,maxlen180224,maxseq12,
  KV6GB/rank unchanged. Proof:adaptive-v4-20260909-r7/original-restoration-audit.json.
- Implemented version4 adaptive verification remains opt-in; defaultpolicyoff.
  Source/calibration/corpora stayed frozen throughout evaluation. All370 local tests
  passed9.49s, threepatches reproduce overlays exactly, stage checksums match.
- Initial180requests:prose+15.80%(95%13.24–18.73%);coding+0.99%(-2.11–4.44%).
  Initial coding gate narrowly missed. Declared90request coding extension yields
  90coding pairs,+1.14%(95%-1.30–3.64%),inside2% margin; follow-up evidence only.
  Aggregate code25.07→25.47tok/s,prose15.36→17.80. Initial locked report retained.
- Device decode energy:prose-17.37%;combined coding-2.85%(95%savings0.40–5.39%).
  Whole-cluster wall-energy gate unmeasured; no hardware idle-watt improvement
  claimed. CX-7 cycling remains paused. No broad promotion.
- R6 one-repeat code/prose repo gains:4k-0.2%/-9.5%;32k+8.0%/-7.9%;
  100k+34.6%/+37.2%;170k+41.7%/+42.1%. R7 declared3repeat follow-ups:
  4k+4.89%/+15.33%;32k+7.19%/+24.07%. Initial prose regressions did not repeat.
  One prompt/category; exploratory. First170k prefill402s cold;decode-only gains.
- Correctness:fourcompletefunctions identical245IDs;C1/C2/cancel/fallback passed;
  C2overlap~81tok/s ratio1.00018;six256-token reasoning controls;blind prose
  openings10ties/adaptive3/fixed2. Random32k/100k count30+marker all pass;32k
  all66IDs identical,100k differs only one newline token before adaptation.
- Once-store/restart passed:102360freshprompt,13.53GBstore,0hits; R7 loaded
  102016external cached tokens,0localhits,11.06GBload,exact127continuationIDs.
  Prompt-only disk policy preserved; supplied80generatedIDs recomputed as suffix.
- Tracing9identical-output pairs:centraltime+0.56%,95%-0.11–1.81%, clocks match.
  Strict upper1%budget not established; separateboots/threeprompts limitation.
- R6audit30154verifyevents,0integrityerrors,FULL M2/4/6/8 on allranks in R6/R7.
  No OOM/guardtrips. Serving min1641MiBhead(R6)/1793MiB(R7); other ranks>=3098.
  Head swap-out2.60MiB R6 and8KiB R7; fullPSIavg10=0. Do not claim zero swaps.
- Copyselector offline CPU/opportunity screen complete; insufficient evidence for
  GPU integration on edit/continuation. See copy-branch-decision.json.
- README.md,COMPLETION-AUDIT.md,and completion-decision.json are final reports.
  All experimental raw results and frozen locks remain available locally.
