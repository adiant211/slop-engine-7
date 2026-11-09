# smollm: the 3x decode win (attempt 4)

## What was wrong
The per-matmul worker handoff cost ~11 us: `pool.run` built an `Arc<dyn Fn>` under a `Mutex`,
workers re-locked the mutex to clone the job; the worker also slept 100 us after 2e6 spins.
Decode ran at 72 tok/s with **mlp 2x SLOWER at 3 threads than at 1** (6113 ms vs 2949 ms / 201 tokens).

## Fix (kept)
- Pool hands the closure to workers as a raw type-erased pointer in `UnsafeCell<[usize;2]>`
  (transmute of `&(dyn Fn(usize)+Sync)`), published by an `AtomicU64` epoch; no alloc, no mutex,
  no sleep until ~1.3 ms of idling. Master participates and spins on a `done` counter.
- Fuse q/k/v into one `Qw` (rows concatenated, out buffer stride `QKV=960`) and gate/up into one
  (`GU=3072`); per layer the matmuls drop 7 -> 4, so ~150 handoffs/token instead of ~210.
  In attention, remember the q row stride becomes QKV (this bug cost me a broken accuracy run:
  `*qb.add(QDNN + d)` for row 1 silently not rewritten while rows 2,3 were).
Result: decode 72 -> 217-252 tok/s (9.5-9.8x the torch fp32 anchor, was 3.1x), prefill 119 -> 520-580
(1.9x), accuracy unchanged (ppl 22.144 vs 22.046, KLD 0.0061/0.0045). The load-collapse of prefill
(0.39x) was the same sleep-in-worker pathology and is gone.

## Roofline facts measured on this box
- pure sequential read: 34 GB/s on 1 core, 46 GB/s on 3 (that is the ceiling).
- engine now streams ~134.5 MB/token at ~30 GB/s = ~4 ms/token; mlp 59% of bytes, lm_head 21%,
  qkv 12%, attn 12%, oproj 7%. We are memory-bound: per-phase rate is a uniform 30 GB/s.
- per-token (one scale per activation vector) instead of per-32-block scales cuts ~half the uint8
  kernel uops: decode 252->263 (+4%, memory-bound) but prefill 582->851 and ppl-mode 309->466,
  at the cost of KLD 0.006 -> 0.030 (limit 0.05) and ppl +2%. Reverted as a bad trade for decode.
  A per-64-element block scale is the untried middle ground.
- software prefetch in the GEMV: no measurable effect (noise dominates in a 1.5 s run).

## Reminders
- tl;dr: fix threading first, then everything else is bytes moved.
- time a 200-token decode with SMOL_TIME=1 SMOL_NT=1,2,3 before optimising: the scaling ratio
  (not the absolute number) is what tells you whether you are bound by handoff or by DRAM.
