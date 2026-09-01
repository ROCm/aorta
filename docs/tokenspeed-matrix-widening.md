# Widening the TokenSpeed serving matrix

[TokenSpeed serving benchmarks](tokenspeed-serving.md) records what the
`tokenspeed_serve` workload has actually measured: five models end to end, a
six-cell load sweep on Qwen3-0.6B, and a tensor-parallel axis that stops at
TP=2 because TP=4 would not come up. This document plans the three directions
that matrix should grow in — more models, heavier loads, wider tensor
parallelism — and reports what came back from running all three on hardware.

Two of the three were expected to be scheduling questions and one a research
question. That was half right. TP=4 was the research question and now works.
Heavier loads behaved as a scheduling question and answered where the
concurrency ceiling is. Adding models was supposed to be the easy one and was
not: the highest-value model in the list cannot come up on this image at all,
for a reason that turns out to explain why the gap it was meant to fill existed.

Everything here is measured on one gfx950 node (MI355X, 8 × 309 GB, 3 TB host
RAM, ROCm 7.0.2.2) against image
`lightseekorg/tokenspeed-amd@sha256:60c12e37c01496891053b9c30c4204e5d1cf9b4b641859d3aadcbd95bccc7c78`,
the digest every committed recipe pins.

## Assumptions

A planning document (`Aorta planning & Updates.docx`) was referenced for this
work and is not reachable from this machine — there is no `/mnt/c` mount and no
`Downloads` directory. Rather than guess at it, the following were assumed, and
each is the kind of thing that document might contradict:

- **The matrix is for engineering triage, not for publication.** Recipes are
  chosen to make a regression attributable, not to produce a headline number.
  That is why the model list below prioritises breaking a confound over adding
  parameter count.
- **Node time is the scarce resource, not wall-clock.** Cost is quoted in
  node-hours per recipe so a scheduler can decide, and recipes are kept small
  enough to run individually rather than as one long sweep.
- **Nothing here is gated in CI yet.** No serving recipe is in the nightly
  matrix, so these recipes are additive: they cannot redden an existing gate.
  If perf gating lands first, the new recipes need blessed baselines before
  they mean anything, which is a second scheduling ask.
- **`gpt-oss-20b` stays the reference model** for anything cross-cutting, since
  it is the model TokenSpeed publishes AMD numbers against and the one both TP
  recipes use.

## Where the matrix is today

| Axis | Was covered | Gap | Now |
|---|---|---|---|
| Model size | 0.6B → 8B dense; 21B MoE | nothing dense above 8B | 32B dense added |
| Architecture | dense (Qwen3), MoE (gpt-oss) | MoE and MXFP4 perfectly confounded | separated, but only in eager mode — the BF16 MoE path cannot capture |
| Quantisation | BF16, MXFP4 | no FP8; no quantised/unquantised pair of one model | unchanged; probe first |
| Repo access | ungated only | the `hf_token_env` path has never run | unchanged; no token on this machine |
| Concurrency | 1, 8, 32, 64 | curve still rising at 64 | 128 and 256 added; ceiling found |
| Request shape | ISL 4096/OSL 128, ISL 128/OSL 1024 | adequate | unchanged |
| Tensor parallelism | 1, 2 | 4 did not come up | **1 / 2 / 4 / 8 all work**, once the host KV tier is bounded, and 8 is worth having on a model that needs it |

## Workstream 3: tensor parallelism beyond TP=2

Taken first because it is the only one that was not simply a matter of booking
a node, and because its answer changes what the other two cost.

### What was wrong

`tokenspeed-serve-gptoss-tp.yaml` stops at TP=2. TP=4 failed reproducibly with
an out-of-memory raised from `flat_host_mirror.py:127`, and the previous
diagnosis got as far as "the host-side KV mirror" and stopped, with two things
noted as puzzling: the node has 3 TB of RAM and eight 309 GB cards, so nothing
looked scarce, and the error named CUDA for what the traceback showed to be a
host allocation.

Reading the source out of the image answers the second puzzle immediately.
`FlatHostMirror.__init__` allocates with

```python
pin = torch.cuda.is_available()
...
torch.zeros((self.num_host_pages * self.page_size, *dev.shape[1:]),
            dtype=dev.dtype, pin_memory=pin)
```

`pin_memory=True` is page-locked host memory, which torch allocates through
`hipHostMalloc`. A failure there is a HIP failure, so torch reports it as
`torch.AcceleratorError: CUDA error: out of memory` — a *host* allocation
correctly reporting a device-runtime error. There was never a device-memory
problem to find.

The first puzzle — why 3 TB is not enough — is answered by the sizing, in
`flat_memory_executor.py`:

```python
if host_size_gb > 0:
    num_pages = int(host_size_gb * 1e9 // bytes_per_host_page)
else:
    num_pages = int(device_pool_size * host_ratio) // page_size + 1
```

with `kvstore_ratio` defaulting to 2.0 and `kvstore_size` to 0, so ratio sizing
is what runs by default. The mirror is therefore **twice this rank's device KV
pool**, and every rank owns a whole MI355X and fills it. Tensor parallelism
does not shrink that: sharding halves the KV bytes a rank holds per token and
doubles the number of tokens that fit in the same ~250 GiB, and the product —
which is what the mirror mirrors — does not move.

So the host tier is **roughly a per-rank constant**, and the node-wide pinned
footprint is therefore roughly linear in TP.

Two cells at TP=4, differing only in `--kvstore-size`, settle the direct case.
Both logged `bytes_per_host_page=393216` and `device_pool.size=40202624`:

| Cell | `host_size_gb` | `num_host_pages` | Per rank | × 4 ranks | Outcome |
|---|---|---|---|---|---|
| default | 0 (ratio 2.0) | 1256333 | 494.01 GB | **1976 GB** | `torch.AcceleratorError: CUDA error: out of memory` |
| `--kvstore-size 128` | 128 | 325520 | 128.00 GB | 512 GB | passed, 32/32 served, 0 failed |

Both per-rank figures are what the engine logged rather than arithmetic done
here, and both match the formula: `128e9 // 393216 = 325520`, and
`int(40202624 × 2.0) // 64 + 1 = 1256333`.

The engine prints `bytes_per_host_page` and `device_pool.size` on every rank,
so one run of the bounded recipe records what the *default* would have asked
for at each TP without ever allocating it. Applying the ratio branch to the
logged values:

| TP | `device_pool.size` | `bytes_per_host_page` | Default per rank | Default node-wide | Source |
|---|---|---|---|---|---|
| 1 | 8587008 | 1572864 | 422.07 GB | 422 GB | logged allocation |
| 2 | 17989504 | 786432 | 442.11 GB | 884 GB | derived |
| 4 | 40202624 | 393216 | 494.01 GB | **1976 GB** | logged allocation |

"Roughly" constant, then, not exactly: per rank it rises 17% from TP=1 to TP=4,
because sharded weights free VRAM and the device pool grows into it. Node-wide
it rises 4.7×, and that is what matters — the pass/pass/fail boundary on a 3 TB
node falls between 884 GB and 1976 GB. It is also why the failure looked like it
had nothing to do with TP: every rank was individually reasonable and only the
sum was not.

Two of those three rows are allocations the engine actually logged: 494.01 GB
per rank from the failing TP=4 control, and 422.07 GB from a TP=1 cell run at
the default for a separate A/B. The derivation predicted 422.07 GB for that
cell before it ran, which is what makes the TP=2 row — the one no cell has
allocated — worth believing.

For scale, the same log across the other models in this document shows default
host tiers between 341 GB and 470 GB per rank. The tier is roughly
model-independent, as `tokenspeed.md` says, because it is sized from the device
pool and the device pool is sized from the card.

**Why the engine's own guard did not catch it.** `FlatMemoryExecutor` checks
its allocation before making it:

```python
available_bytes = psutil.virtual_memory().available - _HOST_MEM_HEADROOM_BYTES
if requested_bytes > available_bytes:
    raise ValueError("Not enough host memory for the flat host tier...")
```

That check is per process. Each of the four ranks compares *its own* 494 GB
against the *whole node's* free memory, sees roughly 2.8 TB, and passes. All
four then allocate concurrently, and the one that loses the race dies inside
`torch.zeros` with the HIP error instead of the clear ValueError the guard was
written to produce. `free -g` during the failing run showed 2483 GB used and
539 GB available, so the ranks had taken about 2.4 TB between them before one
of them could not proceed.

### What was ruled out, and how

- **Container shared memory.** Ruled out on mechanism rather than by
  experiment, which is stronger than the A/B that
  [tokenspeed-serving.md](tokenspeed-serving.md) correctly retracted. Pinned
  host memory from `hipHostMalloc` is not `/dev/shm`; `shm_size` cannot bound
  it or fail it. The re-run that document asks for is not needed.
- **Contamination from an earlier cell.** Already ruled out upstream, and
  independently here: the passing TP=4 run was the *first* TP=4 cell on a
  freshly allocated node.
- **Device memory.** The failing allocation is host-side, and the passing run
  reached the same device configuration.
- **`HIP_VISIBLE_DEVICES` scoping.** Not implicated. Both runs pinned four
  devices identically and differed only in `--kvstore-size`.
- **A genuine 3 TB shortfall.** The node has the RAM; 1976 GB of it simply
  cannot all be page-locked alongside weights, device-pool bookkeeping and page
  cache.

### The fix, and what it costs

`--kvstore-size <GB>` sets an explicit per-rank budget and overrides the ratio,
which is what makes the footprint flat in TP instead of linear.
`recipes/tokenspeed/tokenspeed-serve-gptoss-tp-wide.yaml` carries
`--kvstore-size 128` on **every** cell, TP=1 included: a TP axis whose host
tier changes per cell measures two things at once, and the TP=1 row is what
proves the bound is free.

It should be free here, and it was measured rather than assumed. The KVStore
host tier is an L2 cache for *prefix reuse*, and these cells run
`dataset: random`, whose prompts share no prefixes — the tier is allocated and
never hit. Two TP=1 cells in one run, on one GPU, differing only in the flag:

| Cell | Startup (s) | TTFT p50 (ms) | TPOT p50 (ms) | Output tok/s |
|---|---|---|---|---|
| default tier (422 GB) | 175 | 67.7 | 7.92 | 954.2 |
| `--kvstore-size 128` | 111 | 67.1 | 7.89 | 959.5 |

Serving is unchanged: TTFT differs by 0.9%, TPOT by 0.4%, throughput by 0.6%,
with the bounded cell marginally ahead on all three — noise, in the direction
that costs nothing. That is a claim about *this load*, and it is stated in the
recipe so nobody carries the value into a prefix-caching benchmark without
re-measuring.

The A/B was worth running for a second reason. The bounded TP=1 cell reports
948.9–959.5 output tok/s across three separate runs here, against the 991 the
existing TP recipe recorded — a ~4% gap that could have been read as the bound
costing something. The *default* cell in the same run posts 954.2, inside that
same band, so the gap is run-to-run and node-to-node spread rather than the
flag. Without a same-run control there was no way to tell those apart.

Bring-up is the one thing that does change, and in the useful direction: 175 s
to 111 s, a 37% cut. `tokenspeed.md` notes that bring-up is dominated by
allocating the KV cache and the pinned host tier rather than by loading
weights, and this is that observation from the other side — allocating 128 GB
of page-locked memory instead of 422 GB is most of a minute.

128 GB rather than 494/4 ≈ 123: a round number below the per-rank default, so
TP=4 asks for 512 GB node-wide, slightly *more* than TP=1's 422 GB default and
well under TP=2's 884 GB.

### What the axis measures once it runs

All four cells passed, 96/96 requests served each, none failed. Load is the
same as the existing TP recipe (32 requests per step, ISL 512 / OSL 128,
concurrency 8, three measured steps) so the rows are comparable to it. This
table is the recipe as committed, measured end to end in one allocation on one
node, so the rows are joinable to each other as well.

| Cell | Startup (s) | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | Output tok/s | vs TP=1 |
|---|---|---|---|---|---|---|
| tp1 | 127 | 67.3 | 69.1 | 7.70 | 980.8 | 1.000× |
| tp2 | 129 | 65.7 | 66.9 | 7.27 | 1033.6 | 1.054× |
| tp4 | 114 | 67.0 | 68.7 | 7.11 | 1053.5 | **1.074×** |
| tp8 | 127 | 72.3 | 73.9 | 7.49 | 999.5 | 1.019× |

**The whole axis works, and it buys almost nothing.** TP=4 is the peak at 7.4%
more throughput for four times the hardware; TP=8 gives most of that back and
ends 1.9% above a single card while being 7% worse on TTFT. TTFT is otherwise
flat across the axis to within 2.5%, which is the expected shape: at 21B with
MXFP4 the model fits comfortably in one MI355X, so extra ranks relieve no
constraint and pay collective cost on every step. The result worth recording is
that the path works and reports coherently at up to eight ranks, not that it is
fast — and that a model this size is the wrong instrument for measuring TP
scaling. See `tokenspeed-serve-tp-large.yaml` below for the same axis on a
model that is not.

An earlier measurement of the first three cells, on a different node and a
different allocation, returned 948.9 / 984.2 / 1021.0. That is 3.2% / 4.8% /
3.1% below the table above — same ordering, same conclusion, and a useful
number to have: it is what cross-node run-to-run agreement looks like on a cell
that does not stall, against the 25% the `conc-64` cell produced.

Do not read the startup column as scaling with TP. Every cell here ran against
a warm Gluon cache and came in at 114-129 s. The first cell of the earlier
run, against a cold one, paid 939 s. Startup measures cache state, not rank
count.

The obvious next move is to have `setup()` reject a wide-TP cell that does not
bound the tier. It is deliberately not done, because the threshold is a
property of *this class of node* (3 TB, 309 GB cards) and *this image's*
defaults (`kvstore_ratio` 2.0, `gpu_memory_utilization` 0.95), not of the
workload. On a 6 TB node TP=4 at the default is fine, and a workload that
refused it would be rejecting a valid recipe with a hardcoded number — the
failure mode the rest of this integration works hard to avoid.

What is enforced instead is repo hygiene:
`test_a_wide_tp_recipe_bounds_the_host_kv_tier` fails if any *committed* recipe
asks for `--tensor-parallel-size` ≥ 4 without one of `--kvstore-size`,
`--kvstore-ratio` or `--disable-kvstore`. That keeps our recipes correct on the
nodes we have without deciding anything for someone else's.

The reason it is worth a test at all is the blast radius. Ranks that die this
way hold their GPU memory well past the workload's five-minute wait: after the
failing control run, GPUs 2 and 3 still held 251 GiB each with no KFD processes
and no containers alive. Reclaiming that needs `rocm-smi --gpureset`, which
requires privileges an unprivileged user on a scheduler-allocated node does not
have — the remaining validation for this document had to be moved to GPUs 4–7.
So one naked TP=4 cell does not just fail; it strands half the node for
everything that runs after it, for the life of the allocation.

### TP=8, measured

TP=8 was the open experiment: at `--kvstore-size 128` the arithmetic says it
asks for 1024 GB of pinned host memory node-wide, which fits in 3 TB, and the
arithmetic said TP=4 would fail too and it did, so it was worth measuring
rather than asserting. It could not be measured on the previous allocation for
a mundane reason — the failing TP=4 control had stranded 251 GiB on GPUs 0-3
and TP=8 needs all eight. This was run on a fresh node with `rocm-smi`
confirmed clean beforehand.

**The memory question is answered, and directly.** All eight ranks logged the
allocation:

```
[ATTN TP RANK 0..7] Allocating 128.00 GB pinned host memory for the flat host
  tier (num_host_pages=651041 bytes_per_host_page=196608 host_size_gb=128
  host_ratio=2.0 device_pool.size=78514816)
```

Eight ranks × 128 GB = 1024 GB, allocated, on a 3 TB node. Unbounded at the
default `host_ratio=2.0` the same eight ranks would have asked for the per-rank
~494 GB constant each — 3952 GB — and died the way TP=4 died. The bound is not
a nicety at TP=8; it is the difference between running and not.

**The performance question needs a different model, which is the caveat this
document raised before running it and which the numbers confirm.** The
gpt-oss-20b TP=8 row is in the table above and it is a control, not a result:
999.5 tok/s, 1.019× TP=1, below both TP=2 and TP=4. 21B at MXFP4 with 3.6B
active fits on one MI355X with room to spare, so the axis was never going to
say anything about scaling on that model, and it does not.

So the axis was run again on Qwen3-32B — dense, BF16, ~65 GB of weights,
materially more arithmetic per token — as
`recipes/tokenspeed/tokenspeed-serve-tp-large.yaml`. All four cells passed,
96/96 requests each:

| Cell | Startup (s) | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | Output tok/s | vs TP=1 |
|---|---|---|---|---|---|---|
| tp1 | 117 | 98.6 | 1335.7 | 14.11 | 541.1 | 1.00× |
| tp2 | 115 | 91.2 | 1032.6 | 10.72 | 704.1 | 1.30× |
| tp4 | 115 | 95.8 | 845.4 | 8.54 | 867.4 | **1.60×** |
| tp8 | 114 | 108.1 | 862.1 | 8.60 | 852.3 | 1.58× |

**TP=8 comes up, serves correctly, and returns nothing.** It is 1.7% *below*
TP=4 on throughput and 13% worse on median TTFT, which is what paying
collective cost for capacity you do not need looks like. The curve peaks at
TP=4 and the marginal return is already falling before it: 1.30× for the first
doubling, 1.23× for the second, 0.98× for the third.

So the axis demonstrates three things, and they should not be collapsed into
one:

1. **The eight-way path is correct.** Eight ranks initialise, allocate their
   bounded host tier, capture graphs and serve every request. Nothing about
   TP=8 is broken.
2. **A model that needs the cards does scale, up to a point.** Qwen3-32B
   returns 1.60× over four cards where gpt-oss-20b returns 1.074×. The earlier
   "TP barely helps" reading was a property of the model, not of the stack —
   which is what the caveat predicted, and it is worth having falsified
   cheaply.
3. **Nothing in this matrix needs eight cards.** The largest model available
   here saturates at four. A TP=8 row is worth keeping as the row that shows
   where saturation is, in the same way `conc-256` is kept as the row that
   shows where the concurrency cliff is — not as a configuration to recommend.

A gpt-oss-20b TP=8 cell **is committed**, as `tp8` in
`tokenspeed-serve-gptoss-tp-wide.yaml`, but it took two attempts and the first
failure is worth recording because it looks like a TP failure and is not. On
the first attempt the eight ranks allocated their host tier exactly as above
and torch distributed initialised across the eight-way mapping. What then
failed was the Rust model gateway, on two network fetches that the pre-warmed
HF cache does not cover:

```
Failed to load Harmony encoding: error downloading or loading vocab file
  https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken
  ... client error (Connect) / dns error
Failed to load tokenizer 'openai/gpt-oss-20b': ... error sending request for
  url (https://huggingface.co/api/models/openai/gpt-oss-20b/revision/main)
```

Both hosts resolve and answer 200 from the node itself, so the container lost
egress rather than the network being absent — and the identical cell, re-run
unchanged twenty minutes later on the same node, came up in 118 s and served
96/96. So it was transient. The Harmony encoding is the interesting half: it is
a tiktoken vocabulary fetched from an OpenAI blob endpoint, it is not in the HF
cache and pre-warming that cache does not pre-warm it, so **every gpt-oss
bring-up on this stack depends on a live network fetch that no amount of cache
pre-warming removes**. That is a reliability hazard for the whole gpt-oss line,
not a TP=8 one, and it is recorded in `tokenspeed-serving.md` as such. It also
cost 26 minutes of node time to fail, because it fails after the ranks have
loaded weights and allocated their KV pools rather than before.

Two further items, the first of them now cheap:

- **Confirm the per-rank constant directly at TP=1 and TP=2.** The 494 GB
  figure is measured at TP=4 and derived algebraically for the others. The
  TP=8 log above adds a fourth point at the *bounded* size, which confirms the
  page arithmetic but not the ratio. The sizing log prints `bytes_per_host_page`
  and `device_pool.size` on every rank, so one run of the committed recipe at
  default settings records all of them and turns the derivation into a
  measurement. Cheap: it is a log grep on a run that is happening anyway.
- **Report it upstream.** Done, as evidence on
  [lightseekorg/tokenspeed#297](https://github.com/lightseekorg/tokenspeed/issues/297),
  which reports the same per-process guard in the radix `HostKVCache` path on
  NVIDIA and was closed by the stale bot rather than by a fix. The flat
  executor carries its own copy of the check, so a fix confined to
  `kv_cache_host.py` would not cover what we hit. A rank that knew its world
  size could check `world_size × requested` against available memory and raise
  the ValueError it already has. That is a small upstream patch and it would
  prevent the stranded GPUs, which is the expensive part of the failure.

## Workstream 2: larger loads

The measured concurrency curve on Qwen3-0.6B is the argument for where to go
next, so it is worth restating as marginal return rather than as totals:

| Cap step | Throughput | Ratio | Per unit of cap | p99 TTFT |
|---|---|---|---|---|
| 1 → 8 | 537 → 3631 | 6.8× | 0.85 | 210 → 194 ms |
| 8 → 32 | 3631 → 12770 | 3.5× | 0.88 | 194 → 315 ms |
| 32 → 64 | 12770 → 20125 | 1.58× | 0.79 | 315 → 426 ms |

Efficiency per unit of cap is flat to 32 and then breaks — and the last
doubling bought 58% more throughput for 35% more p99 TTFT. That is a curve
bending, not a curve flat, so 64 is past the efficient point without being the
ceiling. Nothing on this node has bracketed the ceiling.

`recipes/tokenspeed/tokenspeed-serve-load-high.yaml` brackets it with two more
doublings, 128 and 256. Two rather than one: a single point above 64 can only
say "still rising", whereas two separate a curve that is still paying for cap
from one that has stopped. The prediction worth writing down before the run, so
it can be wrong: if 128 returns materially less than the 0.79 per unit that 64
managed, the knee is behind us and 256 says how far.

Three constraints the recipe respects, each of which would otherwise make the
rows unreadable:

- **`num_prompts` scales with the cap**, at the 4× ratio the existing recipe
  uses at 8, 32 and 64. `tokenspeed-serving.md` is explicit that requests which
  drain before the server reaches steady state measure the ramp rather than the
  throughput, and 256 prompts against a cap of 256 would be exactly that. It is
  the *ratio* that is held fixed, not the count, because that is what keeps
  these rows comparable with the ones below them.
- **`conc-64` is repeated rather than borrowed.** The existing recipe's
  conc-64 was measured in a different cell neighbourhood against a different
  server instance, and every ratio above it depends on that denominator. An
  anchor measured in the same run is what makes the two tables joinable; if the
  two conc-64 rows disagree, run-to-run spread is the story and the ratios are
  not yet evidence.
- **It is a separate recipe, not extra cells in the existing one.** The
  existing recipe's six-cell table in `tokenspeed-serving.md` is described as
  measured "exactly as committed". Adding cells to it would make that
  description false and force a re-measure of all six to keep the document
  honest, for no gain — the anchor cell gives the join at a cost of one cell.

### What it measured

All three cells passed, all 5376 requests served, none failed.

| Cell | Cap | Prompts/step | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | Output tok/s | Requests/s |
|---|---|---|---|---|---|---|---|
| conc-64 | 64 | 256 | 88.8 | 977 | 2.59 | 14996 | 117.2 |
| conc-128 | 128 | 512 | 113.0 | 1359 | 3.52 | 18022 | 140.8 |
| conc-256 | 256 | 1024 | 916.7 | 2032 | 3.48 | 20602 | 161.0 |

**The knee is behind us**, which is the prediction above coming out the way it
was written to be falsifiable. Within this run, throughput returns 1.20× for
the 64 → 128 doubling and 1.14× for 128 → 256 — 0.60 and 0.57 per unit of cap,
against the 0.79 the 32 → 64 step managed. Efficiency per unit of cap has
dropped by roughly a quarter and is still falling.

The latency side is where it becomes decisive. Median TTFT is flat from 64 to
128 (88.8 → 113 ms) and then rises tenfold at 256 (916.7 ms), while p99 TTFT
roughly doubles across the same two steps. So the last doubling bought 14% more
throughput for an 8× worse median time to first token. **128 is the end of the
useful range for this model and shape on this node**; 256 is over the cliff and
is worth keeping in the recipe only as the row that shows where the cliff is.

One result the anchor cell exists to catch, and did: `conc-64` reports 14996
tok/s here against the 20125 the existing load recipe recorded for an
identical configuration — 25% lower. Every ratio quoted above is therefore
computed within this run. The section below measures what that gap is.

### What the 25% gap actually is

Measured, on one node in one allocation, with the `conc-64` cell byte-identical
to the way both committed recipes declare it. Two runs: three repeats at the
committed three measured steps, then three fresh server instances at twelve
measured steps each, for 36 step samples.

The cell aggregate is *stable*. Three repeats of the three-step cell returned
17731 / 18062 / 18348 tok/s — 1.7% CV, a 3.5% spread. Taken alone that would
say the 25% gap is not run-to-run noise at all, and it would be the wrong
conclusion, because the per-step breakdown is not unimodal.

**A single bench step is bimodal.** Across the 36 twelve-step samples:

| Population | n | Mean tok/s | CV | Range |
|---|---|---|---|---|
| clean | 33 | 20200 | 1.13% | 19594 – 20530 |
| stalled | 3 | 12897 | 0.25% | 12857 – 12921 |

A step is either clean or it is 36% slower, with nothing in between, at a
measured rate of 3 in 36 (8%). The three stalls all landed in one of the three
server instances (steps 3, 10 and 11 of `inst-1`); the other two instances had
none in twelve steps each, so the propensity looks like a property of a server
instance rather than an independent per-step coin flip.

That reproduces both published numbers without either being wrong. A rolling
three-step mean over this population spans 15368 to 20511 — a **33% spread**,
which contains the 25% gap comfortably. Three clean draws average 20.3k, which
is the 20125 in `tokenspeed-serve-load.yaml`. Two stalls out of three average
about 15.3k, which is the 14996 here. **The two tables differ by how many
stalls each happened to draw, not by node, recipe, cell ordering or cache
warmth.** Cell ordering was the leading hypothesis before this run and it is
ruled out: the stalls did not prefer the first step, or the first cell.

What the stall is, as far as the exports show. Its signature is specific and it
is the same every time:

| | clean step | stalled step |
|---|---|---|
| duration | 1.60 – 1.62 s | 2.54 – 2.55 s |
| p50 TTFT | 76.7 – 79.9 ms | 77.6 – 84.9 ms |
| p90 TTFT | 80.2 – 84.0 ms | 1001 – 1015 ms |
| p50 TPOT | 2.43 – 2.47 ms | 2.44 ms |
| p99 ITL | 4.02 – 5.49 ms | 4.05 – 4.31 ms |

Decode is untouched — TPOT and ITL are identical to three significant figures.
So is median TTFT. What moves is the *tail* of time-to-first-token, by almost
exactly 0.93 s, and the step duration grows by almost exactly the same 0.93 s.
So the tail of the batch waits about a second to be admitted and then runs at
full speed. The magnitude repeats to within 1.4% across all three occurrences,
which argues for a fixed-duration blocking event in the prefill or scheduling
path rather than contention, which would vary. One of the three also admitted
128 concurrent requests instead of the usual 192. Nothing in the server log
marks the event. Attributing it needs server-side profiling and is not done
here; it is filed under Not done below.

**What this means for quoting these numbers.** Clean steady-state serving
throughput on this cell reproduces to 1.13% CV, which is in line with the
0.15–2.63% the sibling gating work measured across byte-identical
configurations, and the two TP recipes above reproduce to 0.06% across their
three steps. The harness is not noisy. What is unsafe is the *three-step mean*
on this particular cell: three samples from an 8%-bimodal population is too
small, and it is the sample size, not the measurement, that produced a 25%
discrepancy in a published table. Concurrency figures from this integration can
be quoted, with two conditions — quote them with the step count they were
measured at, and do not compare two three-step means as though the difference
between them were signal. Twelve steps costs about 20 extra seconds on a cell
whose bring-up is 300; there is no reason for a load cell to run three.

Not proposed: raising the cap past 256 in the same recipe, and adding a
`request_rate` axis. The first is unjustified until 128 and 256 say whether
there is anything left to find; the second changes the load *model* (open-loop
arrivals rather than a closed-loop cap) and belongs in its own recipe with its
own table, since a row from it would not be comparable to any row here.

## Workstream 1: more models

The existing five cover parameter count reasonably and architecture badly.
gpt-oss-20b is simultaneously the only MoE model, the only quantised model and
the only non-Qwen model, so **MoE and MXFP4 are perfectly confounded**: any
difference between it and the Qwen3 dense line could be either, and nothing in
the matrix can separate them. That, not the missing parameter count, is the
gap worth spending a node on.

In priority order, with what testing found:

| Model | Params | Buys | Gated | Outcome |
|---|---|---|---|---|
| `Qwen/Qwen3-32B` | 32B dense | extends the dense curve 4× past 8B, same family so one variable moves | no | **committed**, passes |
| `Qwen/Qwen3-30B-A3B` | 30B total / 3B active | **MoE without MXFP4** — breaks the confound above | no | **committed, eager only** — cannot come up with CUDA graph capture, see below |
| `meta-llama/Llama-3.1-8B-Instruct` | 8B dense | first gated repo; non-Qwen architecture at a size the matrix already has | **yes** | **not committed** — no token available here |

### The MoE model does not come up, and that is itself the answer

`Qwen3-30B-A3B` was the highest-priority addition, because paired with
`Qwen3-32B` it is as close to a controlled MoE-versus-dense comparison as the
matrix can get. It fails during CUDA graph capture on this image:

```
tokenspeed_kernel_amd/ops/moe/gluon_bf16_moe/moe_align_device.py:268
  em, num_blocks = (int(x) for x in meta.tolist())  # single host sync
RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph
capture unless the CPU tensor is pinned.
```

`moe_align_block_size_device` reads a device tensor back to the host — the
comment calls it a single host sync — and torch forbids that inside capture
unless the destination is pinned. It is in the shared `gluon_bf16_moe` path, so
it is not specific to this checkpoint: **any BF16 MoE model hits it.**

That explains why the confound existed in the first place. gpt-oss-20b is the
only MoE in the matrix not because nobody added another but because the MXFP4
MoE path works and the BF16 MoE path does not. The gap was a symptom.

`--enforce-eager` skips capture and the model then serves normally — 32/32
requests, none failed. That is a usable workaround, and it is what
`tokenspeed-serve-moe-vs-dense.yaml` uses, on **both** cells: a comparison is
only meaningful if both sides run in the same execution mode, and eager decode
is several times slower than captured, so one eager cell beside one captured
cell would measure the mode rather than the architecture. The cost of that
choice is that the recipe's rows are comparable to each other and to nothing
else in the directory. The 32B dense cell appears in both that recipe and
`tokenspeed-serve-models-large.yaml` so the size of the eager penalty is
measurable rather than assumed.

The upstream fix is small — pin the metadata tensor, which is what the error
message asks for — and it is the only thing standing between the matrix and a
BF16 MoE cell that runs like every other cell. Filed as
[lightseekorg/tokenspeed#1329](https://github.com/lightseekorg/tokenspeed/issues/1329).

### What the model cells measured

`tokenspeed-serve-models-large.yaml`, captured, both cells passing 96/96:

| Cell | Startup (s) | TTFT p50 (ms) | TPOT p50 (ms) | Output tok/s |
|---|---|---|---|---|
| Qwen3-8B (anchor) | 316 | 60.6 | 4.69 | 1557.9 |
| Qwen3-32B | 195 | 100.2 | 15.36 | 499.4 |

The anchor agrees with the existing model sweep to within 1% on TTFT, 5% on
TPOT and 4% on throughput (61.1 / 4.46 / 1625 there). That is worth stating
beside the load recipe's anchor, which came out 25% low: repeated cells agree
here and did not there. Both anchors were cheap and one of them changed a
conclusion, which is the argument for keeping them — and the follow-up above
explains why the two behaved differently. Model cells run at concurrency 8 and
their steps do not stall; the load cell runs at 64 and 8% of its steps do. The
harness reproduces in both cases. What differs is whether three steps is enough
to average the cell.

The 32B row here is 499.4 tok/s against the 541.1 the TP=1 cell of
`tokenspeed-serve-tp-large.yaml` later measured for the same model and load. The
difference is the KVStore bound — this recipe leaves the host tier at its
default and that one caps it at 128 GB per rank — plus a different node, and
neither row is a control for the other. If the two need to be joinable, the
bound has to match.

4× the parameters costs 3.3× the per-token time and 3.1× the throughput —
close to linear, which is the expected shape for a dense model where decode is
memory-bandwidth bound.

`tokenspeed-serve-moe-vs-dense.yaml`, both cells eager, both passing 96/96:

| Cell | Startup (s) | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | Output tok/s |
|---|---|---|---|---|---|
| Qwen3-30B-A3B (MoE) | 198 | 138.1 | 149.2 | 24.56 | 314.6 |
| Qwen3-32B (dense) | 202 | 128.7 | 2517.0 | 27.22 | 286.5 |

**The MoE is 9.8% faster per token than the dense model of the same total
size**, not the order of magnitude its 3B active parameters might suggest. The
honest read is that eager execution is launch-bound rather than FLOP-bound, so
it compresses exactly the advantage an MoE has; this comparison establishes
that the two architectures can be measured against each other, and puts a
floor under the MoE's advantage, but the number to quote will come from the
captured run once the upstream sync is fixed.

The tail is the more interesting column. Dense p99 TTFT is 2517 ms against the
MoE's 149 ms — 17× — while the medians are within 8% of each other. A dense
32B forward blocks the queue for long enough that a request arriving behind one
waits; the MoE's smaller active set does not. That is a real architectural
difference and it survives being measured in eager mode.

Comparing the 32B dense cell across the two recipes prices the eager penalty
directly, which is why it appears in both: 499.4 → 286.5 tok/s and 15.36 →
27.22 ms TPOT, so capture is worth about 1.75× on this model. That is the
number to weigh when reading any eager row.

**The gated model is deliberately not committed.** `hf_token_env` forwards a
token by name and the path is well tested in unit tests, but no committed
recipe has ever exercised it against a real gated repo, and no `HF_TOKEN` is
present on this machine or in the HF cache, so it could not be validated here.
A recipe that cannot come up should not be committed — the same rule that kept
TP=4 out — so it stays in this table. Running it needs nothing but the token
exported in the shell that launches the sweep and the cache pre-warmed under
that token.

**FP8 is not proposed yet.** It is a real gap — there is no FP8 checkpoint in
the matrix and no quantised/unquantised pair of the same model — but whether
TokenSpeed loads a given FP8 checkpoint on gfx950 is unverified, and proposing
a recipe whose model may not load is the failure mode this section is trying to
avoid. The cheap way to find out is the [serving
probe](tokenspeed.md#serving-probe), which answers "does it come up" for the
price of one bring-up and no benchmark; do that before writing a recipe.

### Practical constraints these recipes have to respect

All of these are recorded in `tokenspeed-serving.md`; they are repeated because
each one is a way a new model cell fails for a reason that is not the model.

- **`ready_timeout_sec` must cover weight load *and* Triton compilation**, and
  startup is noisy — 189 to 319 s observed across repeats of one recipe, and
  283 to 379 s across the model sweep, with the *smallest* model posting the
  largest number. It is not a quantity to tune close to an observation. The
  proposed recipe uses 2400 s, matching the gpt-oss recipes rather than the
  1200 s the Qwen3 sweep uses, because a 30B checkpoint is a much larger read.
- **The HF cache must be pre-warmed as the uid the trial runs as.**
  `run_as_current_user` defaults to true, and a cache populated by a root
  container leaves the run failing with `PermissionError` on the snapshot
  directory. Measured here: 174 GB across gpt-oss-20b and the three Qwen3
  models, fetched in about 220 s.
- **The cache is per-uid by default**, at `<work_dir>/u<uid>/hf`. `hf_home` is
  how a pre-warmed cache is shared, and sharing has to be deliberate — an
  administrator-populated directory, read-only by intent.
- **The per-uid scratch directory must not be group- or world-writable.**
  Pre-warming with a default umask creates it `0775`, and the workload then
  refuses the trial: *"Exports written there can be replaced between the run and
  the audit that reads them."* This cost a cycle here. Create it `0700`.
- **Take the full snapshot.** For gpt-oss-20b, excluding `original/` fails
  during weight loading rather than at download time.

## Cost

Node-hours per recipe, for scheduling. A serving cell is dominated by
bring-up, not by the benchmark: the measured TP=4 cell spent 180 s becoming
ready and about 9 s benchmarking. So cell count, not load size, is what these
cost — `conc-256` moves four times the requests of `conc-64` for about twenty
extra seconds.

Every figure below is measured wall clock from the runs described above, not an
estimate.

| Recipe | Cells | Node-hours | Notes |
|---|---|---|---|
| `tokenspeed-serve-gptoss-tp-wide.yaml` | 3 | **0.39** | 0.16 with a warm Gluon cache; see below |
| `tokenspeed-serve-load-high.yaml` | 3 | **0.31** | Qwen3-0.6B bring-up is the cost; the bench adds under 30 s/cell |
| `tokenspeed-serve-models-large.yaml` | 2 | **0.19** | 32B weight load is 65 GB |
| `tokenspeed-serve-moe-vs-dense.yaml` | 2 | **0.19** | eager, so no capture time, but slower steps |
| `tokenspeed-serve-tp-large.yaml` | 4 | **0.24** | 848 s measured end to end; bring-up 114-117 s per cell against a warm cache |
| **widened matrix, incremental** | 14 | **~1.3** | on top of the existing recipes |

`tokenspeed-serve-tp-large.yaml` is the cheapest cell-for-cell of the five,
which is worth noting because it is the one that needs the whole node: its
Qwen3-32B bring-up was 114-117 s in every cell including TP=8, against the
189-379 s the smaller models have posted. A 65 GB read from a warm page cache
beats a small model's Triton compilation.

The tp-wide figure needs its caveat: its first cell paid 939 s of Gluon
compilation against a cold cache while the other two, reusing it, became ready
in 133 s and 111 s. Run against a warm cache the recipe costs roughly 0.16
node-hours. **Cold-cache compilation, not model size, is the largest single
variable in scheduling this matrix** — it cost more on that one cell than the
entire load recipe's benchmarking.

Three costs paid once per node rather than per recipe:

- **Image pull**, 2 m 19 s by digest on a cold node.
- **HF cache pre-warm**, 174 GB across the four models used here, fetched in
  about 220 s.
- **Cold Gluon cache**, as above.

All three scale with the number of *nodes* the matrix is spread across, not
with the number of recipes, which is a strong argument for scheduling the
widened matrix as one allocation on one node. Doing so puts the whole thing at
roughly **1.3 node-hours** including the one-time costs, against about 2.0 if
each recipe is scheduled separately onto a cold node.

Add roughly 30–45 s per tensor-parallel cell for the post-teardown VRAM drain,
which `elapsed_sec` includes and `container_elapsed_sec` does not.

## Not done

- **What the ~0.93 s serving stall is.** Measured above at 8% of bench steps on
  the `conc-64` cell, with a signature specific enough to chase: fixed
  magnitude, tail-of-TTFT only, decode untouched, and nothing in the server
  log. Ruling it in or out needs server-side profiling across a run long enough
  to catch three or four of them — `rocprof` or `proton` against a
  twelve-step cell would do it. Until then, load cells should run twelve steps
  rather than three so the mean is not a sample of three from a bimodal
  population.
- **RCCL mitigations at wide TP.** Still untouched. TP=8 now runs, so the axis
  they would apply to exists; nothing has varied them.
- **TP=8 on a model that saturates above four cards.** Qwen3-32B peaks at TP=4,
  which means the TP=8 row here shows where saturation is rather than what
  eight cards buy. Answering the second question needs a model this node cannot
  hold on one card, and there is not one in the matrix.
- **A BF16 MoE model with CUDA graph capture.** Blocked on the upstream host
  sync in `moe_align_block_size_device`, filed as
  [tokenspeed#1329](https://github.com/lightseekorg/tokenspeed/issues/1329).
  Until that is fixed, every MoE number outside gpt-oss-20b is an eager number,
  and the eager penalty measured here is about 1.75× on throughput.
- **A captured MoE-versus-dense comparison.** Follows directly from the above.
  The eager one puts a floor under the MoE's advantage (9.8% on TPOT) and
  establishes the tail difference (17× on p99 TTFT), but the headline number
  should come from captured cells.
- **Confirming the per-rank host tier at TP=1 and TP=2 by allocation** rather
  than by applying the logged formula. A log grep on a default-sized run.
- **An upstream fix for the per-process host-memory guard**, which cannot see a
  per-node budget and so lets four ranks each approve an allocation that only
  fits once. Reported on
  [tokenspeed#297](https://github.com/lightseekorg/tokenspeed/issues/297); the
  thread is closed-stale and we lack the access to reopen it, so this may need
  a maintainer nudge or a fresh issue scoped to the flat executor.
- **The gated-repo path end to end.** No token available here.
- **FP8 anywhere.** Probe first.
- **Blessed baselines for any of this.** No serving recipe is gated in the
  nightly yet, so these recipes record trends and gate nothing.
- **Why crashed TP ranks keep their GPU memory.** Unchanged from
  `tokenspeed-serving.md`, and now known to be avoidable without root only by
  not causing it — which is what the bounded host tier achieves.
