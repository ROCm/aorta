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
| Tensor parallelism | 1, 2 | 4 did not come up | **4 works**, once the host KV tier is bounded |

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

All three cells passed, 96/96 requests served each, none failed. Load is the
same as the existing TP recipe (32 requests per step, ISL 512 / OSL 128,
concurrency 8, three measured steps) so the rows are comparable to it.

| Cell | Startup (s) | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | Output tok/s |
|---|---|---|---|---|---|
| tp1 | 939 | 68.1 | 69.4 | 7.98 | 948.9 |
| tp2 | 133 | 66.4 | 68.2 | 7.67 | 984.2 |
| tp4 | 111 | 67.8 | 69.6 | 7.36 | 1021.0 |

**TP=4 works, and it buys almost nothing.** 1021 tok/s against 948.9 at TP=1 is
7.6% more throughput for four times the hardware, and TP=2 already had 3.7% of
that. TTFT is flat across the axis to within 2.5%, which is the expected shape:
at 21B with MXFP4 the model fits comfortably in one MI355X, so extra ranks
relieve no constraint and pay collective cost on every step. The result worth
recording is that the path works and reports coherently at four ranks, not that
it is fast — and that a model this size is the wrong instrument for measuring TP
scaling, which is the argument for a larger model before TP=8.

Do not read the startup column as scaling with TP. tp1 ran first against a cold
Gluon cache and paid 939 s for it; tp2 and tp4 reused what it compiled. The
TP=4 cell in the earlier single-cell run, also against a cold cache, became
ready in 180 s. Startup here measures cache state, not rank count.

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

### TP=8, and what to try next

TP=8 is untested and is the obvious next cell: at `--kvstore-size 128` it asks
for 1024 GB node-wide, which fits. It is not in the committed recipe because
nothing here has run it, and the recipe rule in this repo is that a cell known
not to come up stays in the document rather than in the matrix. Two things make
it worth a deliberate experiment rather than an assumption:

1. **The collective changes shape.** TP=8 spans whatever the node's slowest
   link is, and the RCCL mitigations in aorta's registry — untouched by any TP
   cell so far — become the interesting axis rather than a formality.
2. **gpt-oss-20b may be too small to say anything.** TP=2 already buys only
   ~4.5% throughput because 21B at MXFP4 does not need a second card. At TP=8
   the collective cost per step is likely to dominate outright, and the honest
   read of a bad number would be "wrong model for this axis", not "the stack
   scales badly". A larger model — see the next section — is the precondition
   for TP=8 meaning anything.

Two further experiments, in the order worth doing them:

- **Confirm the per-rank constant directly at TP=1 and TP=2.** The 494 GB
  figure is measured at TP=4 and derived algebraically for the others. The
  sizing log prints `bytes_per_host_page` and `device_pool.size` on every rank,
  so one run of the committed recipe at default settings records all three
  points and turns the derivation into a measurement. Cheap: it is a log grep
  on a run that is happening anyway.
- **Report it upstream.** The per-process guard cannot see a per-node budget,
  and the message it fails to print is the one that would have made this a
  five-minute diagnosis. A rank that knew its world size could check
  `world_size × requested` against available memory and raise the ValueError it
  already has. That is a small upstream patch and it would prevent the stranded
  GPUs, which is the expensive part of the failure.

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
identical configuration — 25% lower. The two tables are therefore **not**
joinable as absolute numbers, and every ratio quoted above is computed within
this run for exactly that reason. What causes the gap is not established here;
the plausible candidates are node-to-node variation and cell ordering (the
existing recipe reaches conc-64 after three lighter cells, this one starts
there). Either way, an absolute serving number from this integration should be
read as node-and-run-specific until someone repeats it, which is a finding
about the harness rather than about the load curve.

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
message asks for — and worth reporting, because it is the only thing standing
between the matrix and a BF16 MoE cell that runs like every other cell.

### What the model cells measured

`tokenspeed-serve-models-large.yaml`, captured, both cells passing 96/96:

| Cell | Startup (s) | TTFT p50 (ms) | TPOT p50 (ms) | Output tok/s |
|---|---|---|---|---|
| Qwen3-8B (anchor) | 316 | 60.6 | 4.69 | 1557.9 |
| Qwen3-32B | 195 | 100.2 | 15.36 | 499.4 |

The anchor agrees with the existing model sweep to within 1% on TTFT, 5% on
TPOT and 4% on throughput (61.1 / 4.46 / 1625 there). That is worth stating
beside the load recipe's anchor, which came out 25% low: repeated cells agree
here and did not there, so the discrepancy is specific to that recipe or that
run rather than a general property of the harness. Both anchors were cheap and
one of them changed a conclusion, which is the argument for keeping them.

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
| **widened matrix, incremental** | 10 | **~1.1** | on top of the existing recipes |

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

- **TP=8.** Argued for above; needs a model large enough for the answer to mean
  something, so it is blocked behind the same thing the MoE comparison is. It
  was also not testable on this allocation for a more mundane reason: the
  failing TP=4 control stranded 251 GiB on two of the eight GPUs, and TP=8
  needs all of them. Run it first on a fresh allocation, before any experiment
  that can leave a rank dead.
- **A BF16 MoE model with CUDA graph capture.** Blocked on the upstream host
  sync in `moe_align_block_size_device`. Until that is fixed, every MoE number
  outside gpt-oss-20b is an eager number, and the eager penalty measured here
  is about 1.75× on throughput.
- **A captured MoE-versus-dense comparison.** Follows directly from the above.
  The eager one puts a floor under the MoE's advantage (9.8% on TPOT) and
  establishes the tail difference (17× on p99 TTFT), but the headline number
  should come from captured cells.
- **Confirming the per-rank host tier at TP=1 and TP=2 by allocation** rather
  than by applying the logged formula. A log grep on a default-sized run.
- **An upstream report on the per-process host-memory guard**, which cannot see
  a per-node budget and so lets four ranks each approve an allocation that only
  fits once.
- **The gated-repo path end to end.** No token available here.
- **FP8 anywhere.** Probe first.
- **Blessed baselines for any of this.** No serving recipe is gated in the
  nightly yet, so these recipes record trends and gate nothing.
- **Why crashed TP ranks keep their GPU memory.** Unchanged from
  `tokenspeed-serving.md`, and now known to be avoidable without root only by
  not causing it — which is what the bounded host tier achieves.
