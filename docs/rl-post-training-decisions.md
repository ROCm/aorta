# RL post-training: framework, model and topology — the decisions

For Manoj. On 3 September you asked for one thing to be made concrete — "are you
using verl? Are you using slime? … just make that decision at some point so that
you have it concrete and then you can put it down into a design doc" — plus two
constraints attached in the same conversation: it has to post-train meaningfully
on **a single node**, and the finished model has to be servable as a **local
endpoint** that aorta can point at instead of the AMD LLM gateway.

This document decides those and asks for what I cannot decide. The reasoning,
the prior measurement and the reward design are in
[tokenspeed-rl-post-training.md](tokenspeed-rl-post-training.md); nothing here
repeats them beyond what a decision needs.

## The decisions

| | Decision | What would change it |
|---|---|---|
| 1 | **Trainer: slime**, not verl | Dropping the requirement that aorta owns the engine process. verl is the better tool if the trainer is allowed to launch and own SGLang itself |
| 2 | **Algorithm: GRPO** | Nothing in sight. No value network is the point, and it is slime's default |
| 3 | **Base model: Qwen3-8B**, not GLM-5.3-Flash | Deciding that a LoRA adapter counts as the deliverable, or a second node. Full-parameter GLM-5.3-Flash on one node is arithmetically impossible, not merely tight |
| 4 | **Topology: one node, all 8 GPUs, colocated with engine sleep/wake**; disaggregated 6 + 2 as the fallback | One measurement, described below. Both fit Qwen3-8B with room |
| 5 | **The trainer lives in a separate repo that imports aorta as a library** | Nothing. Carried forward from the plan; recorded here because it is a decision you should see rather than one to revisit |

Decision 5 is the one that keeps aorta clean: the reward functions call
`aorta.agent`, `aorta.probe.classifier` and `aorta.registry`, so they cannot
drift from the contract, and aorta never grows a dependency on a training
framework.

## What I need from you, in blocking order

1. **Confirm the base model, given the arithmetic below.** GLM-5.3-Flash is a
   real model and we can serve it, but full-parameter post-training of it needs
   2.5 nodes' worth of HBM. The fallback condition you set — "if it is running
   out of memory, if it's too big" — is met on paper, before a node-hour is
   spent. I want your agreement to start on Qwen3-8B rather than discovering
   this by OOM.
2. **Extend the eight failure categories, or tell me who owns them.** They
   contain no numerics slot and no nondeterminism slot, so two of the six
   prioritised use cases classify as `unknown` and cannot be trained at any
   corpus size. Your own flagship example — "numerical instability from training
   coming from … race conditions introduced in TF32 kernels" — is exactly that
   shape. This blocks corpus work, which is the critical path.
3. **Who stands up the trainer.** No trainer exists here, and slime on MI355X is
   a real bring-up, not an afternoon. If it is me, it displaces the corpus work
   in item 2.
4. **Are trajectories in scope for the first run, or single steps?** You
   described the signal as "all the trajectories … and all the decision trees".
   The corpus that exists is single-step. See [Corpus shape](#corpus-shape-the-gap-your-description-opens).
5. **Does the CIA chatbot render structured fields or free text?** Structured
   means the reward stays fully automatic and nothing changes. Free text means a
   presentation layer nobody has scoped. Not blocking; cheapest to answer now.

## 1. Framework: slime

**The single strongest reason: TokenSpeed's SGLang-compatibility layer was
written against slime, by name.** This is not inference from a feature list. It
is in the source of the engine we are committed to:

- `runtime/entrypoints/sglang_compat_http.py`, module docstring: *"The endpoint
  names and JSON fields match the surface used by trainers such as slime."*
- `/destroy_weights_update_group`: *"Body is optional: trainers that always call
  destroy (e.g. slime) may send only `{group_name}` or nothing at all."*
- `/generate` on the control server: *"Slime uses SGLang's `sampling_seed` name;
  TokenSpeed's native parameter is `seed`"* — an explicit translation shim.
- The response unwrapper: *"slime/verl index `output["meta_info"]`, so unwrap a
  1-element list for single prompts."*

Choosing verl means being the first to drive that surface with something it was
not shaped around. Choosing slime means the compatibility work is already done
and, where it is imperfect, upstream has a reason to care.

**The second reason is architectural, and it is the one that would decide this
even without the first.** slime has a first-class mode for attaching to an
SGLang engine it did not launch — `--rollout-external-engine-addrs`, documented
under *External Rollout Engines* — for exactly the case where "another system
deploys and owns the engine lifecycle". aorta's `tokenspeed_serve` **is** that
other system: it stands the engine up, supervises it, waits for VRAM release,
cleans up the container and audits the run, and section 1 of the plan makes
owning that aorta's job. verl always owns its engine. Both of its SGLang paths
launch it — in-process for colocated mode, or as a Ray actor for server mode,
after which it addresses its own server over HTTP (`launch_server=False` in
`sglang_rollout.py` is verl talking to a server verl started). verl also imports
`sglang.srt` directly, so an engine that merely *speaks* SGLang's dialect is not
a drop-in for it. Using verl here means either duplicating `tokenspeed_serve` or
bypassing it.

Point by point, on the axes that matter:

| | slime | verl |
|---|---|---|
| GRPO | Default advantage estimator (`default="grpo"`), alongside GSPO and REINFORCE++ | First-class, 45 GRPO example scripts, more documentation |
| Attaches to an engine it did not launch | **Yes**, `--rollout-external-engine-addrs`, discovers topology from `/server_info` or `/get_server_info` | No such configuration |
| Endpoints it drives over HTTP | `/generate`, `/pause_generation`, `/continue_generation`, `/flush_cache`, `/update_weights_from_disk` — **TokenSpeed implements all five** | Same names via its adapter, but only against its own engine |
| Sleep/wake for colocation | `/release_memory_occupation`, `/resume_memory_occupation` with `weights` / `kv_cache` tags | Same endpoints, same tags |
| Weight transport without an NCCL group | **Yes** — `--update-weight-transport disk`, plus a delta variant | Checkpoint-engine work exists but assumes verl's own engine |
| Logprobs | Recomputes with the actor by default (`--use-rollout-logprobs` defaults off) | Recomputes by default |
| Checkpointing | Megatron checkpoints plus an HF exporter, and a ROCm-specific writer | FSDP and Megatron managers, HF export, resumable |
| Training backend | **Megatron-LM only** (`--train-backend` accepts `megatron` and nothing else) | FSDP *or* Megatron; FSDP eats HF checkpoints directly |
| AMD | `Dockerfile.rocm_MI350-5`, an AMD tutorial, `rocm_checkpoint_writer.py`, a Qwen3-4B AMD script | An AMD tutorial, ROCm CI |
| GLM family | Written by the GLM authors; GLM-5 and GLM-5.2 recipes in-tree | A GLM-4.1V example |

**The honest cost of choosing slime is Megatron.** verl's FSDP backend would
take a Hugging Face Qwen3-8B checkpoint and start; slime needs an HF→Megatron
conversion, a TP/PP/EP configuration, and a heavier container. For an 8B dense
model that is a known quantity rather than research — slime ships an AMD script
for Qwen3-4B that is the same shape — but it is the largest single line item in
the bring-up, and I would rather name it than discover it.

**What would change this recommendation:** if you would rather the trainer own
the engine outright and treat `tokenspeed_serve` as a benchmarking tool only,
verl is the better choice and the Megatron cost disappears. That is a real
option; it trades the seam in section 1 of the plan for a lower bring-up cost. I
recommend against it because the seam is what keeps the engine bring-up,
supervision and audit in the place that already does all three.

**One configuration detail that will silently produce wrong gradients if
missed.** TokenSpeed's `/generate` synthesizes placeholder logprobs of `0.0`
when the engine was not started with `--enable-output-logprobs` — the field is
present, the values are fake, and the source comment says so: the placeholder is
safe *"only when the trainer recomputes logprobs"*. slime's default recomputes
with the actor, so the default combination is safe. Setting
`--use-rollout-logprobs` without `--enable-output-logprobs` would compute
importance ratios from zeros and report nothing wrong. Both flags go in the
run's configuration with a comment, and the smoke test asserts a non-zero
logprob.

One adapter-sized gap, found by reading both sides rather than by running them.
slime's external-engine discovery reads the engine's parallel topology from
`/get_server_info`, looking for `tp_size` or `tensor_parallel_size`; TokenSpeed
returns its server arguments, which carry `pipeline_parallel_size` flat but keep
tensor-parallel width under a nested `mapping`. slime would therefore infer
TP=1. Harmless at TP=1, which is what Qwen3-8B wants, and wrong above it. It is
a few lines upstream or a few lines of shim, and it is the kind of thing worth
finding before it presents as a mysterious sharding error.

## 2. Single node: yes, and the reason it looked worse than it is

**The blocker the plan records has moved, and I would have missed this by not
re-reading upstream.** [tokenspeed#1373](https://github.com/lightseekorg/tokenspeed/issues/1373)
— NCCL weight transfer returning 200 while transferring nothing — is filed
against `runtime/engine/weight_transfer/`. Upstream **deleted that package** in
[#1183](https://github.com/lightseekorg/tokenspeed/pull/1183), merged
2026-08-21, under the title *"remove unused RL weight-transfer APIs, consolidate
on a single control plane"*. We filed on 2026-09-02, against code that had been
gone for twelve days, because we measured the pinned image
(`sha256:60c12e37…`, from the `nightly-20260714` tag) rather than main. The
issue is still worth leaving open — it is a real defect in a shipped build — but
it is no longer the thing standing between us and a closed loop.

**Two corrections follow, and both are in our favour.**

The `ipc` backend raising `NotImplementedError` does **not** block colocation.
Colocation on SGLang does not go through a bespoke IPC weight transport; it goes
through sleep/wake — `/release_memory_occupation` frees the engine's weights and
KV cache before the training step, `/resume_memory_occupation` brings them back
after. TokenSpeed implements both for real, over a torch_memory_saver data plane
with independent `weights` and `kv_cache` tags, and has done since 2026-06-15,
which puts them **in the image we already pin**. The plan concluded "no `ipc`,
therefore no colocation"; those are two different mechanisms and only one of
them is missing.

And there is a weight path that needs no collective at all.
`/update_weights_from_disk` is implemented, and slime drives it as
`--update-weight-transport disk`: the trainer writes an HF checkpoint to a
shared path, the engine hot-reloads it without restarting. For Qwen3-8B that is
a 16 GB write per sync, against the 189–319 s cold start it avoids. It is the
slowest of the three transports and the only one that cannot fail silently.

So the topology question is now a choice rather than a constraint:

| Topology | GPUs generating | Weight path | What has to be true |
|---|---|---|---|
| **A — colocated, 8 shared** | 8 | `/update_weights_from_tensor`, or disk | Sleep/wake works on the image we run; trainer and engine take turns on all 8 GPUs |
| **B — disaggregated 6 + 2** | 6 | `/init_weights_update_group` + `/update_weights_from_distributed` | The SGLang-dialect broadcast works. On main the receive loop in `model_runner.py` posts a real `dist.broadcast`, unlike the deleted path |
| **C — disaggregated 6 + 2, disk transport** | 6 | `/update_weights_from_disk` | Only a shared filesystem. No NCCL group, nothing to fail silently |

A is fastest and B is the plan's existing recommendation; **C is the one that
cannot be blocked**, which makes it the right thing to bring up first and the
right fallback to keep. Generation cost across these is the plan's existing
arithmetic: roughly 2.5 hours across 8 GPUs for a 400-iteration run at the
extrapolated 8B rate, 3.3 hours across 6.

**The one measurement that settles A versus B**, and it is cheap: drive
`init_weights_update_group → update_weights_from_distributed` and the sleep/wake
pair against a live engine, with a **round trip** — push modified weights, push
the originals back, assert the completion returns to baseline. A perturbation
check passes on a broken transport, which is how the last defect hid;
`examples/rl/nccl_roundtrip_check.py` already implements the round-trip
methodology and needs only to be repointed at the SGLang-dialect routes. Half a
day on one node.

That measurement needs a newer engine image than the one we pin, and the pin
matters right now: `tokenspeed-serve-bench-smoke.yaml` is the nightly gating
entry and its digest is deliberately frozen for the baseline window. **Bump the
digest in the RL rollout recipes only and leave the gated recipe alone** until
that window closes.

### Memory arithmetic

Full-parameter GRPO costs about **18 bytes per parameter**: policy in bf16 (2) +
gradients in bf16 (2) + Adam's fp32 moments (8) + an fp32 master copy (4) + the
frozen reference policy in bf16 (2). No value network — that is what GRPO buys,
and it is worth about 16 bytes per parameter, since a critic carries its own
weights, gradients and optimiser state. Activations for 512-token sequences add
20–40 GB on top. One node is 8 × MI355X at 288 GB = **2,304 GB**.

| Model | Params | Weights bf16 | Weights fp8 | Trainer, full-parameter | Share of one node |
|---|---|---|---|---|---|
| **Qwen3-8B** | 8.19 B | 16 GB | — | **147 GB** (+20–40 GB activations) | **7%** |
| GLM-4.7-Flash | 31.2 B | 62 GB | — | 562 GB | 24% |
| **GLM-5.3-Flash** | 321.3 B | 643 GB | 321 GB | **5,784 GB** | **251%** |
| GLM-5.3 | 753.3 B | 1,507 GB | 753 GB | 13,560 GB | 589% |

GLM-4.7-Flash is in the table because it is the only GLM whose arithmetic works
on one node, not because it is an option: TokenSpeed ships model code for GLM-5
and GLM-5.3-Flash and none for the GLM-4 family, so we could not serve it.

The plan's 165–185 GB for the trainer checks out: 147 GB plus activations, and
it does fit on a single 288 GB GPU on paper. Two GPUs remains the right
recommendation — FSDP across a pair rather than betting a multi-hour run on a
single-device fit — but under topology A the trainer shards across all 8 anyway
and the question does not arise.

**Qwen3-8B is not close to the limit in any topology.** The engine wants 16 GB
of weights plus 30–60 GB of KV cache per data-parallel replica; the trainer
wants 187 GB at the top of the range. The binding constraint on this run is the
correctness of the weight path, not memory.

## 3. Base model: Qwen3-8B, and what happened to the GLM one

**"GLM 5.3 flash" resolves cleanly, and to something much larger than the name
suggests.** It is `zai-org/GLM-5.3-Flash`, published 2026-08-25, nine days
before our conversation. Three facts about it, in the order that matters:

**We can serve it.** TokenSpeed added day-0 support on 2026-09-01
([#1259](https://github.com/lightseekorg/tokenspeed/pull/1259),
`runtime/models/glm53_flash.py`). Not on our pinned image, which predates both
the model and the support by six weeks, so serving it is a real task — a newer
image, and a validation pass, since our recipes have only ever exercised Qwen3
0.6B/1.7B/4B/8B and gpt-oss-20b.

**We cannot post-train it on one node.** It is a 321-billion-parameter
mixture-of-experts model — 288 routed experts, 8 active per token, shipped in
FP8, and multimodal, with a vision tower we have no use for — carrying roughly
**16 billion active parameters**. "Flash" describes what it
costs to *run*, not what it costs to *hold*. Its serving compute looks like a
16B model; its storage looks like a 321B model, because every expert has to be
resident. RL post-training memory scales with total parameters, not active ones:
the optimiser carries state for every expert whether or not the router picked
it. **5,784 GB against 2,304 GB of node.** That is 2.5 nodes for the trainer
alone, before the engine gets a byte.

**So the fallback condition you set is already met.** You said to switch down to
Qwen3-8B "if it is running out of memory, if it's too big". It is, and we can
know that from the config file rather than from a crash at hour three.

**Recommendation: Qwen3-8B is the base model for the first run.** It is in the
validated set, it clears the format gate, the memory is a rounding error against
the node, and every throughput number we have is measured on it or on its
smaller siblings.

**If you want GLM-5.3-Flash specifically, there are two honest routes**, and I
would want your steer before spending on either:

- **LoRA rather than full-parameter.** A frozen bf16 base is 643 GB across 8
  GPUs — 80 GB per device — with adapter states in the single-digit GB, and the
  engine's FP8 copy alongside at 321 GB. Roughly 1,000 GB of 2,304, so it fits.
  But it is unvalidated on every axis at once: MoE LoRA, on a hybrid
  architecture six weeks old, on ROCm, in a trainer with no recipe for it. And
  "we post-trained the model" becomes "we trained an adapter", which may or may
  not be the claim you want.
- **Serving-side comparison only.** Stand it up behind `tokenspeed_serve` and
  use it as the strong general-purpose baseline that the post-trained Qwen3-8B
  has to beat. That is cheap, it is useful regardless, and it needs no training
  decision.

A note on the other name: "QUIN 3.827B" does not resolve to a real identifier
with confidence, but the later "switch down to Quinn 3.8" plus the presence of
Qwen3-8B in our validated set makes the fallback unambiguous. I have read it as
Qwen3-8B throughout.

### Why size is not only a cost question

The base model has to emit strict, well-formed JSON **before** training starts.
The proposal contract is the outer gate on both rewards, so a policy that cannot
produce a parseable object scores a flat 0.2 on nearly every sample — and when
every sample in a group scores the same, the group's advantage is zero and
nothing is learned. The run does not fail loudly; it fails by not moving.
Qwen3-8B clears this comfortably. Qwen3-0.6B is borderline, which is why it
belongs in engine measurement and not in the demo.

**A short supervised warm-up is cheap insurance, and it is also the honest
baseline.** Fine-tuning on a few hundred proposals guarantees the format gate is
cleared before RL starts. It also produces the number the RL half has to beat:
if supervised fine-tuning gets most of the way, the RL half has to justify
itself against that rather than against zero. I would rather find that out in
Phase 3 than at the end.

## 4. The local endpoint: mostly already built

You asked for the finished model to back aorta queries locally instead of the
AMD LLM gateway. **This is the cheapest of the three requirements, and it is
worth saying so explicitly, because it is real leverage between the serving
workstream and this one.**

`tokenspeed_serve` already stands up exactly this kind of server — that is the
workload the whole TokenSpeed effort has been building and measuring. And
pointing `aorta agent` at a self-hosted model needs **no code change**, only two
environment variables:

```bash
export OPENAI_API_BASE=http://<engine-host>:<port>/v1
export OPENAI_API_KEY=unused-but-must-be-set
aorta agent --llm-backend litellm --llm-model openai/<served-model-name> ...
```

This was verified against a mock OpenAI-compatible endpoint rather than assumed:
the request arrives at `POST /v1/chat/completions` on the self-hosted address
with `response_format: {"type": "json_object"}` intact, and the reply parses
back into an `AgentStep` unchanged.

**The one gotcha:** the `openai/` prefix is stripped on the wire, so the name
after the slash must match what the engine advertises, not what the CLI default
says. Get that wrong and the call fails in a way that reads like a model
problem.

Two smaller things, neither blocking. `OPENAI_API_BASE` is process-global, so a
hosted judge model and a self-hosted policy cannot coexist in one process — an
explicit `--llm-api-base` flag is about eight lines and would also put the
endpoint in the audit log. And the serving engine must accept
`response_format: {"type": "json_object"}`, which TokenSpeed's OpenAI route
does; it is a dependency of the integration rather than an optional extra.

## Corpus shape: the gap your description opens

You described the training signal as "all the trajectories that aorta has in
terms of debugging and all the decision trees that are included in general
debugging of aorta use cases" — **trajectories and decision trees**.

What exists is **nine scenarios yielding 54 single-step examples** (9 triage, 45
proposal), generated from the committed race reproducers, with ground truth
reproduced exactly. Each one is a single `evidence → proposal` pair. That is not
a trajectory, and the difference is not a matter of volume.

A trajectory corpus needs the agent loop run to completion and credit assigned
per step: which diagnostic was chosen, what came back, what was chosen next,
whether the search converged. That is a different generation path from
`build_corpus.py`, and it is a different reward — the existing scorers grade one
step against ground truth, and a trajectory has to be graded against an outcome
several steps later. It is a natural fit for GRPO, which scores whole sequences,
and the agent's own outcome labels (`converged`, `exhausted_candidates`) are a
ready-made terminal reward.

I have not built it and I would not start it before item 2 is answered, because
half the trajectories worth generating end in categories that do not exist yet.
**Which shape do you want for the first run?** Single-step is ready now;
trajectories are the thing you actually described.

## Blockers, stated plainly

| Blocker | Status | Who |
|---|---|---|
| **The eight failure categories have no numerics and no nondeterminism slot** | Two of the six prioritised use cases classify as `unknown` and are untrainable at any corpus size. Your TF32-race example is exactly that shape | **You**, or whoever owns the set |
| **No weight transport has been proven to move a tensor** | The path we measured was deleted upstream. The SGLang-dialect path looks correct in source and is unmeasured. Disk transport needs no collective | Me — half a day on one node |
| **The engine image is six weeks stale for this purpose** | Predates GLM-5.3-Flash support and the upstream consolidation. Bump the RL recipes only; the gating baseline window holds the smoke recipe's digest | Me, after the window |
| **The corpus is 54 single-step examples** | Generation costs ten minutes a sweep, so this is bounded by workload variety, not node-hours. Not one archived artifact comes from a workload the CIA diagram names | Me, once the category question lands |
| [aorta#449](https://github.com/ROCm/aorta/issues/449) | An unregistered mitigation name is dropped before validation, so a name-resolution failure records as a genuine `agent_stop`. Any reward computed from stop reasons is unsafe while it stands | Me |

## What already exists

Not to be re-planned — listed so the asks above have context.

- **Two reward graders, validated end to end against real data.** They import
  aorta rather than restating it — `AgentStep.from_dict`,
  `AgentPolicy.validate_step`, `get_mitigation`, the verdict resolver — so they
  cannot drift from the contract, and they inherit aorta's own tests. An oracle
  policy scores 1.0; a degenerate always-`pass` policy scores **0.5333**, printed
  beside every real score so no reward is ever read alone. (The 0.629 this
  section used to quote came from the rollout plan and reproduces nowhere — see
  [the end-to-end report](tokenspeed-rl-e2e-sanitizer-routing.md#on-the-0629-floor).
  The floor checks arithmetically as `0.6·(4/9) + 0.4·(6/9)`.)
- **A corpus builder** turning sanitizer runs into JSONL both scorers read with
  no conversion pass, and the 54-example corpus with committed provenance.
- **Rollout-shaped serving** in `tokenspeed_serve`: several sampled completions
  per prompt at temperature, stopping on EOS, reporting the generated-length
  distribution.
- **The measured case for the weight-transfer control plane**, which is the
  sharpest number in the whole effort: 400 iterations × ~250 s of cold start is
  **27.8 hours of loading weights** against ~2.5 hours of generation. Any loop
  that restarts the engine per iteration spends 91% of its life doing nothing.
  That, not throughput, is why the weight path is the thing to get right.

## Numbers I do not have

- **GLM-5.3-Flash throughput on MI355X.** No measurement exists and I have not
  estimated one. To get it: bump the engine digest in
  `tokenspeed-serve-rollout.yaml`, point it at the model, read
  `output_throughput` from `perf.md`. Roughly two hours on one node, and worth
  doing regardless of the training decision because it sizes the serving
  workstream.
- **Qwen3-8B rollout throughput at realistic concurrency.** The plan's 5,720
  tok/s is 1,625 tok/s measured at concurrency 8, scaled by a 3.52x factor
  measured on Qwen3-0.6B. Plan against the conservative 1,625 column until the
  rollout recipe is run at concurrency 32 on the 8B.
- **The cost of a disk-transport weight sync.** A 16 GB checkpoint write plus an
  engine-side reload, per iteration, measured rather than guessed. Falls out of
  the same half-day as the transport round trip.
