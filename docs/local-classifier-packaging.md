# Decision 22 — where the local classifier's weights may live, and which weights answered

This is the packaging decision for the local-classifier integration planned in
[`docs/plans/laya-system1-integration.md`](plans/laya-system1-integration.md).
The classifier is an implementation detail behind the `DecisionPredictor` seam;
the current implementation is the Laya model, which is what the plan adopted and
what the library-specific sections below describe. It
was written before any integration code existed, because both halves of it are
questions that get answered by accident if they are asked later: a dependency
lands in whichever extra was convenient, and a verdict gets recorded without the
identity of the thing that produced it.

The Phase 0 and Phase 1 offline tooling has since landed — the `[local-classifier]`
extra, the `python -m aorta.local_classifier` maintainer tooling, and
`src/aorta/local_classifier/` — and this record has been reconciled against
it, so what follows describes the code as well as prescribing it. The three
integration tracks are still unbuilt.

Two rules, and the reasoning for each.

1. **`torch` may be declared by the opt-in `[local-classifier]` extra and by nothing else.
   The chat path runs the same model through ONNX on the onnxruntime `fastembed`
   already installs. No loader is imported at module scope anywhere.**
2. **A checkpoint is pinned by digest, never by name or tag, and every report a
   local-classifier verdict reaches records which checkpoint, which library version, and
   which quantity the number in it actually is.**

## Rule 1 — the torch boundary

### What is actually enforced today

Decision 19a is usually described as "chat has no torch tier," which is true and
is only part of it. Four separate mechanisms enforce it, and they have different
scopes:

| Mechanism | Scope | What it catches |
| --- | --- | --- |
| the `Verify the embedding stack is torch-free` step in [`nightly.yml`](../.github/workflows/nightly.yml) and [`release.yml`](../.github/workflows/release.yml) | a resolved `.[chat-cli]` install | `torch`, any `nvidia-*`, `chromadb` or `sentence-transformers` arriving as a transitive dependency |
| `_HEAVY_PREFIXES` in [`tests/cli/test_chat_boundaries.py`](../tests/cli/test_chat_boundaries.py), via `test_importing_aorta_cli_does_not_pull_in_langchain` | the `import aorta.cli` graph | an import moved to module scope in `aorta/cli/chat.py` |
| the same tuple, via `test_importing_the_agent_proposer_does_not_pull_in_langchain` | the `import aorta.agent.llm` graph | an import moved to module scope in the probe agent's proposer module |
| `test_importing_the_seam_pulls_in_no_model_machinery` in [`tests/cia/test_local_classifier_predictor.py`](../tests/cia/test_local_classifier_predictor.py) | the `import aorta.local_classifier` graph, including `.eval`, `.gate`, `.corpus` and `.cli` | `torch`, `transformers`, `laya`, `onnxruntime`, `numpy` or `safetensors` reaching module scope in the seam itself |

The fourth row arrived with the Phase 0/1 tooling and is the one that will do the
most work, because it guards the package every integration track reaches through.
It carries its own tuple rather than importing `_HEAVY_PREFIXES`, which is the
right call: the two are enforcing the same rule for different reasons — the chat
boundary tuple is about what `aorta --help` costs to start, and this one is about
a model loader landing on the import path of every Watch poll. Note that
`numpy` and `safetensors` appear here and not in `_HEAVY_PREFIXES`; they are cheap
next to torch, and listing them anyway is what makes the probe fail on a partial
regression rather than only on a total one.

The corpus and eval tooling is deliberately not an `aorta` command. It is
maintainer tooling, reached as `python -m aorta.local_classifier`, so
`aorta --help` never reaches it and the second row has nothing to cover. Its
module, `aorta/local_classifier/cli.py`, is covered by the fourth row instead,
which imports it alongside the rest of the package, so an eager model import
there fails an existing test rather than needing a new one.

The CI gate is `[chat-cli]`-only. `[cia]` is installed in
[`cpu-tests.yml`](../.github/workflows/cpu-tests.yml) and
[`gpu-tests.yml`](../.github/workflows/gpu-tests.yml) with no equivalent
assertion, so the packaging half of this decision genuinely is asymmetric: a
CIA-side integration may resolve torch and a chat-side one may not.

The import half is not asymmetric, and that is the part that is easy to get
wrong. `_HEAVY_PREFIXES` lists `torch`, `onnxruntime` and `fastembed` next to the
langchain stack, and the second test aims it at `aorta.agent.llm` — which is
where `LocalClassifierProposer` goes. That test exists because
`aorta agent mitigate --llm-backend=fake` is the default path, is what the test
suite and `--dry-run` rely on, and must keep working on a base install of
`pip install amd-aorta`, which is `pyyaml` plus `click`. A module-scope
`import laya` in that file would make the offline default backend pay for torch
on every invocation and then fail outright where torch is absent. The existing
`ChatProviderProposer` already shows the shape: its chat seam lives inside
`_chat_model`, not at the top of the module.

### Why not simply add `laya` to `[cia]`

Because the comment above that extra in [`pyproject.toml`](../pyproject.toml)
says the opposite of what that would do. DSPy is there as the model surface Watch
and Autopsy reach through today, the shared chat provider layer is where that
belongs, and the extra "is expected to shrink rather than grow."

There is a harder reason too. The `cia_tests` lane in `cpu-tests.yml` runs the
extra across Python 3.10 through 3.14, and the comment above it says why: it
spans the full supported range *because the extra declares support for that same
range*, and the lane exists to stop a green run that installed no agent at all.
`laya` 0.3.5 declares `torch>=2.0.0`, `transformers>=4.48.0`, `safetensors`,
`huggingface_hub` and `numpy`, and torch wheels routinely lag a new CPython by
months. Putting `laya` in `[cia]` would mean the 3.14 leg of that matrix starts
failing at resolution time on a schedule nobody in this repository controls, for
a feature most `[cia]` users will never turn on. Add the English checkpoint at
~808 MB of weights and the "headless agents, 3.10 through 3.14, no chatbot
needed" property that `[cia]` was written to have is gone.

So the dependency goes in its own `[local-classifier]` extra, and
`pip install 'amd-aorta[cia]'` continues to mean what
[`docs/chat/installation.md`](chat/installation.md) says it means. It is also
kept out of `all`, which is what the dev and CI stacks install: `all` exists so a
contributor gets a working tree, and putting torch into every lint job to support
one offline measurement is not a trade worth making.

### The library version is itself a moving reference

When this section was written the extra declared `laya>=0.3.4` (it now floors at
`>=0.3.5`, for the calibration reason given in `pyproject.toml`), and that floor
was worth a paragraph because it did not describe what an install actually gets.

Verified against the PyPI JSON API, the PEP 658 `.metadata` sidecar on the wheel
itself, and the PEP 691 simple index on 2026-09-22: **0.3.5 is the latest
published release**, and the three of them agree. It declares
`Requires-Python: >=3.10` and `transformers>=4.48.0`. Its predecessor 0.3.4
declared `Requires-Python: >=3.8` and `transformers>=4.45.0`. So an open
`>=0.3.4` floor resolves to 0.3.5 today, and reading the floor as a description
of the dependency closure gets both of those numbers wrong.

That is not an argument for pinning the library exactly — a floor is the right
shape for a dependency we do not control, and aorta's own
`requires-python = ">=3.10"` makes 0.3.4's wider Python range irrelevant anyway.
It is an argument for not quoting the floor as though it were the resolved
version. Anything that states this package's transitive requirements has to state
which release it is describing.

The churn rate is the reason this is more than pedantry: 0.3.3, 0.3.4 and 0.3.5
were published on 19, 20 and 21 September 2026, and a `Requires-Python` bump and
a `transformers` floor bump both landed inside that three-day window. Two readers
checking "what does laya depend on" a day apart came back with different answers,
each correct for the release they saw. That is Rule 2's hazard arriving through
the library rather than through the weights, and it is why the identity a report
records is the library version *and* the checkpoint, not either alone.

### Why the chat path is ONNX and not a sidecar

Two options were considered for getting a torch-free local classifier into
`aorta chat`.

A **sidecar HTTP service**, addressed the way vLLM already is, keeps the tree
clean and costs nothing to package. It also puts a network round-trip back into
the path, and removing a network round-trip is half the reason this integration
exists. On the CIA side the same objection is sharper: a sidecar in Watch's poll
loop means the poll loop can now fail for a reason that has nothing to do with
the job it is watching.

An **ONNX export** onto the onnxruntime `fastembed` already installs costs
nothing at install time — the runtime is already there for
`BAAI/bge-small-en-v1.5` — and costs real work once: we own the export, including
the decision head, which is two transformer layers, an option-marker scorer and
an act/escalate head trained from scratch. `laya.load()` is unavailable without
torch, so there is no way to borrow the loader.

ONNX wins for the chat path, and it buys one more thing. The model card carries
this warning:

> If `laya.load()` hangs: `transformers` probes for TensorFlow at import, and
> when TF is installed its abseil runtime can deadlock model construction. Run
> with `USE_TF=0`.

A deadlock at model construction inside the Chainlit server is a hang with no
error and no log line — the UI simply never answers. Mitigating it by setting
`USE_TF=0` means depending on an environment variable being right on every host
the UI runs on, which is the kind of guarantee that holds until someone deploys
into a TensorFlow image. The ONNX path never imports `transformers` at all, so
the failure mode is not mitigated, it is absent.

### What this means per track

| Path | Extra | Runtime | Import rule |
| --- | --- | --- | --- |
| `python -m aorta.local_classifier` corpus and eval (Phases 0–1, landed) | `[local-classifier]` | torch | inside the method that loads a checkpoint; `aorta.local_classifier` has its own probe |
| Watch clean-gate (Track A) | `[local-classifier]` | torch | inside `LogWatcher.forward`'s tier, not at module scope |
| `LocalClassifierProposer` (Track B) | `[local-classifier]` | torch | inside `propose()`; `aorta.agent.llm` is import-gated by the boundary test |
| `LogFinder` tier 3 (Track C) | `[local-classifier]` | torch | inside the tier |
| Autopsy confidence (Phase 4) | `[local-classifier]` | torch | inside the router call |
| chat router / selector (Phase 5) | `[chat-cli]` | ONNX on the existing onnxruntime | inside the node, and `onnxruntime` is itself in `_HEAVY_PREFIXES` |

The last row is worth reading twice: `onnxruntime` and `fastembed` are in
`_HEAVY_PREFIXES` too. "It is only ONNX, so it is cheap" is not an argument for
importing it at module scope — the rule there is about CLI startup cost, not
about wheel size, and `aorta --help` is measured against it.

Nothing was added to `[cia]`, `[chat-cli]`, `all`, or the base dependencies,
which remain `pyyaml` plus `click`.

## Rule 2 — a verdict is only as good as the checkpoint that produced it

Pin by digest, not by name or tag, and record the checkpoint identity in every
report a local-classifier verdict reaches.

This repository has already been bitten by the general form of this, twice, and
in both cases the failure was silent rather than loud:

- [`docs/sanitizers/rocjitsu-bundle-provenance.md`](sanitizers/rocjitsu-bundle-provenance.md)
  opens with "a sanitizer verdict only means something relative to the hook that
  produced it," and then documents a local prebuilt directory that sat three
  commits behind the branch it came from. Anything pointing `ROCJITSU_PREBUILT`
  at it ran the pre-fix hook and would have reported the fixed bugs as unfixed.
  The verification that caught it only reached a trustworthy answer *because* the
  stale hook was caught.
- Decision 20a, in
  [`src/aorta/chat/rag/manifest.py`](../src/aorta/chat/rag/manifest.py), exists
  for the same reason in the retrieval index: a mismatched index does not raise,
  it answers fluently from vectors that were never comparable to the query's. So
  the manifest records an identity tuple rather than a model name, and an
  embedding or dimension mismatch refuses rather than warns.

[`docker/docker-compose.canary.yaml`](../docker/docker-compose.canary.yaml) makes
the same move in the build: `CANARY_BASE_IMAGE` uses the `:?` form so an unset
value aborts compose instead of building against "whatever `:latest` is now,"
because the alternative is an unattributable canary row.
[`gpu-tests.yml`](../.github/workflows/gpu-tests.yml) pins `busybox:1.37` by its
`sha256:` digest for the same reason: a tag retag must not silently change what a
gate means.

A weights change is exactly this failure with a new surface. Swapping one
fine-tuned checkpoint for another changes verdicts without changing a line of
code, without an error, and without anything in a report that says so. So:

- **Resolve by digest.** `convaiinnovations/laya` is a moving reference and a
  subfolder name is not a version. Pin the revision the way an image is pinned,
  and treat a checkpoint bump as a reviewed change rather than something that
  happens the next time a cache is cold.
- **Record it in the report, not beside it.** This is recommendation 2 of the
  rocjitsu provenance note, and the reason that note gives applies verbatim here:
  fields that travel *beside* a report cover every reader who arrives through a
  run area, but a report read on its own — copied, archived, attached to a
  ticket, or produced locally where there is no run area at all — still cannot
  answer "which model said this?" A Watch event, an Autopsy report and an agent
  step each carry the checkpoint identity inline.
- **Record the temperature fit with it.** The calibration is not in the weights.
  One temperature per (question type, option count) is what the model card
  reports moving mean ECE from 0.466 to 0.081 on its own evaluation, so a probability is a function of the checkpoint *and* the fit
  that was applied to it. A report naming one and not the other names half of
  what produced the number, and the thresholds downstream — Watch's
  `clean_threshold`, Autopsy's escalation cutoff in `probe.py` — are calibrated
  against the pair.
- **Never fetch a checkpoint implicitly at first use on a customer node.** The
  weights are ~808 MB and `[local-classifier]` will be installed beside `[cia]`
  on nodes that may have no egress.
  Follow the pattern `aorta chat doctor` already uses for the embedding model:
  when the weights are absent, say so and print the procedure for staging them,
  rather than surfacing a HuggingFace connection error that reads as a bug in
  aorta.

### Which number the probability is

There is a third thing a report has to be unambiguous about, and it is the one
most likely to be got wrong silently, because the trap is a field name.

Laya's own answer payload carries a field called `confidence`, and it is not
p(argmax). It is normalised Shannon entropy over the distribution, which measures
how *peaked* an answer is rather than how likely the top option is to be right.
Those come apart exactly where it matters: a near-uniform two-option answer and a
near-uniform ten-option answer score differently while being equally
uninformative. `src/aorta/local_classifier/predictor.py` therefore does not surface the field
at all, and `ChoiceAnswer.probability` returns the top option's probability
instead — the number a threshold compares against and the number a Brier score
and an ECE bin are built out of.

This belongs in a packaging and provenance decision rather than only in a
docstring, for the same reason the rest of Rule 2 does. The whole argument for
this integration is that a calibrated probability is worth more than an LLM's
self-reported `confidence` float. Recording a number under the name `confidence`
without recording which quantity it is reproduces that defect with a new source,
and it would be undetectable downstream: an entropy figure and a probability both
sit in [0, 1], both look plausible in a report, and a threshold fitted against
one applied to the other is wrong in a way that no test catches and no reader
notices.

The same care applies to the metrics themselves. `laya.common` ships `ece_score`
and `confidence_from_probs`, and `src/aorta/local_classifier/eval.py` deliberately reimplements
what it needs in about thirty lines of stdlib rather than importing them. The
stated reason is the import boundary — those helpers drag torch and numpy into
the one module that has to work with no model installed — but the second reason
is this one: these are the numbers a go/no-go decision gets made on, and they
should not be taken on trust from a dependency that moved three times in three
days.

## What this decision does not settle

The licence is not in question: Laya is Apache 2.0 with downloadable weights, so
it is clean for both the public and internal trees, and self-hosting is the
supported mode rather than a workaround.

What is deliberately left open is whether the CIA side ever grows a torch-free
path of its own. If the ONNX export built for Phase 5 turns out to be
maintainable, running every track on it would collapse this decision's two
runtimes into one and let the opt-in extra drop torch entirely. That is worth
revisiting after Phase 5 ships, and not before — building the export twice to
find out is more expensive than owning two runtimes for one release.
