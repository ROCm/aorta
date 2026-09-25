# Testing the local classifier

How to test the local calibrated classifier tier: what can be checked today with
nothing installed, what needs model weights, and what needs real CIA jobs. The
packaging rules this relies on are in
[local-classifier-packaging.md](local-classifier-packaging.md) (Decision 22), and
the history of why the tier exists is in
[plans/laya-system1-integration.md](plans/laya-system1-integration.md).

## Before you start: nothing is on, and nothing is measured

Every integration ships disabled, and each one also defaults to off in code
rather than only in configuration. No accuracy, Brier, ECE or latency figure
exists for this tier yet, because producing one needs model weights and a
corpus of real CIA job artifacts. Every threshold in the tree is therefore a
policy choice, and says so where it is defined.

That shapes what "testing" means here. Levels 0 and 1 check that the code is
correct and inert. Level 2 is the measurement that decides whether the tier
should ever be enabled. Levels 3 and 4 run the tier on real work, and Level 3
is also how the data that Level 2 needs gets collected.

| Level | What it tells you | Needs weights | Needs real CIA jobs |
|---|---|---|---|
| 0. Surface and packaging | The tier is invisible to users and the chat install stays torch-free | No | No |
| 1. Test suites | The code is correct against the fake predictor | No | No |
| 2. Measurement | Whether the tier is good enough to enable, per decision | Yes | Yes |
| 3. Watch archive and shadow mode | How the tier behaves on live logs, without changing anything | Shadow mode yes, archive no | Yes |
| 4. Per-integration smoke tests | Each integration works end to end with a real model | Yes | Some |

Level 4 also contains fallback checks that need no weights at all; see
[Checking the fallbacks](#checking-the-fallbacks-no-weights-needed).

## Level 0: surface and packaging checks

These take a minute and catch the two regressions that would matter most to a
user: the classifier appearing in the product surface, and the chat install
pulling in torch.

```bash
# The tier is not a product command.
aorta --help                                # no classifier entry in the list

# The maintainer tooling is reachable as a module instead.
python -m aorta.local_classifier --help     # lists `corpus` and `eval`
```

The second check repeats the one in `.github/workflows/nightly.yml` and
`release.yml`. Run it in a fresh virtual environment with only the chat extra
installed, because that is what CI does:

```bash
python3.11 -m venv /tmp/chat-only && . /tmp/chat-only/bin/activate
pip install -e ".[chat-cli]"
pip list --format=freeze | grep -Ei '^(torch|nvidia-|chromadb|sentence-transformers)'
# no output, and grep exits 1: the gate passes
```

Any output from that `grep` is a CI failure, and it means something in the
dependency tree now resolves torch.

## Level 1: the test suites

Everything here runs against `FakeDecisionPredictor`, so no weights are needed.
The suites live in two environments, because the CIA tests need the `[cia]`
extra and the chat tests need `[chat-cli]`.

```bash
# CIA, agent and CLI suites
pip install -e ".[tests,cia]"
python -m pytest tests/cia/ tests/agent/ tests/cli/ -q

# Chat suite, in an environment with the chat extra
pip install -e ".[tests,chat-cli]"
python -m pytest tests/chat/ -q -p no:xdist
```

Two things can make a run look green when it tested nothing:

- **Check that `tests/chat/` actually collected tests.** Its `conftest.py`
  skips collection entirely when `langchain_core` is missing, so without the
  chat extra it reports zero tests and exits successfully. With the extra it
  collects about 2,400 (2,378 at the time of PR #522). `tests/cia/` behaves the
  same way when `dspy` is missing.
- **`-p no:xdist` is there for large machines.** On a host with hundreds of
  cores, `-n auto` starts hundreds of workers and one timing assertion in
  `test_triage_threads.py` fails for reasons unrelated to the code. CI runners
  have 2 to 4 cores and are not affected.

The files specific to this tier, if you want to run only those:

| Area | Test files |
|---|---|
| Predictor, scoring, corpus, gate, CLI, question text | `tests/cia/test_local_classifier_*.py` |
| Watch clean-gate and archive | `tests/cia/test_watch_clean_gate.py`, `tests/cia/test_watch_clean_deltas.py` |
| Log finder | `tests/cia/test_tier3_answers_from_the_listing.py` |
| Autopsy | `tests/cia/test_autopsy_confidence_source.py` |
| Probe agent | `tests/agent/test_local_classifier_proposer.py` |
| Chat router, selector, ONNX path | `tests/chat/test_local_classifier_*.py` |
| Import boundaries | `tests/cli/test_chat_boundaries.py` |

## Level 2: the measurement

This answers the only question that decides whether a tier gets enabled: can
the classifier make this decision well enough, on these logs? It is scored per
decision, so the Watch gate can fail while the proposer passes.

### Set up

Use a separate environment from any `[chat-cli]` one, because this extra
installs torch. On a ROCm node, install the ROCm build of torch first. Otherwise
pip resolves the CUDA build, which is the hazard Decision 19a was written about.

```bash
python3.11 -m venv ~/local-classifier && . ~/local-classifier/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/rocm<your-rocm-version>
pip install -e ".[cia,local-classifier]"
```

`[cia]` is needed as well because the Watch and log-finder corpus builders
import their question text from the agents that ask it.

The default checkpoint is downloaded from Hugging Face on first use. On a node
without network access, stage a checkpoint directory yourself and pass it with
`--checkpoint`.

### Build the corpora

A corpus is joined from artifacts that already exist on disk. The Watch and
log-finder builders walk `CIA_JOBS_ROOT`, or `~/cia-jobs` if that is unset. The
proposer builder needs `--root` pointing at the `--output` directory of an
`aorta agent mitigate` run.

```bash
python -m aorta.local_classifier corpus watch      --output watch.jsonl
python -m aorta.local_classifier corpus log-finder --output log-finder.jsonl
python -m aorta.local_classifier corpus proposer   --root <agent-run-output> --output proposer.jsonl
```

Read the skip counts and warnings before anything else. They are half the
output: a builder that walked four hundred job directories and labelled nine
examples has told you Phase 1 cannot run yet. `corpus` exits non-zero when it
labelled nothing, and in that case it does not write the output file.

The Watch corpus needs job directories containing both
`bundle/logs/watch.stderr.log` and `bundle/report.json`. The plan's threshold is
at least 300 labelled examples for a decision; below that, a result cannot be
told apart from luck.

Two defects in the Watch corpus are known and documented in its build warnings:

- **There is no full-length healthy class yet.** `write_bundle` only runs after
  an alert, so every full-length log on disk belongs to a failure. The only
  healthy logs are 500-character excerpts, and a classifier can score well on
  that corpus by measuring length. Level 3's archive is the fix.
- **Watch's slugs and Autopsy's categories do not map cleanly.** Several
  categories collapse onto `WATCH_UNKNOWN_ERROR`, and `WATCH_LOSS_STALL` cannot
  appear as a label at all. This is why that decision is scored on macro
  accuracy.

### Score a checkpoint

```bash
python -m aorta.local_classifier eval --corpus watch.jsonl --backend laya          --output eval-base.json
python -m aorta.local_classifier eval --corpus watch.jsonl --backend laya-typed-decisions --output eval-td.json
python -m aorta.local_classifier eval --corpus watch.jsonl --checkpoint ./my-finetune     --output eval-ft.json
```

The first two score the published checkpoints as a floor, and the third is how
a fine-tune is compared against them. Useful options: `--holdout 0` fits
temperatures in-sample on a corpus too small to split (the report says so),
`--device` picks the torch device, and `--no-latency` / `--no-census` skip the
CPU timing and the token-length census.

`eval` scores against three baselines: the majority class, the existing regex,
and the current DSPy assessment's own agreement with the eventual Autopsy
category. The last is the real bar. `eval` exits non-zero when the kill
criterion is not met, so it works as a gate in a script. `--backend fake` is
refused outright, because a results file computed from hashes is
indistinguishable from a measurement.

Each decision is scored on the metric that fits how it is used:

| Decision | Primary metric | Why |
|---|---|---|
| `watch_healthy`, `proposer_stop` | `safe_coverage` | Both are thresholded, not argmaxed, and a false positive is the expensive error. It reports the coverage a model earns while staying inside a false-positive budget, printed beside `false_positive_rate` |
| `log_finder_useful`, `proposer_candidate` | `group_top1` | Both rank a set; plain accuracy over one-true-in-many rows rewards answering "no" to everything |
| `watch_signal` | `macro_accuracy` | The label distribution is lossy and skewed (see above) |
| `proposer_mitigation`, `proposer_category` | `accuracy` | Genuine argmax choices |

## Level 3: Watch in shadow mode

Shadow mode runs the classifier on every log delta that the sanitizer regex
does not answer, records what it would have decided, and changes nothing else.
Jobs behave exactly as they do today. It is the safe way to see the tier on
real logs, and it is also the only way to build the healthy class that Level 2
needs.

### Turn it on

Watch reads the `watch_config.yaml` that ships inside the package, and nothing
on the command line currently selects a different file. (The file's first
comment mentions `--config` on `watchdog poll`, but no command exposes that
flag.) Edit the packaged copy: in an editable install that is
`src/aorta/cia/watch/watch_config.yaml` in your checkout, and otherwise this
prints its path:

```bash
python -c "import aorta.cia.watch as w, pathlib; print(pathlib.Path(w.__file__).parent / 'watch_config.yaml')"
```

```yaml
watch:
  local_classifier:
    enabled: false                # leave the gate off
    shadow: true                  # record the classifier's verdicts
    shadow_archive_bytes: 5000000 # bytes per job; any positive value starts the healthy-delta archive
```

The two settings have different requirements, and that difference matters:

- **The archive needs no model.** `shadow_archive_bytes` is honoured whenever it
  is positive, independently of `shadow` and of whether any weights exist. It
  only needs the environment that already runs Watch (`[cia]`). You can turn it
  on today, and it is what the Level 2 measurement is actually waiting for.
- **Shadow mode needs the model.** The environment that launches jobs needs
  `pip install -e ".[cia,local-classifier]"` and access to the weights, as in
  Level 2.

### Run a job

Launch through the CIA driver as usual, either directly:

```bash
python -m aorta.cia.triage --command "<your training command>" --node <node>
python -m aorta.cia.triage --recipe <sanitizer-recipe.yaml> --node <node>
```

or by asking `aorta chat` to run a workload on the cluster, which reaches the
same driver. For a complete, copy-pasteable walkthrough with sample workloads
and the expected output of each, see
[Running a CIA sample job with the classifier on top](#running-a-cia-sample-job-with-the-classifier-on-top).

### What to look for

Each job writes into `<jobs-root>/<job-id>/`:

- **`events.jsonl`** gains a `watchdog_shadow` line after each verdict. Its
  `assessment` begins "Shadow only", and it carries the classifier's view beside
  the verdict Watch actually used:

  | Field | Meaning |
  |---|---|
  | `model_id` | Which checkpoint produced the numbers |
  | `clean_probability`, `clean_threshold`, `gated` | p(healthy), the threshold, and whether the gate would have skipped the LLM |
  | `vetoed` | The deterministic NaN scanner overruled the gate |
  | `signal`, `signal_probability` | The slug the classifier chose |
  | `clean_caveat`, `signal_caveat` | Non-empty when the calibration is clamped or unknown |
  | `watch_*` | The verdict Watch actually used, for comparison |

- **`local_classifier_clean_deltas.jsonl`** holds the logs that did not alert,
  with the verdict beside each. Every record is redacted with the same
  scrubbing Watch applies to anything it sends to a provider, truncated to
  4,000 characters so it is length-comparable with the failure bundles, and
  marked `"redacted": true`. The file stops growing at the byte cap.

If the tier cannot load, Watch prints one
`[watch] the local-classifier tier failed and will not be retried this run`
line and carries on exactly as before.

### Before anyone sets `enabled: true`

The YAML block lists four conditions, and all of them must hold. In short:
there is a real healthy class (which the archive provides); the chosen
checkpoint's `noul:2` bucket is not clamped (`is_clamped("noul:2")` is `False`,
not `None`); a false-clean rate has been measured on shadow traffic; and the
shadow period included a real failure.

Keep the archive running before and after the gate goes on. Once the gate
skips an LLM call, that delta no longer produces a healthy example, so enabling
the gate first would starve the corpus that validates it.

A false clean is not just a delay. A healthy verdict commits Watch's file
cursor, so a line printed once, such as a single `loss nan`, is never read
again. That is why the NaN veto exists and why raising `clean_threshold` is the
safe direction.

## Running a CIA sample job with the classifier on top

This is a start-to-finish run of the CIA pipeline (Launch, then Watch, then
Autopsy) with the classifier switched on over it. It uses
`python -m aorta.cia.triage`, the same driver `aorta chat` calls.

### Where each part runs

This decides what you install where, and it is easy to get backwards.

| Part | Runs on | Needs |
|---|---|---|
| The workload | A compute node, through `sbatch` | Whatever the workload needs; nothing from this tier |
| Watch (including the classifier tier and the archive) | The submit host, inside the triage process | `[cia,local-classifier]`, the weights for shadow mode, and an LLM provider |
| Autopsy (including its classifier tier) | The submit host, inside the triage process | The same, plus `CIA_AUTOPSY_LOCAL_CLASSIFIER_*` in that process's environment |

Triage calls `sbatch` and `sacct` directly, with no SSH hop, so run it on a
Slurm submit host. The job's log and bundle go under the jobs root, which must
be on a filesystem both the submit host and the compute node can see. The
classifier's forward passes run on the submit host's CPU.

### Prerequisites

1. **An environment on the submit host.** Install ROCm torch first (see
   [Level 2](#set-up)), then:

   ```bash
   pip install -e ".[cia,local-classifier]"
   ```

2. **An LLM provider for Watch and Autopsy.** This is independent of the
   classifier, and the pipeline does not work without it. With the gate off,
   Watch calls the LLM on every log delta the sanitizer regex does not answer.
   If that call fails three times, Watch records `WATCH_ASSESSMENT_FAILED`, and
   no shadow event or archive record is written for that delta. The agents
   read the same settings as `aorta chat`, either from
   `~/.config/aorta/chat.toml` (see [chat/configuration.md](chat/configuration.md))
   or from the environment:

   ```bash
   # A local or shared vLLM endpoint
   export AORTA_CHAT_LLM_PROVIDER=vllm
   export AORTA_CHAT_VLLM_BASE_URL=http://<host>:8000/v1
   export AORTA_CHAT_VLLM_MODEL=<model>

   # or any OpenAI-compatible API
   export AORTA_CHAT_LLM_PROVIDER=openai
   export AORTA_CHAT_REMOTE_LLM_BASE_URL=<endpoint>
   export AORTA_CHAT_REMOTE_LLM_API_KEY=<key>
   export AORTA_CHAT_REMOTE_LLM_MODEL=<model>
   ```

   Log text Watch sends to the provider is redacted by default.

3. **A separate jobs root for sample runs.** The samples below produce
   synthetic logs. If they land in the jobs root the Level 2 corpus is built
   from, they become training examples. Keep them apart:

   ```bash
   export CIA_JOBS_ROOT=/shared/path/cia-samples   # visible from the compute nodes
   ```

4. **Cluster settings**, as needed for your site:

   | Variable | Purpose |
   |---|---|
   | `CIA_PARTITION` | Slurm partition, for example `interactive` |
   | `CIA_TIME_LIMIT` | Job time limit (default `04:00:00`) |
   | `CIA_DEMO_NODE` or `--node` | Pin to one node; leave empty to let the scheduler choose |
   | `CIA_SBATCH_EXTRA` | Extra `#SBATCH` directives |
   | `CIA_CONTAINER_IMAGE`, `CIA_CONTAINER_EXTRA` | Run the workload inside a container |

### Turn the classifier on

In the packaged `watch_config.yaml` (see [Turn it on](#turn-it-on) above):

```yaml
watch:
  local_classifier:
    enabled: false
    shadow: true
    shadow_archive_bytes: 5000000
```

In the shell that will run triage, since Autopsy runs in that process:

```bash
export CIA_AUTOPSY_LOCAL_CLASSIFIER_ENABLED=1
# Leave CIA_AUTOPSY_LOCAL_CLASSIFIER_ESCALATION_THRESHOLD unset; see Level 4.
```

Both default to the `laya-typed-decisions` checkpoint. To use a staged
checkpoint directory instead, set `backend:` in the YAML and
`CIA_AUTOPSY_LOCAL_CLASSIFIER_BACKEND` to its path.

### Sample 1: a healthy training log (CPU only, about 4 minutes)

This is a synthetic workload that prints decreasing loss. It needs no GPU, and
it is the best first run because every log delta it produces reaches the
classifier: nothing in it matches the sanitizer regex. Watch polls every 60
seconds (`poll_interval_sec`), so a four-minute job gives it about four
deltas.

```bash
mkdir -p "$CIA_JOBS_ROOT/samples"
cat > "$CIA_JOBS_ROOT/samples/healthy_train.py" <<'EOF'
import time
for step in range(1, 25):
    print(f"step {step} loss {2.5 / step:.4f} tokens/s 1200", flush=True)
    time.sleep(10)
EOF

python -m aorta.cia.triage --label healthy-sample \
  --command "python3 -u $CIA_JOBS_ROOT/samples/healthy_train.py"
```

### Sample 2: a NaN partway through (CPU only, about 4 minutes)

The same workload, except the loss turns into `nan` at step 12. This exercises
the alert path: Watch should alert, assemble the bundle and start Autopsy.
Watch's deterministic NaN scanner also stops the classifier from ever marking
these deltas as clean.

```bash
cat > "$CIA_JOBS_ROOT/samples/nan_train.py" <<'EOF'
import time
for step in range(1, 25):
    loss = "nan" if step >= 12 else f"{2.5 / step:.4f}"
    print(f"step {step} loss {loss} tokens/s 1200", flush=True)
    time.sleep(10)
EOF

python -m aorta.cia.triage --label nan-sample \
  --command "python3 -u $CIA_JOBS_ROOT/samples/nan_train.py"
```

### Sample 3: a real GPU run with a sanitizer recipe

This runs the committed racy ConSan repro on a gfx950 node. It tests the whole
chain on real hardware, and Autopsy's category tier gets real sanitizer
evidence. The sanitizers need your RocJITsu build passed into the job:

```bash
python -m aorta.cia.triage --label consan-racy --arch gfx950 \
  --recipe recipes/sanitizers/daily-consan-racy.yaml \
  --env ROCJITSU_BUILD=<path-to-rocjitsu-build> \
  --env LD_PRELOAD=<rocjitsu-preload-library>
```

This sample is a weak test of Watch's classifier. The line that decides the
run is the sanitizer's own `[sanitizer] consan: verdict=… state=…` summary,
which Watch's regex answers before the classifier tier is consulted. The
classifier only sees the earlier deltas, such as build and progress output.
`recipes/sanitizers/daily-consan-clean.yaml` is the clean counterpart. For a
real training job, pass its normal launch command with `--command`.

### What triage prints

Progress lines (`[triage] …`, including `── Launch ──`, `── Watch ──` and
`── Autopsy … ──`) go to stderr. Watch's per-poll `[watch] …` lines and a final
JSON result go to stdout. The result includes `ok`, `job_id`, `slurm_job_id`,
`job_dir`, `watch_alerted`, `watch_tail` and an `autopsy` summary. The exit
status is 0 when `ok` is true.

Every run ends with a `report.json`. If Watch alerted, Watch produced it.
Otherwise triage assembled the bundle and ran Autopsy itself after the job
finished (`── Autopsy (direct) ──`).

### Inspect the job directory

```bash
J=$(ls -td "$CIA_JOBS_ROOT"/cia-* | head -1)   # the newest run
ls "$J" "$J/bundle"
```

| File | What it is |
|---|---|
| `job.json`, `launch.sbatch` | The job record and the batch script that was submitted |
| `watch.log` | The workload's stdout and stderr, which is what Watch reads |
| `events.jsonl` | Watch's verdicts (`watchdog_ok` / `watchdog_alert`), each followed by a `watchdog_shadow` line |
| `local_classifier_clean_deltas.jsonl` | Deltas that did not alert, redacted and truncated (present when the archive is on) |
| `bundle/manifest.yaml`, `bundle/logs/watch.stderr.log` | The evidence bundle Autopsy reads |
| `bundle/report.json` | Autopsy's verdict |

What the classifier said on each poll, next to what Watch used:

```bash
jq -c 'select(.event_type=="watchdog_shadow")
       | {model_id, clean_probability, gated, vetoed, signal, signal_probability,
          watch_signal, watch_healthy, clean_caveat}' "$J/events.jsonl"
```

What the archive kept:

```bash
jq -c '{healthy, signal, delta_chars, truncated}' "$J/local_classifier_clean_deltas.jsonl"
```

Autopsy's verdict and where its confidence came from:

```bash
jq '{category, confidence, confidence_source, local_classifier}' "$J/bundle/report.json"
```

### What to expect

| | Sample 1 (healthy) | Sample 2 (NaN) |
|---|---|---|
| `watch_alerted` | `false` | `true` |
| `events.jsonl` | `watchdog_ok` lines, each followed by `watchdog_shadow` | `watchdog_ok` for the early deltas, then `watchdog_alert` with `WATCH_NUMERIC_NAN` |
| `gated` in shadow events | `true` whenever `clean_probability` reached `clean_threshold` | `false` on the NaN deltas; `vetoed` is `true` wherever the classifier was confident but the NaN scanner overruled it |
| Archive | One record per poll | Records for the polls before the NaN |
| `report.json` | From the direct Autopsy | From Watch's Autopsy |
| `report.json` classifier fields | A `local_classifier` block with `model_id`, `bucket` (`choice:11+`), `clamped` and `caveat`, and `confidence_source.source` of `local_classifier` | The same |

The categories and probabilities themselves come from the models and are not
fixed. Treat these samples as a test of the plumbing. They show that every
tier ran, fell back, or disclosed what it should, and they say nothing about
whether the classifier is good. That question belongs to Level 2, on real jobs.

### If a tier did not run

| Symptom | Likely cause |
|---|---|
| No `watchdog_shadow` lines and one `[watch] the local-classifier tier failed…` line | `[local-classifier]` is not installed on the submit host, or the weights could not be loaded |
| No `watchdog_shadow` lines and no error | `shadow` is not `true` in the YAML that was actually read (check the path from [Turn it on](#turn-it-on)) |
| `WATCH_ASSESSMENT_FAILED` events | No LLM provider is configured, or it is unreachable |
| No `local_classifier` block in `report.json` | `CIA_AUTOPSY_LOCAL_CLASSIFIER_ENABLED` was not set in the triage process, or the tier failed and Autopsy fell back |
| `sbatch not found on PATH` | Triage is not running on a Slurm submit host |

### From samples to real data

Once the samples show every tier working, point `CIA_JOBS_ROOT` back at the
jobs root your real runs use, and leave the archive and shadow mode on across
real jobs. Those job directories are what
`python -m aorta.local_classifier corpus watch` reads in Level 2.

## Level 4: each integration with a real model

Each integration has its own switch, and each keeps working on the LLM when the
switch is off.

| Integration | How to turn it on | What confirms it ran |
|---|---|---|
| Probe agent | `aorta agent mitigate <usual arguments> --llm-backend local`, optionally `--llm-model <checkpoint or fine-tune dir>` | Each step's hypothesis shows `p(<mitigation>)=… of N candidates (<bucket>)`, with a `NOT CALIBRATED` note when the choice falls in a clamped bucket |
| Autopsy | `export CIA_AUTOPSY_LOCAL_CLASSIFIER_ENABLED=1`, optionally `CIA_AUTOPSY_LOCAL_CLASSIFIER_BACKEND=<checkpoint>` | `report.json` gains a `local_classifier` block (`model_id`, `options`, `bucket`, `clamped`, `caveat`), and `confidence_source.source` reads `local_classifier` instead of `adapter_rules` or `llm_self_report` |
| Log finder | `log_finder.local_classifier.enabled: true` in the same YAML | See the note below |
| Chat router and selector | `export AORTA_CHAT_LOCAL_CLASSIFIER_ENABLED=true` and an exported ONNX artifact | The log shows `Router classified as: <route> [<model_id>: p(action)=… at threshold …]` instead of `(raw: '…')` |

**Autopsy:** leave `CIA_AUTOPSY_LOCAL_CLASSIFIER_ESCALATION_THRESHOLD` unset.
The existing 0.85 escalation cutoff was fitted against an LLM's self-report, so
Autopsy deliberately never applies it to a classifier probability. With no
threshold set, escalation keeps using the adapters' figure, and enabling the
tier changes what the report says without changing what the pipeline does.
Only set that variable once you have derived a threshold for the classifier.

**Log finder:** this tier only runs for jobs whose `job.json` has an empty
`log_path`, and `aorta.cia.triage` always sets one. It is therefore hard to
reach with triage-launched jobs; the unit tests in
`test_tier3_answers_from_the_listing.py` are the practical coverage. The
containment half of this change, where the LLM tier can only answer with files
that were in the listing, applies whether the flag is on or off.

**Chat:** the router and selector run on an exported ONNX artifact, so the chat
install never needs torch. There is no command for the export yet. If you set
`AORTA_CHAT_LOCAL_CLASSIFIER_ENABLED=true` without an artifact, the warning
prints the snippet to run. In short, on a machine with torch:

```bash
pip install -e ".[local-classifier]"
python - <<'EOF'
from aorta.chat.local_classifier.export import export_artifact
export_artifact('<checkpoint-dir-or-name>', '/path/to/artifact', temperatures={})
EOF
```

Copy the directory to the chat machine and point at it with
`AORTA_CHAT_LOCAL_CLASSIFIER_ARTIFACT_PATH`. Set
`AORTA_CHAT_LOCAL_CLASSIFIER_VERIFY_DIGEST=true` wherever a verdict gets
published, so a graph and manifest that no longer match are refused rather than
producing a plausible answer. The selector only ranks tools; the act node still
sees every tool whatever the classifier says.

## Checking the fallbacks (no weights needed)

Every tier is designed to degrade to today's behaviour rather than fail the
work it sits inside. Because the weights are the thing that is missing, this
can be tested now:

| Tier | Try this | Expected |
|---|---|---|
| Watch | `shadow: true` in an environment without the `[local-classifier]` extra | One `[watch] the local-classifier tier failed…` line, then normal assessment |
| Autopsy | `CIA_AUTOPSY_LOCAL_CLASSIFIER_ENABLED=1` without the extra | The report is produced as today: no `local_classifier` block, and `confidence_source.source` is not `local_classifier` |
| Chat | `AORTA_CHAT_LOCAL_CLASSIFIER_ENABLED=true` with no artifact | A `local-classifier tier unavailable (…); falling back to the LLM` warning, then a normal answer |
| Probe agent | `--llm-backend local` without the extra | Stops with `ClassifierUnavailableError` and an install hint |

The probe agent is the exception on purpose: its backend is something you chose
explicitly on the command line, so it fails loudly rather than quietly
substituting another model.

## Recommended order

1. Levels 0 and 1, to confirm the branch is correct and inert on your setup.
2. The fallback checks above.
3. Turn on the Level 3 archive now. It needs no weights, changes nothing, and
   builds the healthy class the measurement is blocked on. Add shadow mode once
   weights are available.
4. Level 2, once a decision has at least 300 labelled examples.
5. Level 4, only for the integrations whose Level 2 verdict passed.
