# Probe-mode harness for the `llm_determinism` replay

Runs aorta's bit-exact deterministic-replay workload as the **opaque user
command** of a probe-mode recipe, so `aorta agent mitigate` can search a
mitigation axis against a real multi-GPU workload.

Findings from the run these scripts performed are in
[`docs/probe/llm-determinism-gpu-archive.md`](../../../docs/probe/llm-determinism-gpu-archive.md).
Short version: on 8× MI350X the replay is clean in every configuration tried,
so the archive is a **control**, not a reproduction.

## Why a probe recipe and not the matrix recipe

`recipes/llm-determinism/example-llm-determinism.yaml` has the right mitigation
axis but is matrix-mode: no `mode: probe`, so `recipe.probe_extras is None` and
`run_agent_loop` rejects it at `agent/loop.py:104`. The workload is still the
right thing to run; it just has to arrive as the launch command.
`recipes/probe/probe-llm-determinism-replay.yaml` supplies the axis.

## The container boundary

The workload needs PyTorch+ROCm, which lives in a container, and `docker run`
does not forward the host environment. Under the `inherit` default the
mitigation bundle would be stamped on the wrapper's environment and dropped at
the container edge — every cell would silently be the baseline and the matrix
would be a lie that looks like a result. The recipe therefore sets
`env_passthrough_mode: file`, and `launch_det.sh` passes the per-trial
`probe.env` through as `docker run --env-file "$AORTA_ENV_FILE"`.

Check this rather than assume it: each `trial_*/result.json` carries the cell's
`env`, and the `tf32_off-none` trials must show `{"DISABLE_TF32": "1"}`.

## Files

| File | Role |
|---|---|
| `det_launch.py` | In-container entry point. Builds `LlmDeterminismWorkload` from `DET_CFG`, exits 1 on divergence. Unmodified workload. |
| `det_launch_injected.py` | **Detector validation only.** The perturbation `docs/llm-determinism.md` ships to prove the detector fires. Its verdict is a property of the injection; never a training scenario. |
| `launch_det.sh` | The opaque user command: `docker run … torchrun --nproc_per_node=8 <entry>`, carrying `probe.env` across the container boundary. |
| `scout.sbatch` | Seven shipped configurations, with and without `DISABLE_TF32`, to find out whether any divergence exists to search for. |
| `run_probe.sbatch` | The archive run: agent (unmodified), sweep (full axis), harness validation (injected). |

## Site-specific paths

Both sbatch files and `launch_det.sh` are committed **as they ran**, so they
hard-code `/apps/vikhande/probe-gpu` and `-w cv350-rck-g03-c07-18`. Reusing
them elsewhere means changing `ROOT`, the `-w` line and `IMAGE`. The
`smci350` guard in both sbatch files should not be removed while a perf
baseline window is live on the MI350 CI runner.
