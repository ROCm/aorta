#!/usr/bin/env bash
# Kernel-level probe for TokenSpeed's AMD kernels. Runs INSIDE the TokenSpeed
# container; `host_launch.sh` is what puts it there (TS_ENTRY=ts_kernel_probe.sh).
#
# This is the cheap half of the TokenSpeed integration. It exercises the Gluon /
# Triton kernels directly through TokenSpeed's own harnesses -- no model weights,
# no HTTP server, no readiness polling -- so a trial is seconds rather than the
# ~5 minutes a serving bring-up costs.
#
#   TS_KERNEL_MODE=numerics  python -m tokenspeed_kernel.numerics
#       Correctness against the PyTorch reference implementation. Its CLI exits
#       1 on failure, so the exit code alone is a trustworthy verdict.
#
#   TS_KERNEL_MODE=bench     python -m tokenspeed_kernel.benchmark --verify
#       Latency / TFLOPs / bandwidth per shape per kernel, plus verification.
#
# The benchmark CLI ALWAYS returns 0 -- including when --verify finds a numerics
# mismatch (see tokenspeed_kernel/benchmark/cli.py, which ends in a bare
# `return 0`). Trusting its exit code would turn a wrong-answer kernel into a
# green aorta cell, so this script re-reads the exported JSON and fails the trial
# itself when any record has numerics_passed == false.
#
# Exit codes:
#   0   requested kernels ran, and verification passed where it was requested
#   30  the tokenspeed_kernel CLI exited non-zero
#   31  benchmark exported no records (bad --op / --dtype-role / shape filter)
#   32  a record reported numerics_passed == false
#   33  export file missing or unparseable
#   34  export parsed but numerics_passed is absent or not bool|null, so the
#       wrong-answer gate could not be evaluated (upstream schema change)
#   64  usage / environment error
#
# Env:
#   TS_KERNEL_MODE        numerics | bench | both     (default bench)
#   TS_KERNEL_OP          operator family.mode, e.g. gemm.mm
#   TS_KERNEL_NAME        single kernel, e.g. gluon_mm_a16w16_gfx950
#                         (takes precedence over TS_KERNEL_OP; this is the knob
#                          a mitigations sidecar flips to pit the Gluon, Triton
#                          and torch solutions against each other)
#   TS_KERNEL_DTYPE       bf16 | fp16 | fp32 | fp8    (default bf16)
#   TS_KERNEL_DTYPE_ROLE  tensor role selected by --dtype; operator-specific --
#                         a/b for gemm.mm, q/k/v for attention.mha_prefill,
#                         q/k_cache/v_cache for the with_kvcache modes, x for
#                         moe.apply, logits for sampling.argmax (default a)
#   TS_KERNEL_WARMUP      warmup iters                (default 5)
#   TS_KERNEL_ITERS       bench iters                 (default 20)
#   TS_KERNEL_ARGS        extra args, word-split
#   TS_OUT_DIR            where the export lands      (default /ts-out)
#   TS_RUN_TOKEN          tag qualifying this trial's export filename; set by
#                         host_launch.sh. Falls back to $$, which is only
#                         distinct when running outside a fresh PID namespace.

set -uo pipefail

MODE="${TS_KERNEL_MODE:-bench}"
DTYPE="${TS_KERNEL_DTYPE:-bf16}"
DTYPE_ROLE="${TS_KERNEL_DTYPE_ROLE:-a}"
WARMUP="${TS_KERNEL_WARMUP:-5}"
ITERS="${TS_KERNEL_ITERS:-20}"
OUT_DIR="${TS_OUT_DIR:-/ts-out}"

mkdir -p "${OUT_DIR}" || {
  echo "TS_KERNEL_FAIL: cannot create out dir ${OUT_DIR}"
  exit 64
}

case "${MODE}" in
  numerics|bench|both) ;;
  *)
    echo "TS_KERNEL_FAIL: TS_KERNEL_MODE must be numerics|bench|both, got '${MODE}'"
    exit 64
    ;;
esac

# --op and a bare kernel name are mutually exclusive positional/flag forms in the
# upstream CLI; a single kernel wins so a sidecar can pin one solution.
selector=()
selector_desc=""
if [ -n "${TS_KERNEL_NAME:-}" ]; then
  selector=("${TS_KERNEL_NAME}")
  selector_desc="kernel=${TS_KERNEL_NAME}"
elif [ -n "${TS_KERNEL_OP:-}" ]; then
  selector=(--op "${TS_KERNEL_OP}")
  selector_desc="op=${TS_KERNEL_OP}"
else
  echo "TS_KERNEL_FAIL: set TS_KERNEL_OP (e.g. gemm.mm) or TS_KERNEL_NAME"
  exit 64
fi

echo "TS_KERNEL_INFO: mode=${MODE} ${selector_desc} dtype=${DTYPE} role=${DTYPE_ROLE}"
echo "TS_KERNEL_INFO: tokenspeed=$(tokenspeed version 2>&1 | tr '\n' ' ')"
echo "TS_KERNEL_INFO: rocm=$(cat /opt/rocm/.info/version 2>/dev/null || echo unknown)"
echo "TS_KERNEL_INFO: arch=$(python3 -c 'import torch; print(torch.cuda.get_device_properties(0).gcnArchName)' 2>/dev/null || echo unknown)"

if [ "${MODE}" = "numerics" ] || [ "${MODE}" = "both" ]; then
  echo "TS_KERNEL_INFO: running numerics"
  # shellcheck disable=SC2086
  python3 -m tokenspeed_kernel.numerics \
    "${selector[@]}" --dtype "${DTYPE}" --dtype-role "${DTYPE_ROLE}" \
    ${TS_KERNEL_ARGS:-}
  rc=$?
  if [ "${rc}" -ne 0 ]; then
    echo "TS_KERNEL_FAIL: numerics_cli_rc=${rc}"
    exit 30
  fi
  echo "TS_KERNEL_OK: numerics"
fi

if [ "${MODE}" = "bench" ] || [ "${MODE}" = "both" ]; then
  # Token-qualified: TS_OUT_DIR is one host directory shared by every trial in
  # the matrix, so a fixed name would leave only the last trial's export behind
  # and let concurrent trials read each other's results. TS_RUN_TOKEN comes from
  # host_launch.sh -- see the note there on why an in-container $$ cannot do
  # this job.
  export_path="${OUT_DIR}/kernel_bench.${TS_RUN_TOKEN:-$$}.json"
  rm -f "${export_path}"
  echo "TS_KERNEL_INFO: running benchmark (warmup=${WARMUP} iters=${ITERS})"
  # shellcheck disable=SC2086
  python3 -m tokenspeed_kernel.benchmark \
    "${selector[@]}" --dtype "${DTYPE}" --dtype-role "${DTYPE_ROLE}" \
    --verify --warmup-iters "${WARMUP}" --bench-iters "${ITERS}" \
    --export "${export_path}" \
    ${TS_KERNEL_ARGS:-}
  rc=$?
  if [ "${rc}" -ne 0 ]; then
    echo "TS_KERNEL_FAIL: benchmark_cli_rc=${rc}"
    exit 30
  fi

  # The export is the real verdict for this mode: see the header note about the
  # CLI's unconditional `return 0`.
  summary=$(python3 - "${export_path}" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    with open(path) as fh:
        records = json.load(fh)
except FileNotFoundError:
    print("STATUS missing")
    raise SystemExit(0)
except Exception as exc:  # noqa: BLE001 - any parse problem is one outcome here
    print(f"STATUS unparseable {type(exc).__name__}: {exc}")
    raise SystemExit(0)

if not isinstance(records, list) or not records:
    print("STATUS empty")
    raise SystemExit(0)

# Validate the one field the verdict depends on, rather than reading it with
# .get(). `numerics_passed` is legitimately null when verification was not
# requested, so a renamed or removed field is indistinguishable from that under
# .get() -- an upstream schema change would silently disable the wrong-answer
# gate while every trial stayed green. Absent or wrongly-typed is a hard error;
# null still means "not verified".
schema_errors = []
for index, rec in enumerate(records):
    if not isinstance(rec, dict):
        schema_errors.append(f"record[{index}] is {type(rec).__name__}, expected object")
        continue
    if "numerics_passed" not in rec:
        schema_errors.append(f"record[{index}] has no numerics_passed field")
        continue
    value = rec["numerics_passed"]
    if value is not None and not isinstance(value, bool):
        schema_errors.append(
            f"record[{index}].numerics_passed is {value!r}, expected bool or null"
        )
if schema_errors:
    print(f"STATUS schema_error count={len(schema_errors)}")
    for problem in schema_errors[:10]:
        print(f"SCHEMAERR {problem}")
    raise SystemExit(0)

failed = [r for r in records if r["numerics_passed"] is False]
print(f"STATUS ok records={len(records)} failed={len(failed)}")

for rec in failed:
    print(
        "FAILREC "
        f"{rec.get('kernel_name')} shape={rec.get('shape_params')} "
        f"max_abs_diff={rec.get('max_abs_diff')} "
        f"max_rel_diff={rec.get('max_rel_diff')}"
    )

# One METRIC line per kernel+shape. aorta's probe path has no metrics dict --
# that arrives with the Phase 1 workload -- so stdout is the only channel, and
# these lines are what a reader (or a later parser) keys on.
for rec in records:
    shape = rec.get("shape_params") or {}
    shape_key = "x".join(f"{k}{v}" for k, v in sorted(shape.items()))
    print(
        "METRIC "
        f"kernel={rec.get('kernel_name')} solution={rec.get('solution')} "
        f"shape={shape_key} arch={rec.get('platform_arch')} "
        f"median_latency_us={rec.get('median_latency_us')} "
        f"p99_latency_us={rec.get('p99_latency_us')} "
        f"tflops={rec.get('tflops')} bandwidth_gb_s={rec.get('bandwidth_gb_s')} "
        f"numerics_passed={rec.get('numerics_passed')}"
    )
PY
  )

  status_line=$(printf '%s\n' "${summary}" | grep '^STATUS ' | head -1)
  printf '%s\n' "${summary}" | grep -E '^(METRIC|FAILREC) ' | sed 's/^/TS_KERNEL_/'

  case "${status_line}" in
    "STATUS ok"*)
      n_failed=$(printf '%s\n' "${status_line}" | sed -n 's/.*failed=\([0-9]*\).*/\1/p')
      n_records=$(printf '%s\n' "${status_line}" | sed -n 's/.*records=\([0-9]*\).*/\1/p')
      echo "TS_KERNEL_INFO: ${n_records} record(s), ${n_failed} numerics failure(s)"
      if [ "${n_failed:-0}" -gt 0 ]; then
        echo "TS_KERNEL_FAIL: numerics_mismatch count=${n_failed}"
        exit 32
      fi
      ;;
    "STATUS empty")
      echo "TS_KERNEL_FAIL: benchmark_exported_no_records"
      echo "  ${selector_desc} dtype=${DTYPE} role=${DTYPE_ROLE} matched nothing."
      echo "  Check the role is valid for this operator (a/b, q/k/v, x, logits)."
      exit 31
      ;;
    "STATUS missing")
      echo "TS_KERNEL_FAIL: export_missing path=${export_path}"
      exit 33
      ;;
    "STATUS schema_error"*)
      # The verdict field this probe gates on is gone or the wrong type, so the
      # export cannot be trusted either way. Reported as its own failure rather
      # than folded into "unparseable": the file parsed fine, it is the contract
      # that moved, and someone has to update this probe.
      echo "TS_KERNEL_FAIL: export_schema_changed ${status_line}"
      printf '%s\n' "${summary}" | grep '^SCHEMAERR ' | sed 's/^/TS_KERNEL_/'
      echo "  numerics_passed is missing or not bool|null -- the wrong-answer"
      echo "  gate cannot be evaluated. Re-check the tokenspeed_kernel export"
      echo "  schema for this image and update ts_kernel_probe.sh."
      exit 34
      ;;
    *)
      echo "TS_KERNEL_FAIL: export_unparseable ${status_line}"
      exit 33
      ;;
  esac
  echo "TS_KERNEL_INFO: export=${export_path}"
  echo "TS_KERNEL_OK: benchmark"
fi

echo "TS_KERNEL_RESULT: pass"
exit 0
