#!/usr/bin/env bash
# Kernel probe via TokenSpeed's own op test suites. Runs INSIDE the TokenSpeed
# container; `host_launch.sh` puts it there (TS_ENTRY=ts_pytest_probe.sh).
#
# Why this exists alongside ts_kernel_probe.sh: the benchmark harness can only
# drive gemm.mm, because every other operator family is missing an input
# generator and a shape list. TokenSpeed's own pytest suites build those inputs
# themselves, so they reach the attention, MoE, quantization, sampling and
# transform kernels the benchmark harness cannot. Measured on nightly-20260714
# the suites enter 20 of the 38 registered kernels, against 9 the benchmark
# harness can reach (all of them gemm.mm) -- and the two sets do not overlap.
# "Enter" means the implementation was actually run, which is narrower than
# either of the two things easily mistaken for it: being looked up, and being one
# of the candidates considered. See docs/tokenspeed.md for the per-status
# breakdown.
#
# The trade-off is what you get back. These suites assert, they do not measure,
# so there are no TFLOPs or latency numbers here -- use ts_kernel_probe.sh for
# gemm perf. What this gives is a correctness verdict per family and, because
# running the tests JITs the kernels, a populated code-object cache that
# harvest_code_objects.py can turn into a Waitcheck corpus.
#
# pytest exits 0 when every test was skipped or deselected, and these suites
# skip heavily (NVIDIA-only solutions are not registered on AMD, so a single
# file can report ~500 skips). Trusting the exit code alone would turn "nothing
# ran" into a green cell, so this script requires at least one passing test and
# fails the trial otherwise.
#
# Exit codes:
#   0   the selected suite ran and every executed test passed
#   40  pytest reported failures or errors
#   41  nothing executed -- everything skipped or deselected (see above)
#   42  the JUnit report is missing or unparseable
#   64  usage / environment error
#
# Env:
#   TS_PYTEST_SUITE   test path to run, relative to TS_WORKSPACE or absolute.
#                     Required. e.g. tokenspeed-kernel/test/ops/test_attention.py
#   TS_PYTEST_K       -k selection expression                  (default unset)
#   TS_PYTEST_ARGS    extra pytest args, word-split            (default unset)
#   TS_WORKSPACE      TokenSpeed source tree root              (default /workspace)
#   TS_MIN_PASSED     minimum passing tests required           (default 1)
#   TS_OUT_DIR        where the JUnit report lands             (default /ts-out)
#   TS_RUN_TOKEN      tag qualifying this trial's filenames; set by
#                     host_launch.sh. Falls back to $$, which is only distinct
#                     when running outside a fresh PID namespace.

set -uo pipefail

WORKSPACE="${TS_WORKSPACE:-/workspace}"
OUT_DIR="${TS_OUT_DIR:-/ts-out}"
MIN_PASSED="${TS_MIN_PASSED:-1}"
TOKEN="${TS_RUN_TOKEN:-$$}"

if [ -z "${TS_PYTEST_SUITE:-}" ]; then
  echo "TS_PYTEST_FAIL: usage TS_PYTEST_SUITE must name a test path"
  echo "  e.g. TS_PYTEST_SUITE=tokenspeed-kernel/test/ops/test_attention.py"
  exit 64
fi

if [ ! -d "${WORKSPACE}" ]; then
  echo "TS_PYTEST_FAIL: usage workspace ${WORKSPACE} is not a directory"
  exit 64
fi

# Validated as >= 1 before it is ever used in a comparison. This is the
# all-skipped guard: a suite where every test skipped exits pytest 0 with zero
# passes, so `passed >= MIN_PASSED` is the only thing standing between that and
# a green trial. TS_MIN_PASSED=0 disables the guard outright, and a non-numeric
# value makes `[` error out -- which, because pytest itself returned 0, also
# falls through to a pass.
case "${MIN_PASSED}" in
  ''|*[!0-9]*)
    echo "TS_PYTEST_FAIL: usage TS_MIN_PASSED must be a positive integer, got '${MIN_PASSED}'"
    exit 64
    ;;
esac
# All-digits is not enough: `[ -lt ]` evaluates arithmetically, so a leading
# zero is read as octal (`08` aborts with "invalid octal number" and exit 1
# rather than the documented 64) and a value past 2^63 wraps back into range.
case "${MIN_PASSED}" in
  0) ;;
  0*)
    echo "TS_PYTEST_FAIL: usage TS_MIN_PASSED must not have leading zeros, got '${MIN_PASSED}'"
    exit 64
    ;;
esac
if [ "${#MIN_PASSED}" -gt 10 ]; then
  echo "TS_PYTEST_FAIL: usage TS_MIN_PASSED is too large to evaluate, got '${MIN_PASSED}'"
  exit 64
fi
if [ "${MIN_PASSED}" -lt 1 ]; then
  echo "TS_PYTEST_FAIL: usage TS_MIN_PASSED must be >= 1, got '${MIN_PASSED}'"
  echo "  0 would disable the all-skipped guard, which is the only check that"
  echo "  distinguishes a real pass from a suite that skipped everything."
  exit 64
fi

mkdir -p "${OUT_DIR}" || {
  echo "TS_PYTEST_FAIL: usage cannot create out dir ${OUT_DIR}"
  exit 64
}

case "${TS_PYTEST_SUITE}" in
  /*) SUITE="${TS_PYTEST_SUITE}" ;;
  *)  SUITE="${WORKSPACE}/${TS_PYTEST_SUITE}" ;;
esac

if [ ! -e "${SUITE}" ]; then
  echo "TS_PYTEST_FAIL: usage suite ${SUITE} does not exist"
  exit 64
fi

# Each suite is a package rooted at the parent of its `test` directory, with its
# own conftest and a top-level `utils` module it expects to import. Running from
# anywhere else fails collection, and running two suites from one interpreter
# collides on those module names -- hence one suite per trial, from its own root.
suite_root() {
  local d="$1"
  while [ "${d}" != "/" ]; do
    if [ "$(basename "${d}")" = "test" ]; then
      dirname "${d}"
      return 0
    fi
    d="$(dirname "${d}")"
  done
  echo "${WORKSPACE}"
}

ROOT="$(suite_root "${SUITE}")"
REPORT="${OUT_DIR}/pytest.${TOKEN}.xml"

echo "TS_PYTEST_INFO: suite=${SUITE}"
echo "TS_PYTEST_INFO: root=${ROOT} token=${TOKEN}"
[ -n "${TS_PYTEST_K:-}" ] && echo "TS_PYTEST_INFO: -k ${TS_PYTEST_K}"

pytest_args=("${SUITE}" -q --no-header)
# The source tree is read-only for a non-root container user, and pytest's cache
# plugin treats an unwritable rootdir as a warning-worthy failure. Disable it
# rather than let that noise into every trial's stderr.
pytest_args+=(-p no:cacheprovider)
pytest_args+=("--junit-xml=${REPORT}")
[ -n "${TS_PYTEST_K:-}" ] && pytest_args+=(-k "${TS_PYTEST_K}")
# shellcheck disable=SC2206  # word-splitting TS_PYTEST_ARGS is deliberate
[ -n "${TS_PYTEST_ARGS:-}" ] && pytest_args+=(${TS_PYTEST_ARGS})

cd "${ROOT}" || {
  echo "TS_PYTEST_FAIL: usage cannot cd to ${ROOT}"
  exit 64
}

start=$(date +%s)
python3 -m pytest "${pytest_args[@]}"
rc=$?
elapsed=$(( $(date +%s) - start ))
echo "TS_PYTEST_INFO: pytest rc=${rc} in ${elapsed}s"
echo "TS_PYTEST_METRIC: suite_walltime_sec=${elapsed}"

if [ ! -r "${REPORT}" ]; then
  echo "TS_PYTEST_FAIL: report_missing ${REPORT}"
  exit 42
fi

# Counts come from the JUnit report rather than the terminal summary: the
# summary line's wording shifts between pytest versions, whereas the XML
# attributes are stable and are what the verdict below depends on.
counts=$(python3 - "${REPORT}" <<'PY'
import sys
import xml.etree.ElementTree as ET

try:
    root = ET.parse(sys.argv[1]).getroot()
except Exception as exc:  # noqa: BLE001 - any parse failure is the same verdict
    print(f"PARSE_ERROR {exc}")
    raise SystemExit(0)

# pytest emits <testsuites><testsuite .../></testsuites>; older versions emit a
# bare <testsuite>. Sum across suites so both shapes work.
suites = root.findall("testsuite") or ([root] if root.tag == "testsuite" else [])
total = sum(int(s.get("tests", 0)) for s in suites)
failures = sum(int(s.get("failures", 0)) for s in suites)
errors = sum(int(s.get("errors", 0)) for s in suites)
skipped = sum(int(s.get("skipped", 0)) for s in suites)
print(f"{total} {failures} {errors} {skipped}")
PY
)

case "${counts}" in
  PARSE_ERROR*)
    echo "TS_PYTEST_FAIL: report_unparseable ${counts}"
    exit 42
    ;;
esac

read -r total failures errors skipped <<<"${counts}"
passed=$(( total - failures - errors - skipped ))

echo "TS_PYTEST_METRIC: tests_total=${total}"
echo "TS_PYTEST_METRIC: tests_passed=${passed}"
echo "TS_PYTEST_METRIC: tests_failed=${failures}"
echo "TS_PYTEST_METRIC: tests_errored=${errors}"
echo "TS_PYTEST_METRIC: tests_skipped=${skipped}"

if [ "${failures}" -gt 0 ] || [ "${errors}" -gt 0 ]; then
  echo "TS_PYTEST_FAIL: tests_failed failures=${failures} errors=${errors}"
  exit 40
fi

# Before the all-skipped guard, because a pytest that failed internally -- a
# usage error, a collection error, a plugin dying at teardown -- can still write
# a zero-test report, and the guard below would then call that "nothing_executed"
# (exit 41, a selection problem) when it is really a pytest failure (exit 40).
#
# rc=5 is the one nonzero code that genuinely belongs to the guard: it is
# pytest's "no tests collected", which is exactly what an over-narrow -k or an
# all-skipped suite produces. Everything else is a failure of the run itself.
if [ "${rc}" -ne 0 ] && [ "${rc}" -ne 5 ]; then
  echo "TS_PYTEST_FAIL: pytest_rc_${rc} with no failures in report"
  exit 40
fi

# The silent-pass guard. Reached only when pytest itself was happy (rc=0) or
# said it collected nothing (rc=5), which is exactly the case that needs it.
if [ "${passed}" -lt "${MIN_PASSED}" ]; then
  echo "TS_PYTEST_FAIL: nothing_executed passed=${passed} skipped=${skipped} required=${MIN_PASSED}"
  echo "  pytest exits 0 when every test is skipped or deselected. Check the"
  echo "  suite path and -k expression, and that the platform registers these"
  echo "  solutions -- NVIDIA-only ones are absent on AMD."
  exit 41
fi

# Only rc=5 can still reach here, and only if the report credits passing tests
# that pytest says it never collected. That disagreement is not a pass.
if [ "${rc}" -ne 0 ]; then
  echo "TS_PYTEST_FAIL: pytest_rc_${rc} with no failures in report"
  exit 40
fi

echo "TS_PYTEST_OK: ${passed} passed, ${skipped} skipped in ${elapsed}s"
echo "TS_PYTEST_RESULT: pass"
exit 0
