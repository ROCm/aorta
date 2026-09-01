#!/usr/bin/env bash
#
# Self-contained reproducer for the rocjitsu ConSan transform rejection
# "status=4112 / partially overlapping patch ranges" on gfx950.
#
# Depends only on a ROCm install (hipcc, clang-offload-bundler, public hipBLASLt)
# and a built librocjitsu_dbi_hooks.so. Nothing from aorta is imported, so this
# file plus consan_4112_load.hip can be attached to an upstream issue as-is.
#
#   ./consan_4112_repro.sh --hook /path/to/librocjitsu_dbi_hooks.so
#
# Options:
#   --hook PATH     the ConSan hook to load (or set HSA_TOOLS_LIB)
#   --gpu N         HIP_VISIBLE_DEVICES value, default 0
#   --workdir DIR   keep intermediates here instead of a temp dir
#   --timeout SEC   wall-clock ceiling for the hooked run, default 6000 (or set
#                   CONSAN_4112_TIMEOUT). Sized against the slower of the two
#                   outcomes *on the 16,265,200-byte ROCm 7.0.2.2 object*: a hook
#                   that still has the defect rejects it after ~1420 s, but a
#                   fixed hook instruments it and runs for ~4150 s. Those numbers
#                   do not transfer to a bigger fixture, and the object a modern
#                   hipBLASLt ships is one: ~183 MiB with 9.3x the access sites,
#                   which nobody has timed. Compare the object size this script
#                   prints and raise the ceiling if it is much larger. A
#                   pre-#9964 hook never terminates MOI inventory at all, so an
#                   unbounded run would hang instead of reporting inconclusive
#   --object PATH   use an already-unbundled gfx950 code object instead of
#                   extracting one from the local hipBLASLt install
#   --keep          do not delete the work directory on exit
#
# status=4112 is a shared "transform-error" bucket, not a defect identity, so the
# verdict below discriminates on the hook's stated *reason*: only the "partially
# overlapping patch ranges" diagnostic counts as a reproduction. A 4112 from the
# patched-image growth ceiling is a capacity policy and is reported inconclusive.
#
# Exit codes:
#   0  reproduced   -- rejected with status=4112 AND the overlapping-patch-range
#                     diagnostic present (defect still present)
#   1  fixed        -- the object transformed and the module loaded, AND the hook
#                     terminated the run with its own exit 86. Both are required:
#                     this driver never dispatches, so strict require-records is
#                     expected to fail afterwards, and the loader's success marker
#                     alone would also appear if no hook had loaded at all.
#   2  environment unusable (missing tool, no gfx950 bundle, hook not found,
#                     bad --timeout value)
#   3  inconclusive -- no verdict could be established: the ceiling was hit, the
#                     hook never announced itself, the run was rejected on the
#                     patched-image growth ceiling or some other transform error,
#                     a different rejection status came back, or the module loaded
#                     without the expected exit 86. Read the log; deliberately NOT
#                     reported as "fixed" or "reproduced".
set -uo pipefail

HOOK="${HSA_TOOLS_LIB:-}"
GPU="${HIP_VISIBLE_DEVICES:-0}"
WORKDIR=""
KEEP=0
TIMEOUT="${CONSAN_4112_TIMEOUT:-6000}"
OBJECT_IN=""

# Print the header block itself rather than a hardcoded line range, so editing
# the documentation above cannot silently truncate --help.
usage() {
    awk 'NR > 1 { if (!/^#/) exit; sub(/^# ?/, ""); print }' "$0"
    exit 2
}

# `shift 2` is a no-op when only the flag itself is left, so without this check
# the loop would spin forever on a value-less trailing option.
need_value() {
    [ $# -ge 2 ] || { echo "missing value for $1" >&2; usage; }
}

while [ $# -gt 0 ]; do
    case "$1" in
        --hook)    need_value "$@"; HOOK="$2";    shift 2 ;;
        --gpu)     need_value "$@"; GPU="$2";     shift 2 ;;
        --workdir) need_value "$@"; WORKDIR="$2"; shift 2 ;;
        --timeout) need_value "$@"; TIMEOUT="$2"; shift 2 ;;
        --object)  need_value "$@"; OBJECT_IN="$2"; shift 2 ;;
        --keep)    KEEP=1; shift ;;
        -h|--help) usage ;;
        *) echo "unknown argument: $1" >&2; usage ;;
    esac
done

die() { echo "ERROR: $*" >&2; exit 2; }

[ -n "${HOOK}" ] || die "no hook given; pass --hook /path/to/librocjitsu_dbi_hooks.so"
[ -f "${HOOK}" ] || die "hook not found: ${HOOK}"
[ -z "${OBJECT_IN}" ] || [ -f "${OBJECT_IN}" ] || die "object not found: ${OBJECT_IN}"

ROCM="${ROCM_PATH:-${ROCM_HOME:-/opt/rocm}}"
export PATH="${ROCM}/lib/llvm/bin:${ROCM}/bin:${PATH}"
command -v hipcc >/dev/null || die "hipcc not on PATH (looked under ${ROCM})"
command -v timeout >/dev/null || die "timeout(1) not on PATH (coreutils)"
# Only needed for extraction; --object supplies an already-unbundled object, and
# demanding the bundler there would reject a perfectly usable invocation.
if [ -z "${OBJECT_IN}" ]; then
    command -v clang-offload-bundler >/dev/null \
        || die "clang-offload-bundler not on PATH (or pass --object to skip extraction)"
fi

case "${TIMEOUT}" in
    ''|*[!0-9]*) die "--timeout wants whole seconds, got: ${TIMEOUT}" ;;
esac
[ "${TIMEOUT}" -gt 0 ] || die "--timeout must be greater than 0"

# The affected object: the heavy f32 (Type_SS) NT/Ailk_Bjlk hipBLASLt Tensile
# bundle. This is public ROCm content, so the repro carries no customer data.
# hipBLASLt has shipped these both flat under library/ and under a per-arch
# subdirectory, so check both rather than assuming one install layout.
BUNDLE_NAME="TensileLibrary_SS_SS_HA_Bias_SAV_UA_Type_SS_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx950.co"
BUNDLE=""
if [ -z "${OBJECT_IN}" ]; then
    for candidate in \
        "${ROCM}/lib/hipblaslt/library/${BUNDLE_NAME}" \
        "${ROCM}/lib/hipblaslt/library/gfx950/${BUNDLE_NAME}"
    do
        if [ -f "${candidate}" ]; then
            BUNDLE="${candidate}"
            break
        fi
    done
    [ -n "${BUNDLE}" ] || die "no gfx950 f32 SS Tensile bundle under \
${ROCM}/lib/hipblaslt/library (looked for ${BUNDLE_NAME} flat and under gfx950/). \
Pass --object with an already-unbundled gfx950 code object to skip extraction."
fi

# Without `set -e` a failed mktemp/mkdir would leave WORKDIR empty and every
# derived path would land at the filesystem root, which a root ROCm container
# would happily create. Check both explicitly.
if [ -z "${WORKDIR}" ]; then
    WORKDIR="$(mktemp -d)" || die "mktemp -d failed"
    [ -n "${WORKDIR}" ] || die "mktemp -d produced an empty path"
    [ "${KEEP}" -eq 1 ] || trap 'rm -rf "${WORKDIR}"' EXIT
fi
mkdir -p "${WORKDIR}" || die "cannot create work directory: ${WORKDIR}"

HERE="$(cd "$(dirname "$0")" && pwd)"
LOADER_SRC="${HERE}/consan_4112_load.hip"
[ -f "${LOADER_SRC}" ] || die "loader source not found next to this script: ${LOADER_SRC}"

OBJECT="${WORKDIR}/consan_gemm_f32.hsaco"
LOADER="${WORKDIR}/consan_4112_load"
LOG="${WORKDIR}/hook.log"

if [ -n "${OBJECT_IN}" ]; then
    echo "== using caller-supplied code object $(basename "${OBJECT_IN}")"
    OBJECT="${OBJECT_IN}"
else
    echo "== extracting gfx950 code object from $(basename "${BUNDLE}")"
    clang-offload-bundler --type=o --unbundle --input="${BUNDLE}" --output="${OBJECT}" \
        --targets=hipv4-amdgcn-amd-amdhsa--gfx950 || die "unbundle failed"
fi

# The path is about to be baked into the loader as a C string literal
# (-DOBJECT below), and the loader echoes it when hipModuleLoad fails, into the
# log every verdict check below reads a line at a time. So three characters are
# refused rather than escaped:
#
#   newline      breaks one-event-per-line directly -- the tail starts at column
#                0 and can impersonate hook output past the ^ anchor
#   backslash    the compiler unescapes it, so the two characters \n in a
#                filename become that same real newline in OBJECT
#   double quote closes the literal, which is arbitrary C in the loader
#
# Checked here rather than at parse time so it covers the path built from
# --workdir as well as the one --object supplies. No legitimate code object
# needs them, so refusing is honest where escaping would be a guess.
case "${OBJECT}" in
    *$'\n'*|*\\*|*'"'*)
        die "object path must not contain a newline, backslash or double quote: ${OBJECT}" ;;
esac

bytes=$(stat -c%s "${OBJECT}")
# Kernel count is informational only, so a missing llvm-readelf should say so
# rather than silently reporting "0 kernels" and looking like a wrong object.
if command -v llvm-readelf >/dev/null; then
    kernels="$(llvm-readelf --symbols "${OBJECT}" 2>/dev/null | grep -c 'FUNC.*GLOBAL')"
else
    kernels="unknown (llvm-readelf not on PATH)"
fi
echo "   object: ${bytes} bytes, ${kernels} kernels"
echo "   (originally observed at 16265200 bytes / 490 kernels on ROCm 7.0.2.2)"

echo "== building load-only driver"
hipcc --offload-arch=gfx950 -DOBJECT="\"${OBJECT}\"" "${LOADER_SRC}" -o "${LOADER}" \
    >/dev/null 2>&1 || die "hipcc failed to build the loader"

echo "== running under the ConSan hook (record-replay / strict)"
echo "   On the 16 MB reference object: ~24 min against a hook that still has the"
echo "   defect (it is rejected at the transform), or ~69 min against a fixed one,"
echo "   which instruments it and does the work the rejection used to skip. MOI"
echo "   inventory alone is ~4-11 min either way. A larger object costs more by an"
echo "   amount nobody has measured -- compare the size printed above."
echo "   timeout ${TIMEOUT}s"
start=$(date +%s)
timeout --kill-after=30s "${TIMEOUT}" \
    env HIP_VISIBLE_DEVICES="${GPU}" \
    HSA_TOOLS_DISABLE_REGISTER=1 \
    HSA_TOOLS_LIB="${HOOK}" \
    RJ_CONSAN_MODE=record-replay \
    RJ_CONSAN_POLICY=strict \
    RJ_CONSAN_LOG=3 \
    "${LOADER}" > "${LOG}" 2>&1
rc=$?
elapsed=$(( $(date +%s) - start ))
echo "   exit ${rc} after ${elapsed}s"

# Every line the hook emits starts with this prefix, so both patterns below are
# start-anchored and every verdict grep is written as "${HOOK_LINE} ..." or
# "${LOADER_LINE} ...". Requiring the prefix is not enough on its own: the loader
# echoes the object path when hipModuleLoad fails, and the prefix is itself just
# text a filename can contain, so an --object named to look like hook output gets
# echoed into the log the verdict is read from and satisfies an unanchored match.
# The anchor is what makes it hook-owned -- only the hook can put its prefix at
# column 0. Leading whitespace is tolerated because indentation is not
# caller-controlled; anything the loader echoes has its own prefix in front.
HOOK_LINE='^[[:space:]]*\[rocjitsu-dbi-hooks\]'
LOADER_LINE='^[[:space:]]*\[consan_4112_load\]'

echo
echo "== relevant hook output"
grep -E "MOI inventory (begin|end)|auto report plan|final validation|load rejection|loaded and instrumented" \
    "${LOG}" | cut -c1-200 || true

echo
# timeout(1) reports 124 on TERM, 137 when the follow-up KILL was needed. Either
# way the run never reached a verdict, so it is inconclusive -- notably, a
# pre-#9964 hook does not terminate MOI inventory for this object at all.
if [ "${rc}" -eq 124 ] || [ "${rc}" -eq 137 ]; then
    echo "RESULT: inconclusive -- killed at the ${TIMEOUT}s ceiling, no verdict reached."
    if ! grep -qE "${HOOK_LINE} ConSan MOI inventory end" "${LOG}"; then
        echo "        MOI inventory never ended; this looks like a pre-#9964 hook."
    else
        echo "        Raise --timeout if this hook is simply slower than ${TIMEOUT}s."
    fi
    echo "        Full log: ${LOG}"
    trap - EXIT
    exit 3
fi

# Establish that the hook actually loaded before reading anything into the run.
# The loader's own marker below only proves hipModuleLoad returned success, which
# it also does with no hook at all -- so without this check a missing, unreadable
# or non-rocjitsu HSA_TOOLS_LIB produces "marker present, rc=0" and would be
# reported as "fixed" when nothing was ever instrumented.
if ! grep -qE "${HOOK_LINE} installed ConSan hook" "${LOG}"; then
    echo "RESULT: inconclusive -- the ConSan hook never announced itself."
    echo "        ${HOOK}"
    echo "        may not be a rocjitsu DBI hook, or failed to load under HSA_TOOLS_LIB."
    echo "        Full log: ${LOG}"
    trap - EXIT
    exit 3
fi

# status=4112 is a generic "transform-error" bucket, NOT a defect identity: at
# least two unrelated rejections share it. So the status alone must never decide
# the verdict -- discriminate on the hook's own explanation of *why* it rejected.
#
#   overlapping anchor ranges  -> "final validation found partially overlapping
#                                  patch ranges"  = the defect this script tests
#   patched-image growth cap   -> "first-light probe rejected patched-image file
#                                  growth"        = a capacity policy, not a defect
#
# Handle the capacity rejection first, because on any ROCm whose hipBLASLt ships a
# large Tensile bundle it is the one that fires: the extracted object is ~183 MiB
# on ROCm 7.2.4 and needs ~1.39 GiB of patched image against a ~400 MiB default
# ceiling. Reading that as "reproduced" would tell an upstream maintainer their
# fix did not work, which is the most damaging direction this script can be wrong
# in.
GROWTH_LINE='ConSan MOI first-light probe rejected patched-image file growth'
OVERLAP_LINE='ConSan final validation found partially overlapping'
if grep -qE "${HOOK_LINE} ${GROWTH_LINE}" "${LOG}"; then
    # Both diagnostics can appear in one log when it covers several loads, and
    # nothing here attributes them to the same object -- so this stays
    # inconclusive. It gets its own branch because the wording below asserts the
    # transform stopped before final validation, and the overlap diagnostic is a
    # final-validation diagnostic: printing both would state as fact something
    # this script has evidence against.
    if grep -qE "${HOOK_LINE} ${OVERLAP_LINE}" "${LOG}"; then
        echo "RESULT: inconclusive -- the log carries BOTH the patched-image growth"
        echo "        ceiling (a capacity policy) and the overlapping-patch-range"
        echo "        diagnostic (the defect). Nothing here ties them to the same object,"
        echo "        so this is not called a reproduction. Read the log."
        grep -E "${HOOK_LINE} (${GROWTH_LINE}|${OVERLAP_LINE})" "${LOG}" | head -4 | cut -c1-200
    else
        echo "RESULT: inconclusive -- rejected on the patched-image growth ceiling, which is"
        echo "        a configurable capacity policy and NOT the overlapping-patch defect."
        grep -E "${HOOK_LINE} ${GROWTH_LINE}" "${LOG}" | head -1 | cut -c1-200
        echo "        The transform never reached final validation, so this run is evidence"
        echo "        neither for nor against the defect."
    fi
    echo "        Retry with a ceiling above the required total, e.g."
    echo "          RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_PERCENT=900"
    echo "        or pass a smaller --object. Raise --timeout with it: an object big"
    echo "        enough to hit this ceiling has far more sites than the ~4150 s the"
    echo "        default was measured against, so clearing the ceiling is likely to"
    echo "        trade this verdict for a timeout at the current ${TIMEOUT}s."
    echo "        Full log: ${LOG}"
    trap - EXIT
    exit 3
fi

# Match the hook's own rejection line, its explanation, and its exit code -- not a
# bare "status=4112" substring. The loader echoes the object path when
# hipModuleLoad fails, so an --object argument whose filename contains that
# substring would otherwise be echoed back into the log and read as a
# reproduction. HOOK_LINE is start-anchored precisely so that echo cannot pass
# for hook output here.
if grep -qE "${HOOK_LINE} ConSan load rejection .*reason=transform-error .*status=4112" "${LOG}"; then
    if ! grep -qE "${HOOK_LINE} ${OVERLAP_LINE}" "${LOG}"; then
        echo "RESULT: inconclusive -- a status=4112 transform rejection, but WITHOUT the"
        echo "        overlapping-patch-range diagnostic this script tests for, and without"
        echo "        the known growth-ceiling explanation either. 4112 is a shared bucket,"
        echo "        so this is some third transform error. Read the log before concluding."
        grep -E "${HOOK_LINE} ConSan (patch end|load rejection)" "${LOG}" | tail -2 | cut -c1-200
        echo "        Full log: ${LOG}"
        trap - EXIT
        exit 3
    fi
    if [ "${rc}" -eq 92 ]; then
        echo "RESULT: reproduced -- transform rejected with status=4112 on overlapping"
        echo "        patch ranges."
        grep -E "${HOOK_LINE} ${OVERLAP_LINE}" "${LOG}" | head -1 | cut -c1-200
        [ "${KEEP}" -eq 1 ] && echo "log: ${LOG}"
        exit 0
    fi
    echo "RESULT: inconclusive -- the hook logged the 4112 overlap rejection, but the"
    echo "        process exited ${rc} rather than the 92 that strict policy should give."
    echo "        Full log: ${LOG}"
    trap - EXIT
    exit 3
fi

# A clean transform is "loader marker AND exit 86": this driver never dispatches,
# so once the transform succeeds the hook itself terminates the process under
# strict moi_require_records ("no kernel dispatch packet was observed"). Requiring
# that hook-owned exit code, rather than the marker alone, keeps "the module
# loaded for some other reason" from being read as "the defect is gone".
if grep -qE "${LOADER_LINE} loaded and instrumented" "${LOG}"; then
    if [ "${rc}" -eq 86 ]; then
        echo "RESULT: fixed -- the object transformed and the module loaded."
        echo "        exit 86 is the expected post-fix state here: strict require-records"
        echo "        with no dispatch, not a second defect."
        [ "${KEEP}" -eq 1 ] && echo "log: ${LOG}"
        exit 1
    fi
    echo "RESULT: inconclusive -- the module loaded, but the run ended with exit ${rc}"
    echo "        rather than the exit 86 strict require-records should produce here,"
    echo "        so this is not evidence that the transform defect is fixed."
    echo "        Full log: ${LOG}"
    trap - EXIT
    exit 3
fi

# No 4112 and no successful load: a different rejection status, a timeout, or a
# build/runtime failure. None of those are evidence either way, so keep the log --
# the caller needs it to tell them apart.
echo "RESULT: inconclusive -- the run failed some other way (exit ${rc}), not 4112."
echo "        Full log: ${LOG}"
trap - EXIT
exit 3
