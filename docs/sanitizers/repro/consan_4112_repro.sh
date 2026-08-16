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
# Exit codes:
#   0  reproduced   -- object rejected with status=4112 (defect still present)
#   1  fixed        -- the transform succeeded and the module loaded. Note the
#                     process still exits 86 in that case: this driver never
#                     dispatches, so strict require-records fails afterwards. That
#                     is expected and is not a 4112 reproduction.
#   2  environment unusable (missing tool, no gfx950 bundle, hook not found)
#   3  inconclusive -- the run failed some other way (timeout, a different
#                     rejection status, ...). Read the log; deliberately NOT
#                     reported as "fixed".
set -uo pipefail

HOOK="${HSA_TOOLS_LIB:-}"
GPU="${HIP_VISIBLE_DEVICES:-0}"
WORKDIR=""
KEEP=0

usage() {
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
}

while [ $# -gt 0 ]; do
    case "$1" in
        --hook)    HOOK="${2:-}"; shift 2 ;;
        --gpu)     GPU="${2:-}"; shift 2 ;;
        --workdir) WORKDIR="${2:-}"; shift 2 ;;
        --keep)    KEEP=1; shift ;;
        -h|--help) usage ;;
        *) echo "unknown argument: $1" >&2; usage ;;
    esac
done

die() { echo "ERROR: $*" >&2; exit 2; }

[ -n "${HOOK}" ] || die "no hook given; pass --hook /path/to/librocjitsu_dbi_hooks.so"
[ -f "${HOOK}" ] || die "hook not found: ${HOOK}"

ROCM="${ROCM_PATH:-${ROCM_HOME:-/opt/rocm}}"
export PATH="${ROCM}/lib/llvm/bin:${ROCM}/bin:${PATH}"
command -v hipcc >/dev/null || die "hipcc not on PATH (looked under ${ROCM})"
command -v clang-offload-bundler >/dev/null || die "clang-offload-bundler not on PATH"

# The affected object: the heavy f32 (Type_SS) NT/Ailk_Bjlk hipBLASLt Tensile
# bundle. This is public ROCm content, so the repro carries no customer data.
BUNDLE="${ROCM}/lib/hipblaslt/library/TensileLibrary_SS_SS_HA_Bias_SAV_UA_Type_SS_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx950.co"
[ -f "${BUNDLE}" ] || die "no gfx950 f32 SS Tensile bundle at ${BUNDLE}"

if [ -z "${WORKDIR}" ]; then
    WORKDIR="$(mktemp -d)"
    [ "${KEEP}" -eq 1 ] || trap 'rm -rf "${WORKDIR}"' EXIT
fi
mkdir -p "${WORKDIR}"

HERE="$(cd "$(dirname "$0")" && pwd)"
LOADER_SRC="${HERE}/consan_4112_load.hip"
[ -f "${LOADER_SRC}" ] || die "loader source not found next to this script: ${LOADER_SRC}"

OBJECT="${WORKDIR}/consan_gemm_f32.hsaco"
LOADER="${WORKDIR}/consan_4112_load"
LOG="${WORKDIR}/hook.log"

echo "== extracting gfx950 code object from $(basename "${BUNDLE}")"
clang-offload-bundler --type=o --unbundle --input="${BUNDLE}" --output="${OBJECT}" \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx950 || die "unbundle failed"

bytes=$(stat -c%s "${OBJECT}")
kernels=$(llvm-readelf --symbols "${OBJECT}" 2>/dev/null | grep -c 'FUNC.*GLOBAL')
echo "   object: ${bytes} bytes, ${kernels} kernels"
echo "   (originally observed at 16265200 bytes / 490 kernels on ROCm 7.0.2.2)"

echo "== building load-only driver"
hipcc --offload-arch=gfx950 -DOBJECT="\"${OBJECT}\"" "${LOADER_SRC}" -o "${LOADER}" \
    >/dev/null 2>&1 || die "hipcc failed to build the loader"

echo "== running under the ConSan hook (record-replay / strict); this takes ~20 min"
echo "   MOI inventory alone is ~11 min for this object"
start=$(date +%s)
env HIP_VISIBLE_DEVICES="${GPU}" \
    HSA_TOOLS_DISABLE_REGISTER=1 \
    HSA_TOOLS_LIB="${HOOK}" \
    RJ_CONSAN_MODE=record-replay \
    RJ_CONSAN_POLICY=strict \
    RJ_CONSAN_LOG=3 \
    "${LOADER}" > "${LOG}" 2>&1
rc=$?
echo "   exit ${rc} after $(( $(date +%s) - start ))s"

echo
echo "== relevant hook output"
grep -E "MOI inventory (begin|end)|auto report plan|final validation|load rejection|loaded and instrumented" \
    "${LOG}" | cut -c1-200 || true

echo
if grep -q "status=4112" "${LOG}"; then
    echo "RESULT: reproduced -- transform rejected with status=4112"
    grep -E "final validation found" "${LOG}" | head -1
    [ "${KEEP}" -eq 1 ] && echo "log: ${LOG}"
    exit 0
fi

# Key on the loader marker, not on rc: this driver never dispatches, so once the
# transform succeeds the hook still terminates the process with exit 86 under
# strict moi_require_records ("no kernel dispatch packet was observed"). A clean
# transform therefore looks like "marker present, rc=86", not "rc=0".
if grep -q "loaded and instrumented" "${LOG}"; then
    echo "RESULT: fixed -- the object transformed and the module loaded (exit ${rc})."
    if [ "${rc}" -eq 86 ]; then
        echo "        exit 86 here is expected: strict require-records with no dispatch."
    fi
    [ "${KEEP}" -eq 1 ] && echo "log: ${LOG}"
    exit 1
fi

# No 4112 and no successful load: a different rejection status, a timeout, or a
# build/runtime failure. None of those are evidence either way, so keep the log --
# the caller needs it to tell them apart.
echo "RESULT: inconclusive -- the run failed some other way (exit ${rc}), not 4112."
echo "        Full log: ${LOG}"
trap - EXIT
exit 3
