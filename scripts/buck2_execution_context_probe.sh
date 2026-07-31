#!/usr/bin/env bash
#
# Empirical Buck2 execution-context probe for AORTA.
#
# This script intentionally does not create a Buck project or an RE backend.
# It runs against an existing, configured checkout so Q1/Q2 observations use
# the same prelude, toolchains, configuration, and execution platforms as the
# workload under investigation.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  buck2_execution_context_probe.sh \
    --repo-root DIR \
    --envdump-target LABEL \
    --workload-target LABEL \
    --out-dir DIR \
    [--buck2 PATH] \
    [--aorta-probe-target LABEL] \
    [--cache-bust-file RELATIVE_PATH] \
    [--context-option TYPE=VALUE]...

Target contracts:
  --envdump-target
      A repository-native action with one text output containing:
        ---ENV---
        sorted KEY=VALUE lines
        ---CGROUP---
        /proc/self/cgroup
        ---UNAME---
        uname output

  --cache-bust-file
      A repository-relative source declared by --envdump-target. The script
      temporarily changes it before each build and restores it on exit so
      local and remote measurements cannot be satisfied by the same cache hit.
      If --aorta-probe-target is used, that target must also declare this file
      as an input so its remote snapshot cannot be satisfied from stale cache.

  --aorta-probe-target
      Optional repository-native target whose output is an AORTA env.json.
      AORTA must be a declared dependency; host PYTHONPATH injection is not
      valid for a remote action.

Raw environment/config/log artifacts remain in --out-dir. The generated
q1_q2_findings.txt contains only status, hashes, and candidate variable names.
EOF
}

REPO_ROOT=""
ENVDUMP_TARGET=""
WORKLOAD_TARGET=""
OUT_DIR=""
BUCK2="buck2"
AORTA_PROBE_TARGET=""
CACHE_BUST_FILE=""
MODE_FILES=()
CONFIG_OVERRIDES=()
MODIFIERS=()
ORDERED_CONTEXT=()

while (($#)); do
    case "$1" in
        --repo-root) REPO_ROOT="${2:?missing value for --repo-root}"; shift 2 ;;
        --envdump-target) ENVDUMP_TARGET="${2:?missing value for --envdump-target}"; shift 2 ;;
        --workload-target) WORKLOAD_TARGET="${2:?missing value for --workload-target}"; shift 2 ;;
        --out-dir) OUT_DIR="${2:?missing value for --out-dir}"; shift 2 ;;
        --buck2) BUCK2="${2:?missing value for --buck2}"; shift 2 ;;
        --aorta-probe-target) AORTA_PROBE_TARGET="${2:?missing value for --aorta-probe-target}"; shift 2 ;;
        --cache-bust-file) CACHE_BUST_FILE="${2:?missing value for --cache-bust-file}"; shift 2 ;;
        --context-option) ORDERED_CONTEXT+=("${2:?missing value for --context-option}"); shift 2 ;;
        --mode-file) MODE_FILES+=("${2:?missing value for --mode-file}"); shift 2 ;;
        --config) CONFIG_OVERRIDES+=("${2:?missing value for --config}"); shift 2 ;;
        --modifier) MODIFIERS+=("${2:?missing value for --modifier}"); shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

for required in REPO_ROOT ENVDUMP_TARGET WORKLOAD_TARGET OUT_DIR; do
    if [[ -z "${!required}" ]]; then
        echo "ERROR: required argument is missing: $required" >&2
        usage >&2
        exit 2
    fi
done

if ! command -v "$BUCK2" >/dev/null 2>&1; then
    echo "ERROR: Buck2 not found: $BUCK2" >&2
    exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 is required" >&2
    exit 1
fi
if [[ ! -d "$REPO_ROOT" ]]; then
    echo "ERROR: repository root does not exist: $REPO_ROOT" >&2
    exit 1
fi

REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
chmod 700 "$OUT_DIR"
FINDINGS="$OUT_DIR/q1_q2_findings.txt"

CONTEXT_ARGS=()
OPTION_ORDER=()
if ((${#ORDERED_CONTEXT[@]})) &&
   ((${#MODE_FILES[@]} || ${#CONFIG_OVERRIDES[@]} || ${#MODIFIERS[@]})); then
    echo "ERROR: --context-option cannot be combined with grouped context options" >&2
    exit 2
fi
if ((${#ORDERED_CONTEXT[@]})); then
    for option in "${ORDERED_CONTEXT[@]}"; do
        kind="${option%%=*}"
        value="${option#*=}"
        [[ "$kind" != "$option" && -n "$value" ]] || {
            echo "ERROR: --context-option requires TYPE=VALUE" >&2
            exit 2
        }
        case "$kind" in
            mode)
                [[ "$value" == @* ]] || value="@$value"
                MODE_FILES+=("${value#@}")
                CONTEXT_ARGS+=("$value")
                ;;
            config)
                [[ "$value" == *=* ]] || {
                    echo "ERROR: config context requires config=KEY=VALUE" >&2
                    exit 2
                }
                CONFIG_OVERRIDES+=("$value")
                CONTEXT_ARGS+=("-c" "$value")
                ;;
            modifier)
                MODIFIERS+=("$value")
                CONTEXT_ARGS+=("-m" "$value")
                ;;
            *) echo "ERROR: unknown context type: $kind" >&2; exit 2 ;;
        esac
        OPTION_ORDER+=("$kind")
    done
else
    for mode in "${MODE_FILES[@]}"; do
        [[ "$mode" == @* ]] || mode="@$mode"
        CONTEXT_ARGS+=("$mode")
        OPTION_ORDER+=("mode")
    done
    for config in "${CONFIG_OVERRIDES[@]}"; do
        [[ "$config" == *=* ]] || {
            echo "ERROR: --config requires KEY=VALUE, got: $config" >&2
            exit 2
        }
        CONTEXT_ARGS+=("-c" "$config")
        OPTION_ORDER+=("config")
    done
    for modifier in "${MODIFIERS[@]}"; do
        CONTEXT_ARGS+=("-m" "$modifier")
        OPTION_ORDER+=("modifier")
    done
fi

context_fingerprint="$(
    python3 - "${CONTEXT_ARGS[@]}" <<'PY'
import hashlib
import json
import sys

payload = {"buck_argv": sys.argv[1:]}
encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
print("sha256:" + hashlib.sha256(encoded).hexdigest())
PY
)"

config_keys=()
for config in "${CONFIG_OVERRIDES[@]}"; do
    config_keys+=("${config%%=*}")
done

BUCK2_VERSION="$("$BUCK2" --version 2>/dev/null || true)"
if [[ -z "$BUCK2_VERSION" ]]; then
    echo "ERROR: could not read Buck2 version" >&2
    exit 1
fi

(
    cd "$REPO_ROOT"
    "$BUCK2" root >"$OUT_DIR/buck_root.txt"
    "$BUCK2" targets "$ENVDUMP_TARGET" >"$OUT_DIR/envdump_target.txt"
    "$BUCK2" targets "$WORKLOAD_TARGET" >"$OUT_DIR/workload_target.txt"
)
for required_output in buck_root.txt envdump_target.txt workload_target.txt; do
    [[ -s "$OUT_DIR/$required_output" ]] || {
        echo "ERROR: preflight output is empty: $required_output" >&2
        exit 1
    }
done

CACHE_BUST_ABS=""
CACHE_BUST_BACKUP=""
CACHE_BUST_EXISTED=0
if [[ -n "$CACHE_BUST_FILE" ]]; then
    [[ "$CACHE_BUST_FILE" != /* ]] || {
        echo "ERROR: --cache-bust-file must be repository-relative" >&2
        exit 2
    }
    CACHE_BUST_ABS="$REPO_ROOT/$CACHE_BUST_FILE"
    case "$(cd "$(dirname "$CACHE_BUST_ABS")" 2>/dev/null && pwd)/$(basename "$CACHE_BUST_ABS")" in
        "$REPO_ROOT"/*) ;;
        *) echo "ERROR: cache-bust file escapes repository root" >&2; exit 2 ;;
    esac
    mkdir -p "$(dirname "$CACHE_BUST_ABS")"
    CACHE_BUST_BACKUP="$OUT_DIR/cache_bust.original"
    if [[ -e "$CACHE_BUST_ABS" ]]; then
        cp "$CACHE_BUST_ABS" "$CACHE_BUST_BACKUP"
        CACHE_BUST_EXISTED=1
    fi
fi

restore_cache_bust() {
    if [[ -n "$CACHE_BUST_ABS" ]]; then
        if [[ "$CACHE_BUST_EXISTED" -eq 1 ]]; then
            cp "$CACHE_BUST_BACKUP" "$CACHE_BUST_ABS"
        else
            rm -f "$CACHE_BUST_ABS"
        fi
    fi
}
trap restore_cache_bust EXIT

write_nonce() {
    local label="$1"
    [[ -n "$CACHE_BUST_ABS" ]] || return 0
    python3 - "$label" >"$CACHE_BUST_ABS" <<'PY'
import secrets
import sys
print(f"{sys.argv[1]}:{secrets.token_hex(16)}")
PY
}

resolve_output_path() {
    local show_output="$1"
    local candidate
    candidate="$(awk 'NF >= 2 {print $NF; exit}' "$show_output")"
    [[ -n "$candidate" ]] || return 1
    if [[ "$candidate" = /* ]]; then
        printf '%s\n' "$candidate"
    else
        printf '%s\n' "$REPO_ROOT/$candidate"
    fi
}

capture_what_ran() {
    local label="$1"
    (
        cd "$REPO_ROOT"
        "$BUCK2" log what-ran
    ) >"$OUT_DIR/what_ran_${label}.txt" 2>"$OUT_DIR/what_ran_${label}.stderr" || true
}

placement_proven() {
    local label="$1"
    local expected="$2"
    local telemetry="$OUT_DIR/what_ran_${label}.txt"
    [[ -s "$telemetry" ]] || return 1
    if [[ "$expected" == "local" ]]; then
        grep -Eqi 'LocalExecute|local[_ -]?execute|"executor"[[:space:]]*:[[:space:]]*"local"' "$telemetry"
    else
        grep -Eqi 'RemoteExecute|remote[_ -]?execute|"executor"[[:space:]]*:[[:space:]]*"remote"' "$telemetry"
    fi
}

build_envdump() {
    local label="$1"
    local placement="$2"
    local show_output="$OUT_DIR/show_output_${label}.txt"
    local stderr_file="$OUT_DIR/build_${label}.stderr"
    local output_path

    write_nonce "$label"
    if ! (
        cd "$REPO_ROOT"
        "$BUCK2" build \
            "${CONTEXT_ARGS[@]}" \
            "--${placement}-only" \
            "$ENVDUMP_TARGET" \
            --show-output
    ) >"$show_output" 2>"$stderr_file"; then
        return 1
    fi
    capture_what_ran "$label"
    output_path="$(resolve_output_path "$show_output")" || return 1
    [[ -s "$output_path" ]] || return 1
    cp "$output_path" "$OUT_DIR/envdump_${label}.txt"
}

split_envdump() {
    local label="$1"
    local input="$OUT_DIR/envdump_${label}.txt"
    grep -qx -- '---ENV---' "$input"
    grep -qx -- '---CGROUP---' "$input"
    grep -qx -- '---UNAME---' "$input"
    awk '
        /^---ENV---$/ {in_env=1; next}
        /^---CGROUP---$/ {in_env=0}
        in_env {print}
    ' "$input" >"$OUT_DIR/env_${label}.txt"
    awk '
        /^---CGROUP---$/ {in_cgroup=1; next}
        /^---UNAME---$/ {in_cgroup=0}
        in_cgroup {print}
    ' "$input" >"$OUT_DIR/cgroup_${label}.txt"
    awk '
        /^---UNAME---$/ {in_uname=1; next}
        in_uname {print}
    ' "$input" >"$OUT_DIR/uname_${label}.txt"
    [[ -s "$OUT_DIR/env_${label}.txt" ]]
}

LOCAL_CAPTURED=0
REMOTE_CAPTURED=0
LOCAL_PROVEN=0
REMOTE_PROVEN=0

if build_envdump local local && split_envdump local; then
    LOCAL_CAPTURED=1
    if placement_proven local local; then
        LOCAL_PROVEN=1
    fi
fi

if [[ -n "$CACHE_BUST_ABS" ]]; then
    if build_envdump remote remote && split_envdump remote; then
        REMOTE_CAPTURED=1
        if placement_proven remote remote; then
            REMOTE_PROVEN=1
        fi
    fi
else
    printf '%s\n' \
        "Remote measurement skipped: --cache-bust-file is required to rule out a cache hit." \
        >"$OUT_DIR/remote_skipped.txt"
fi

if [[ "$LOCAL_CAPTURED" -eq 1 && "$REMOTE_CAPTURED" -eq 1 ]]; then
    diff -u "$OUT_DIR/env_local.txt" "$OUT_DIR/env_remote.txt" \
        >"$OUT_DIR/env_local_vs_remote.diff" || true
fi

CANDIDATE_MARKERS="$OUT_DIR/q1_candidate_marker_names.txt"
: >"$CANDIDATE_MARKERS"
if [[ "$REMOTE_PROVEN" -eq 1 ]]; then
    awk -F= '
        /^[A-Za-z_][A-Za-z0-9_]*=/ {
            key=toupper($1)
            if (key ~ /^(BUCK|RE_|REMOTE_|RBE_|CAS_|BUILDBARN|BUILDBUDDY|ENGFLOW)/) {
                print $1
            }
        }
    ' "$OUT_DIR/env_remote.txt" | sort -u >"$CANDIDATE_MARKERS"
fi

(
    cd "$REPO_ROOT"
    "$BUCK2" audit execution-platform-resolution \
        "${CONTEXT_ARGS[@]}" \
        "$WORKLOAD_TARGET"
) >"$OUT_DIR/execution_platform_resolution.txt" \
  2>"$OUT_DIR/execution_platform_resolution.stderr" || true

(
    cd "$REPO_ROOT"
    "$BUCK2" cquery \
        "${CONTEXT_ARGS[@]}" \
        "$WORKLOAD_TARGET" \
        --output-attribute '^buck\.execution_platform$' \
        --output-attribute '^buck\.target_configuration$' \
        --json
) >"$OUT_DIR/workload_configured_attributes.json" \
  2>"$OUT_DIR/workload_configured_attributes.stderr" || true

Q2_NAMED=0
if [[ -s "$OUT_DIR/workload_configured_attributes.json" ]] &&
   grep -q '"buck.execution_platform"' "$OUT_DIR/workload_configured_attributes.json"; then
    Q2_NAMED=1
fi

IN_ACTION_CAPTURED=0
if [[ -n "$AORTA_PROBE_TARGET" && "$REMOTE_PROVEN" -eq 1 ]]; then
    write_nonce aorta_probe
    if (
        cd "$REPO_ROOT"
        "$BUCK2" build \
            "${CONTEXT_ARGS[@]}" \
            --remote-only \
            "$AORTA_PROBE_TARGET" \
            --show-output
    ) >"$OUT_DIR/show_output_aorta_probe.txt" \
      2>"$OUT_DIR/build_aorta_probe.stderr"; then
        capture_what_ran aorta_probe
        aorta_output="$(resolve_output_path "$OUT_DIR/show_output_aorta_probe.txt" || true)"
        if placement_proven aorta_probe remote &&
           [[ -n "$aorta_output" && -s "$aorta_output" ]]; then
            cp "$aorta_output" "$OUT_DIR/in_action_env.json"
            if python3 -m json.tool "$OUT_DIR/in_action_env.json" >/dev/null; then
                IN_ACTION_CAPTURED=1
            fi
        fi
    fi
fi

{
    echo "Buck2 execution-context findings"
    echo "buck2_version=$(printf '%s' "$BUCK2_VERSION" | tr '\n' ' ')"
    echo "context_fingerprint=$context_fingerprint"
    echo "mode_file_count=${#MODE_FILES[@]}"
    echo "config_keys=$(IFS=,; echo "${config_keys[*]}")"
    echo "modifier_count=${#MODIFIERS[@]}"
    echo "option_order=$(IFS=,; echo "${OPTION_ORDER[*]}")"
    echo "local_capture=$LOCAL_CAPTURED"
    echo "local_placement_proven=$LOCAL_PROVEN"
    echo "remote_capture=$REMOTE_CAPTURED"
    echo "remote_placement_proven=$REMOTE_PROVEN"
    if [[ "$LOCAL_PROVEN" -eq 1 && "$REMOTE_PROVEN" -eq 1 ]]; then
        echo "q1_status=measured"
        echo "q1_candidate_marker_names=$(paste -sd, "$CANDIDATE_MARKERS")"
    else
        echo "q1_status=untested"
        echo "q1_candidate_marker_names="
    fi
    echo "q2_resolved_platform_named=$Q2_NAMED"
    echo "in_action_env_captured=$IN_ACTION_CAPTURED"
    echo "raw_artifacts_private=yes"
} >"$FINDINGS"

[[ -s "$FINDINGS" ]]
cat "$FINDINGS"
