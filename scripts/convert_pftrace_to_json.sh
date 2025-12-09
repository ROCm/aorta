OUTPUT_JSON_PATH=""
INPUT_PFTRACE_PATH=""
TRACECONV=""

log() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] ${message}" 2>&1
}

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -f, --pftrace PFTRACE_PATH  Path to the merged pftrace file to convert to JSON"
    echo "  -o, --output  OUTPUT_PATH   Path to the output JSON file"
    echo "  -h, --help                  Print this help message and exit"
    exit 1
}

for arg in "$@"; do
    shift
    case "$arg" in
        --pftrace)        set -- "$@" "-f" ;;
        --output)         set -- "$@" "-o" ;;
        --help)           set -- "$@" "-h" ;;
        *)                set -- "$@" "$arg";;
    esac
done

download_traceconv() {
    log "Downloading traceconv binary..."
    curl -LO https://get.perfetto.dev/traceconv
    chmod u+x ./traceconv
    TRACECONV=$(realpath ./traceconv)
    log "traceconv saved to $TRACECONV"
}

while getopts "f:o:h-:" opt; do
    case $opt in
        f)
            INPUT_PFTRACE_PATH="$OPTARG"
            ;;
        o)
            OUTPUT_JSON_PATH="$OPTARG"
            ;;
        h)
            usage
            ;;
        -)
            echo "Invalid option: --$OPTARG" >&2
            usage
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage
            ;;
    esac
done

assert_input_file_present() {
    if [ -z "$INPUT_PFTRACE_PATH" ]; then
        log "Empty pftrace path. Exiting"
        exit 1
    fi
    if [ ! -f "$INPUT_PFTRACE_PATH" ]; then
        log "Invalid pftrace filepath. Exiting"
        exit 1
    fi
}

main() {
    assert_input_file_present
    download_traceconv
    log "Generating JSON from $INPUT_PFTRACE_PATH..."
    "$TRACECONV" json "$INPUT_PFTRACE_PATH" "$OUTPUT_JSON_PATH"
    log "Generated JSON. Saved to $OUTPUT_JSON_PATH."
}

main
