# Map src/aorta/foo.py -> aorta/foo.py so the package is importable as
# `aorta.foo` rather than `src.aorta.foo`. Use slice notation (universally
# supported in Starlark) instead of str.removeprefix, which only exists in
# newer starlark-rust revisions.
_SRC_PREFIX_LEN = len("src/")

# Non-.py package data that must ship inside the Buck-built `aorta` zip
# because runtime code resolves it as `Path(__file__).parent / ...`:
#   - aorta/ebpf/scripts/*.bt -- vendored bpftrace programs loaded by
#     aorta.ebpf.runner.SCRIPTS_DIR (see BpftraceScriptVariant.value).
#   - aorta/ebpf/scripts/PROVENANCE.md -- per-variant Heisenberg-risk
#     table referenced from the runner docstring; small enough that
#     shipping it inside the package keeps the doc reachable via
#     `python -c "import aorta.ebpf, pathlib; ..."` after pip/buck install.
_RESOURCES = {
    p[_SRC_PREFIX_LEN:]: p
    for p in (
        glob(["src/aorta/ebpf/scripts/*.bt"]) +
        ["src/aorta/ebpf/scripts/PROVENANCE.md"]
    )
}

python_library(
    name = "aorta_lib",
    srcs = {
        p[_SRC_PREFIX_LEN:]: p
        for p in glob(["src/aorta/**/*.py"])
    },
    resources = _RESOURCES,
    deps = [
        "//third-party/python:click",
        "//third-party/python:pyyaml",
    ],
    visibility = ["PUBLIC"],
)
python_binary(
    name = "aorta",
    main_function = "aorta.cli.main",
    deps = [":aorta_lib"],
    visibility = ["PUBLIC"],
)
