# Map src/aorta/foo.py -> aorta/foo.py so the package is importable as
# `aorta.foo` rather than `src.aorta.foo`. Use slice notation (universally
# supported in Starlark) instead of str.removeprefix, which only exists in
# newer starlark-rust revisions.
_SRC_PREFIX_LEN = len("src/")

python_library(
    name = "aorta_lib",
    srcs = {
        p[_SRC_PREFIX_LEN:]: p
        for p in glob(["src/aorta/**/*.py"])
    },
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
