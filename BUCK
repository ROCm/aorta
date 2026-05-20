python_library(
    name = "aorta_lib",
    srcs = {
        p.removeprefix("src/"): p
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
