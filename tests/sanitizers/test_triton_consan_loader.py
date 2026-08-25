"""Unit tests for the generic Triton ConSan loader (pure logic; no GPU/ROCm needed).

Everything that touches HIP lives behind ``Hip``; these tests cover the cache
resolution, launch-ABI packing, and shim generation around it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import struct
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Shape of a real Triton 3.7.1 metadata sidecar (gfx950). Note the absence of
# ``signature``: that field is why dispatch mode needs --launch-spec.
_METADATA = {
    "hash": "66c1005ec4e1c371fab4395b6074084db48e3da68423542e611f0a14baf82c84",
    "target": {"backend": "hip", "arch": "gfx950", "warp_size": 64},
    "num_warps": 4,
    "num_stages": 2,
    "shared": 0,
    "name": "add_kernel",
}


def _load():
    path = _REPO_ROOT / "scripts" / "sanitizers" / "triton_consan_loader.py"
    spec = importlib.util.spec_from_file_location("triton_consan_loader", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations through sys.modules, so register before exec.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


loader = _load()


def _entry_dir(root: Path, *, name: str = "add_kernel", **overrides) -> Path:
    """Write one Triton-cache-shaped entry directory and return it."""

    metadata = {**_METADATA, "name": name, **overrides}
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{name}.hsaco").write_bytes(b"\x7fELF fake code object")
    (root / f"{name}.json").write_text(json.dumps(metadata), encoding="utf-8")
    # Triton also writes a sibling group index that is not kernel metadata.
    (root / f"__grp__{name}.json").write_text(json.dumps({"child_paths": {}}), encoding="utf-8")
    return root


# --------------------------------------------------------------------------
# Cache entry resolution
# --------------------------------------------------------------------------


def test_entry_from_hsaco_pairs_the_adjacent_sidecar(tmp_path):
    entry_dir = _entry_dir(tmp_path / "ENTRY")
    entry = loader.entry_from_hsaco(entry_dir / "add_kernel.hsaco")
    assert entry.kernel_name == "add_kernel"
    assert entry.arch == "gfx950"
    assert entry.metadata_path == (entry_dir / "add_kernel.json").resolve()


def test_entry_from_hsaco_without_metadata_fails_closed(tmp_path):
    tmp_path.joinpath("orphan.hsaco").write_bytes(b"\x7fELF")
    with pytest.raises(loader.LoaderError, match="no Triton metadata beside"):
        loader.entry_from_hsaco(tmp_path / "orphan.hsaco")


def test_entry_from_hsaco_missing_object_fails_closed(tmp_path):
    with pytest.raises(loader.LoaderError, match="code object not found"):
        loader.entry_from_hsaco(tmp_path / "absent.hsaco")


def test_arch_falls_back_to_flat_key(tmp_path):
    entry_dir = _entry_dir(tmp_path / "ENTRY", target={"backend": "hip"}, arch="gfx942")
    assert loader.entry_from_hsaco(entry_dir / "add_kernel.hsaco").arch == "gfx942"


def test_kernel_name_missing_fails_closed(tmp_path):
    entry_dir = tmp_path / "ENTRY"
    entry_dir.mkdir()
    (entry_dir / "k.hsaco").write_bytes(b"\x7fELF")
    (entry_dir / "k.json").write_text(json.dumps({"target": {}}), encoding="utf-8")
    entry = loader.entry_from_hsaco(entry_dir / "k.hsaco")
    with pytest.raises(loader.LoaderError, match="metadata has no 'name'"):
        _ = entry.kernel_name


def test_read_metadata_rejects_non_object(tmp_path):
    path = tmp_path / "meta.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(loader.LoaderError, match="must be a JSON object"):
        loader.read_metadata(path)


def test_discover_entries_recurses_and_skips_group_index(tmp_path):
    _entry_dir(tmp_path / "AAAA", name="mediumm_kernel")
    _entry_dir(tmp_path / "BBBB", name="largem_kernel")
    entries = loader.discover_entries(tmp_path)
    assert sorted(entry.kernel_name for entry in entries) == [
        "largem_kernel",
        "mediumm_kernel",
    ]


def test_discover_entries_on_empty_cache_fails_closed(tmp_path):
    with pytest.raises(loader.LoaderError, match="no Triton code objects under"):
        loader.discover_entries(tmp_path)


def test_discover_entries_missing_directory_fails_closed(tmp_path):
    with pytest.raises(loader.LoaderError, match="cache directory not found"):
        loader.discover_entries(tmp_path / "nope")


def test_select_entry_is_ambiguous_when_several_objects_match(tmp_path):
    """One logical Triton kernel compiles to several shape-selected objects.

    ConSan takes exactly one code object per run, so an ambiguous selection must
    fail closed and list the candidates rather than silently picking one.
    """

    _entry_dir(tmp_path / "AAAA", name="mm_kernel", hash="aaaa1111")
    _entry_dir(tmp_path / "BBBB", name="mm_kernel", hash="bbbb2222")
    entries = loader.discover_entries(tmp_path)
    with pytest.raises(loader.LoaderError, match="2 Triton cache entries match") as excinfo:
        loader.select_entry(entries)
    assert "aaaa1111" in str(excinfo.value)
    assert "bbbb2222" in str(excinfo.value)


def test_select_entry_narrows_by_name_and_hash_prefix(tmp_path):
    _entry_dir(tmp_path / "AAAA", name="mediumm_kernel", hash="aaaa1111")
    _entry_dir(tmp_path / "BBBB", name="largem_kernel", hash="bbbb2222")
    entries = loader.discover_entries(tmp_path)
    assert loader.select_entry(entries, kernel_name="largem_kernel").kernel_name == "largem_kernel"
    assert loader.select_entry(entries, cache_hash="aaaa").kernel_name == "mediumm_kernel"


def test_select_entry_without_a_match_lists_what_is_available(tmp_path):
    _entry_dir(tmp_path / "AAAA", name="mediumm_kernel")
    entries = loader.discover_entries(tmp_path)
    with pytest.raises(loader.LoaderError, match="mediumm_kernel"):
        loader.select_entry(entries, kernel_name="absent_kernel")


# --------------------------------------------------------------------------
# Launch ABI
# --------------------------------------------------------------------------


def test_parse_signature_drops_constexpr_and_keeps_order():
    specs = loader.parse_signature(
        {"x_ptr": "*fp32", "n_elements": "i32", "BLOCK_SIZE": "constexpr"}
    )
    assert [spec.name for spec in specs] == ["x_ptr", "n_elements"]
    assert specs[0].is_pointer and not specs[1].is_pointer


@pytest.mark.parametrize("bad", [["*fp32"], "*fp32", 3])
def test_parse_signature_rejects_non_mapping(bad):
    with pytest.raises(loader.LoaderError, match="signature must be a JSON object"):
        loader.parse_signature(bad)


def test_parse_signature_rejects_non_string_type():
    with pytest.raises(loader.LoaderError, match="must map to a type string"):
        loader.parse_signature({"x": 32})


def test_unsupported_argument_type_fails_closed():
    with pytest.raises(loader.LoaderError, match="unsupported Triton type"):
        _ = loader.ArgSpec(name="x", ttype="tensor").size


def test_pack_arguments_matches_the_real_add_kernel_layout():
    """3 pointers + i32, padded to the widest member -- 32 bytes on the device.

    Verified against a real gfx950 ``add_kernel`` dispatch.
    """

    specs = loader.parse_signature(
        {
            "x_ptr": "*fp32",
            "y_ptr": "*fp32",
            "out_ptr": "*fp32",
            "n_elements": "i32",
            "BLOCK_SIZE": "constexpr",
        }
    )
    packed = loader.pack_arguments(
        specs,
        pointers={"x_ptr": 0x1000, "y_ptr": 0x2000, "out_ptr": 0x3000},
        scalars={"n_elements": "1024"},
    )
    assert len(packed) == 32
    assert struct.unpack_from("<QQQi", packed) == (0x1000, 0x2000, 0x3000, 1024)


def test_pack_arguments_aligns_a_wide_scalar_after_a_narrow_one():
    specs = loader.parse_signature({"small": "i8", "wide": "i64"})
    packed = loader.pack_arguments(specs, pointers={}, scalars={"small": "7", "wide": "1"})
    # i64 must start at offset 8, not 1.
    assert len(packed) == 16
    assert struct.unpack_from("<q", packed, 8) == (1,)


def test_pack_arguments_defaults_every_scalar_to_zero():
    specs = loader.parse_signature({"n": "i32", "m": "i32"})
    assert loader.pack_arguments(specs, pointers={}, scalars={}) == b"\x00" * 8


def test_pack_arguments_supports_hex_and_negative_integers():
    specs = loader.parse_signature({"n": "i32"})
    assert loader.pack_arguments(specs, pointers={}, scalars={"n": "0x10"}) == b"\x10\x00\x00\x00"
    assert loader.pack_arguments(specs, pointers={}, scalars={"n": "-1"}) == b"\xff" * 4


def test_pack_arguments_rejects_out_of_range_and_malformed_scalars():
    specs = loader.parse_signature({"n": "i32"})
    with pytest.raises(loader.LoaderError, match="does not fit in i32"):
        loader.pack_arguments(specs, pointers={}, scalars={"n": str(2**40)})
    with pytest.raises(loader.LoaderError, match="expects an integer"):
        loader.pack_arguments(specs, pointers={}, scalars={"n": "abc"})


@pytest.mark.parametrize(
    ("ttype", "value", "expected"),
    [
        ("fp32", "1.5", struct.pack("<f", 1.5)),
        ("fp64", "1.5", struct.pack("<d", 1.5)),
        ("fp16", "1.5", struct.pack("<e", 1.5)),
        ("bf16", "1.5", b"\xc0\x3f"),
        ("bf16", "0", b"\x00\x00"),
    ],
)
def test_float_scalars_pack_to_their_device_widths(ttype, value, expected):
    specs = loader.parse_signature({"v": ttype})
    assert loader.pack_arguments(specs, pointers={}, scalars={"v": value}) == expected


def test_float_scalar_rejects_non_numeric():
    specs = loader.parse_signature({"v": "fp32"})
    with pytest.raises(loader.LoaderError, match="expects a float"):
        loader.pack_arguments(specs, pointers={}, scalars={"v": "not-a-number"})


def test_block_dim_multiplies_warps_by_warp_size():
    assert loader.block_dim(_METADATA) == 256
    assert loader.block_dim({"num_warps": 2, "warp_size": 64}) == 128


@pytest.mark.parametrize(
    ("metadata", "match"),
    [
        ({"num_warps": 4}, "warp_size"),
        ({"warp_size": 64}, "num_warps"),
        ({"num_warps": 0, "warp_size": 64}, "num_warps"),
    ],
)
def test_block_dim_fails_closed_on_incomplete_metadata(metadata, match):
    with pytest.raises(loader.LoaderError, match=match):
        loader.block_dim(metadata)


def test_shared_bytes_reads_lds_requirement():
    assert loader.shared_bytes({"shared": 16384}) == 16384
    assert loader.shared_bytes({}) == 0
    with pytest.raises(loader.LoaderError, match="non-negative integer"):
        loader.shared_bytes({"shared": -1})


@pytest.mark.parametrize("bad", ["1,1", "1,1,1,1", "a,b,c", "0,1,1"])
def test_parse_grid_rejects_malformed_input(bad):
    with pytest.raises(loader.LoaderError):
        loader.parse_grid(bad)


def test_parse_grid_accepts_three_dimensions():
    assert loader.parse_grid("4, 2,1") == (4, 2, 1)


def test_parse_arg_overrides_splits_on_first_equals():
    assert loader.parse_arg_overrides(["n=1", "s=a=b"]) == {"n": "1", "s": "a=b"}
    with pytest.raises(loader.LoaderError, match="must be 'name=value'"):
        loader.parse_arg_overrides(["bare"])


def test_resolve_signature_prefers_the_launch_spec(tmp_path):
    entry_dir = _entry_dir(tmp_path / "ENTRY", signature={"a": "i32"})
    entry = loader.entry_from_hsaco(entry_dir / "add_kernel.hsaco")
    assert loader.resolve_signature(entry, {"signature": {"b": "i64"}}) == {"b": "i64"}
    assert loader.resolve_signature(entry, None) == {"a": "i32"}


def test_resolve_signature_without_one_points_at_the_workaround(tmp_path):
    entry_dir = _entry_dir(tmp_path / "ENTRY")
    entry = loader.entry_from_hsaco(entry_dir / "add_kernel.hsaco")
    with pytest.raises(loader.LoaderError, match="--launch-spec") as excinfo:
        loader.resolve_signature(entry, None)
    assert "--mode load" in str(excinfo.value)


# --------------------------------------------------------------------------
# kernarg cross-check
# --------------------------------------------------------------------------


def test_kernarg_segment_size_read_from_the_amdgcn_listing(tmp_path):
    entry_dir = _entry_dir(tmp_path / "ENTRY")
    (entry_dir / "add_kernel.amdgcn").write_text(
        ".amdhsa_kernarg_size 48\n"
        "  .kernarg_segment_align: 8\n"
        "  .kernarg_segment_size: 48\n",
        encoding="utf-8",
    )
    entry = loader.entry_from_hsaco(entry_dir / "add_kernel.hsaco")
    assert loader.kernarg_segment_size(entry) == 48


def test_kernarg_segment_size_absent_listing_is_not_an_error(tmp_path):
    entry_dir = _entry_dir(tmp_path / "ENTRY")
    entry = loader.entry_from_hsaco(entry_dir / "add_kernel.hsaco")
    assert loader.kernarg_segment_size(entry) is None


def test_check_kernarg_fit_rejects_only_the_overrun_direction():
    # HIP does not validate the launch buffer size, so an over-long buffer is the
    # one provably-wrong case we can catch before it scribbles past the segment.
    with pytest.raises(loader.LoaderError, match="does not match this code object"):
        loader.check_kernarg_fit(56, 48, kernel="add_kernel")
    # Hidden arguments make the compiled segment larger than the explicit args.
    loader.check_kernarg_fit(32, 48, kernel="add_kernel")
    loader.check_kernarg_fit(999, None, kernel="add_kernel")


# --------------------------------------------------------------------------
# consan_command shim
# --------------------------------------------------------------------------


def test_emitted_shim_is_executable_and_self_contained(tmp_path):
    entry_dir = _entry_dir(tmp_path / "ENTRY")
    output = tmp_path / "bin" / "consan_add_kernel"
    assert (
        loader.main(
            [
                "emit-command",
                "--cache-entry",
                str(entry_dir),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert os.access(output, os.X_OK)
    assert output.stat().st_mode & stat.S_IXUSR
    body = output.read_text(encoding="utf-8")
    assert body.startswith("#!/bin/sh\n")
    # The shim must name absolute paths: ConSan runs it with no arguments and
    # from an unspecified working directory.
    assert str((entry_dir / "add_kernel.hsaco").resolve()) in body
    assert str((entry_dir / "add_kernel.json").resolve()) in body
    assert "--mode load" in body.replace(" \\\n    ", " ")


def test_shim_bakes_in_dispatch_options(tmp_path):
    entry_dir = _entry_dir(tmp_path / "ENTRY")
    output = tmp_path / "consan_dispatch"
    spec = tmp_path / "launch.json"
    spec.write_text(json.dumps({"signature": {"x": "*fp32"}}), encoding="utf-8")
    assert (
        loader.main(
            [
                "emit-command",
                "--cache-entry",
                str(entry_dir),
                "--output",
                str(output),
                "--mode",
                "dispatch",
                "--grid",
                "8,1,1",
                "--arg",
                "n_elements=1024",
                "--launch-spec",
                str(spec),
            ]
        )
        == 0
    )
    flattened = output.read_text(encoding="utf-8").replace(" \\\n    ", " ")
    assert "--mode dispatch" in flattened
    assert "--grid 8,1,1" in flattened
    assert "--arg n_elements=1024" in flattened
    assert f"--launch-spec {spec.resolve()}" in flattened


def test_shim_quotes_paths_containing_spaces(tmp_path):
    entry_dir = _entry_dir(tmp_path / "cache dir")
    output = tmp_path / "shim"
    assert loader.main(["emit-command", "--cache-entry", str(entry_dir), "--output", str(output)]) == 0
    assert "'" in output.read_text(encoding="utf-8")


def test_loader_argv_omits_dispatch_flags_in_load_mode(tmp_path):
    entry_dir = _entry_dir(tmp_path / "ENTRY")
    entry = loader.entry_from_hsaco(entry_dir / "add_kernel.hsaco")
    args = loader.build_parser().parse_args(
        ["run", "--cache-entry", str(entry_dir), "--grid", "4,1,1"]
    )
    argv = loader._loader_argv(args, entry)
    assert "--grid" not in argv
    assert argv[:2] == ["run", "--hsaco"]


# --------------------------------------------------------------------------
# CLI wiring
# --------------------------------------------------------------------------


def test_main_reports_loader_errors_without_a_traceback(tmp_path, capsys):
    assert loader.main(["list", "--cache-entry", str(tmp_path)]) == 1
    assert "triton_consan_loader:" in capsys.readouterr().err


def test_list_prints_one_row_per_code_object(tmp_path, capsys):
    _entry_dir(tmp_path / "AAAA", name="mediumm_kernel", hash="aaaa1111")
    _entry_dir(tmp_path / "BBBB", name="largem_kernel", hash="bbbb2222")
    assert loader.main(["list", "--cache-entry", str(tmp_path)]) == 0
    rows = capsys.readouterr().out.strip().splitlines()
    assert len(rows) == 2
    assert all(row.count("\t") == 3 for row in rows)


def test_hsaco_and_cache_entry_are_mutually_exclusive(tmp_path):
    with pytest.raises(SystemExit):
        loader.build_parser().parse_args(
            ["run", "--cache-entry", str(tmp_path), "--hsaco", str(tmp_path / "k.hsaco")]
        )
