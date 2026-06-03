"""End-to-end ``aorta bundle`` + redaction integration (issue #188 Phase 3)."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from aorta.bundle import MANIFEST_FILENAME, Manifest, bundle_run_dir
from aorta.bundle.redactor import IdentityRedactor
from aorta.cli import main
from aorta.probe.bundle_hook import build_redactor_from_recipe, load_redaction_cfg
from aorta.probe.redaction import RedactingRedactor, RedactionCfg
from aorta.registry.errors import UnknownMitigationError
from aorta.triage.recipe import RecipeSchemaError, load_recipe


def _member_text(tar: tarfile.TarFile, name: str) -> str:
    """Read a tar member as UTF-8 text (``extractfile`` may return None)."""
    fh = tar.extractfile(name)
    assert fh is not None, f"member {name} not found in tarball"
    return fh.read().decode("utf-8")


def _write_trial(cell_dir: Path, trial_idx: int = 0) -> None:
    trial = cell_dir / f"trial_{trial_idx}"
    trial.mkdir(parents=True, exist_ok=True)
    (trial / "stdout.log").write_text(
        "connect 192.168.1.1 from /home/user/secret/data\n",
        encoding="utf-8",
    )
    (trial / "stderr.log").write_text("", encoding="utf-8")
    (trial / "result.json").write_text(
        json.dumps(
            {
                "verdict": "pass",
                "exit_code": 0,
                "walltime_sec": 0.1,
                "argv": ["/home/user/secret/train.py"],
                "cell_name": cell_dir.name,
                "trial_index": trial_idx,
                "env": {"AWS_TOKEN": "secret", "HIP_VISIBLE_DEVICES": "0"},
                "env_passthrough_mode": "inherit",
                "timed_out": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (trial / "probe.env").write_text(
        "AWS_TOKEN=secret\nHIP_VISIBLE_DEVICES=0\n",
        encoding="utf-8",
    )


@pytest.fixture
def redaction_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "probe-out" / "TKT-RED"
    run_dir.mkdir(parents=True)
    _write_trial(run_dir / "none-none")
    (run_dir / "host_env.json").write_text(
        json.dumps({"env": {"AWS_KEY": "x", "PATH": "/usr/bin"}}, indent=2),
        encoding="utf-8",
    )
    (run_dir / "recipe.resolved.yaml").write_text(
        """\
schema_version: 1
mode: probe
trials: 1
mitigation_axis: [none]
diagnostic_axis: [none]
redaction:
  scrub_env_keys: ["AWS_*"]
  scrub_paths: true
  scrub_ip_addresses: true
""",
        encoding="utf-8",
    )
    return run_dir


def test_refuses_no_ticket(tmp_path: Path):
    run_dir = tmp_path / "probe-out" / "_no_ticket_"
    run_dir.mkdir(parents=True)
    _write_trial(run_dir / "none-none")
    runner = CliRunner()
    result = runner.invoke(main, ["bundle", str(run_dir)])
    assert result.exit_code != 0
    assert "--ticket" in result.output


def test_review_pause_proceed(redaction_run_dir: Path, tmp_path: Path):
    runner = CliRunner()
    out = tmp_path / "bundle.tar.gz"
    result = runner.invoke(
        main,
        ["bundle", str(redaction_run_dir), "--review", "--output", str(out)],
        input="y\n",
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_review_pause_abort(redaction_run_dir: Path, tmp_path: Path):
    runner = CliRunner()
    out = tmp_path / "bundle.tar.gz"
    result = runner.invoke(
        main,
        ["bundle", str(redaction_run_dir), "--review", "--output", str(out)],
        input="n\n",
    )
    assert result.exit_code == 1
    assert not out.exists()


def test_manifest_records_per_file_counts(redaction_run_dir: Path, tmp_path: Path):
    runner = CliRunner()
    out = tmp_path / "bundle.tar.gz"
    result = runner.invoke(
        main,
        ["bundle", str(redaction_run_dir), "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    with tarfile.open(out, "r:gz") as tar:
        member = next(n for n in tar.getnames() if n.endswith(MANIFEST_FILENAME))
        manifest = Manifest.from_json(_member_text(tar, member))
    assert manifest.redaction_applied is True
    assert manifest.redactor_kind == "probe.v1"
    stdout_row = next(f for f in manifest.files if f.path.endswith("stdout.log"))
    assert stdout_row.paths_rewritten >= 1
    assert stdout_row.ips_rewritten >= 1


def test_originals_untouched(redaction_run_dir: Path, tmp_path: Path):
    before = {
        p: p.read_bytes()
        for p in redaction_run_dir.rglob("*")
        if p.is_file()
    }
    out = tmp_path / "bundle.tar.gz"
    bundle_run_dir(
        redaction_run_dir,
        output=out,
        redactor=RedactingRedactor(
            RedactionCfg(
                scrub_env_keys=("AWS_*",),
                scrub_paths=True,
                scrub_ip_addresses=True,
            )
        ),
    )
    after = {
        p: p.read_bytes()
        for p in redaction_run_dir.rglob("*")
        if p.is_file()
    }
    assert before == after


def test_fallback_redaction_resolves_despite_unresolvable_axis(tmp_path: Path):
    """recipe.resolved.yaml fallback must not require sidecars (oyazdanb review).

    A probe run driven by sidecar-defined mitigations leaves a
    recipe.resolved.yaml whose axes name mitigations the registry cannot
    resolve without the original sidecar files. Bundling only needs the
    ``redaction:`` block, so ``load_redaction_cfg`` parses just that block
    rather than running the full recipe loader (which raises here).
    """
    run_dir = tmp_path / "probe-out" / "TKT-SIDE"
    run_dir.mkdir(parents=True)
    _write_trial(run_dir / "none-none")
    recipe = run_dir / "recipe.resolved.yaml"
    recipe.write_text(
        """\
schema_version: 1
mode: probe
trials: 1
mitigation_axis: [acme_customer_only_mitigation]
diagnostic_axis: [none]
redaction:
  scrub_env_keys: ["AWS_*"]
  scrub_paths: true
  scrub_ip_addresses: true
""",
        encoding="utf-8",
    )
    # The full loader cannot resolve the sidecar-only mitigation name...
    with pytest.raises(UnknownMitigationError):
        load_recipe(recipe)
    # ...but redaction resolution succeeds because it parses only the block.
    cfg = load_redaction_cfg(recipe)
    assert cfg is not None
    assert cfg.scrub_paths is True
    redactor = build_redactor_from_recipe(None, run_dir)
    assert isinstance(redactor, RedactingRedactor)


def test_malformed_redaction_block_clean_cli_error(tmp_path: Path):
    """A bad redaction: block renders a clean CLI error, not a traceback.

    build_redactor_from_recipe raises RecipeSchemaError (a ValueError, not
    a BundleError) for a malformed block; the CLI now catches it (Copilot
    review).
    """
    run_dir = tmp_path / "probe-out" / "TKT-BAD"
    run_dir.mkdir(parents=True)
    _write_trial(run_dir / "none-none")
    (run_dir / "recipe.resolved.yaml").write_text(
        """\
schema_version: 1
mode: probe
trials: 1
mitigation_axis: [none]
diagnostic_axis: [none]
redaction:
  scrub_paths: "yes-please"
""",
        encoding="utf-8",
    )
    result = CliRunner().invoke(main, ["bundle", str(run_dir)])
    assert result.exit_code != 0
    assert "scrub_paths" in result.output
    assert not isinstance(result.exception, RecipeSchemaError)


def test_fallback_absent_recipe_uses_identity_redactor(tmp_path: Path):
    run_dir = tmp_path / "probe-out" / "TKT-NONE"
    run_dir.mkdir(parents=True)
    assert isinstance(build_redactor_from_recipe(None, run_dir), IdentityRedactor)


@pytest.mark.parametrize(
    "content",
    ["", "- a\n- b\n", "just-a-scalar\n"],
    ids=["empty", "list", "scalar"],
)
def test_non_mapping_recipe_fails_closed(tmp_path: Path, content: str):
    """A recipe that is not a top-level mapping must fail closed (Copilot).

    Returning None there would hand back an IdentityRedactor and emit an
    unredacted bundle the operator believed was scrubbed -- a fail-open. A
    valid mapping with no redaction: key still legitimately returns None.
    """
    recipe = tmp_path / "recipe.resolved.yaml"
    recipe.write_text(content, encoding="utf-8")
    with pytest.raises(RecipeSchemaError):
        load_redaction_cfg(recipe)


def test_corrupt_fallback_recipe_clean_cli_error(tmp_path: Path):
    """A non-mapping recipe.resolved.yaml fallback renders a clean CLI error."""
    run_dir = tmp_path / "probe-out" / "TKT-CORRUPT"
    run_dir.mkdir(parents=True)
    _write_trial(run_dir / "none-none")
    (run_dir / "recipe.resolved.yaml").write_text("- not-a-mapping\n", encoding="utf-8")
    result = CliRunner().invoke(main, ["bundle", str(run_dir)])
    assert result.exit_code != 0
    assert not isinstance(result.exception, RecipeSchemaError)


def test_redaction_from_auto_fallback(redaction_run_dir: Path, tmp_path: Path):
    runner = CliRunner()
    out = tmp_path / "bundle.tar.gz"
    result = runner.invoke(
        main,
        ["bundle", str(redaction_run_dir), "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    with tarfile.open(out, "r:gz") as tar:
        names = tar.getnames()
        stdout_member = next(n for n in names if n.endswith("stdout.log"))
        text = _member_text(tar, stdout_member)
    assert "/home/user" not in text
    assert "192.168.1.1" not in text
