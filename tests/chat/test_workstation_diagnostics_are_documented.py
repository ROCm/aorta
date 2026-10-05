"""The no-Slurm path is a supported setup, not an implementation secret."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUIDE = (ROOT / "docs" / "chat" / "workstation-diagnostics.md").read_text(
    encoding="utf-8"
)
CONFIGURATION = (ROOT / "docs" / "chat" / "configuration.md").read_text(
    encoding="utf-8"
)


def test_the_guide_says_slurm_is_optional() -> None:
    assert "do **not** need Slurm" in GUIDE
    assert 'cia_job_backend = "local"' in GUIDE


def test_the_guide_covers_install_verify_and_start() -> None:
    for command in (
        "pip install -e '.[chat-ui,cia]'",
        "aorta chat doctor",
        "aorta chat tools",
        "aorta chat ui",
    ):
        assert command in GUIDE


def test_all_backend_modes_are_explained() -> None:
    for backend in ("auto", "local", "slurm"):
        assert f"`{backend}`" in GUIDE
    assert "`CIA_JOB_BACKEND`" in CONFIGURATION


def test_local_execution_security_is_explicit() -> None:
    assert "runs pasted code" in GUIDE
    assert "account serving" in GUIDE
    assert "process group" in GUIDE
    assert "cluster-only SSH escalation" in GUIDE
