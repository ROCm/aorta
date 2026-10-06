"""The optional cluster tools need an install and capability boundary in docs."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INSTALLATION = (
    ROOT / "docs" / "chat" / "installation.md"
).read_text(encoding="utf-8")
README = (ROOT / "docs" / "chat" / "README.md").read_text(encoding="utf-8")


def test_every_chat_interface_has_a_published_cia_install() -> None:
    for extra in ("chat-cli", "chat-ui", "chat-all"):
        assert f"amd-aorta[{extra},cia]" in INSTALLATION


def test_every_chat_interface_has_an_editable_cia_install() -> None:
    for extra in ("chat-cli", "chat-ui", "chat-all"):
        assert f'uv pip install -e ".[{extra},cia]"' in INSTALLATION


def test_python_ranges_and_safe_default_are_explicit() -> None:
    assert "Python 3.10–3.14" in INSTALLATION
    assert "`chat-cli` supports 3.11–3.14" in INSTALLATION
    assert "`chat-ui` and `chat-all` support 3.11–3.13" in INSTALLATION
    assert "`allow_cluster_jobs = false`" in INSTALLATION


def test_reading_and_submitting_tools_are_distinguished() -> None:
    for name in ("list_cluster_jobs", "read_autopsy_report"):
        assert name in INSTALLATION
    for name in (
        "triage_kernel_source",
        "triage_assembly_source",
        "triage_workload",
    ):
        assert name in INSTALLATION
    assert "read-only" in INSTALLATION
    assert "submit scheduler work" in INSTALLATION


def test_verification_missing_extra_and_security_links_are_present() -> None:
    assert "aorta chat tools" in INSTALLATION
    assert "aorta chat tools --json" in INSTALLATION
    assert "pip install 'amd-aorta[cia]'" in INSTALLATION
    assert "configuration.md#the-cluster-diagnostic-tools" in INSTALLATION
    assert "extending.md#the-exception-and-why-it-is-one" in INSTALLATION
    assert "redaction.md" in INSTALLATION


def test_the_chat_readme_has_the_same_entry_point() -> None:
    assert "amd-aorta[chat-cli,cia]" in README
    assert "`allow_cluster_jobs` remains `false`" in README
    assert "aorta chat tools" in README
    assert "installation.md#add-cia-backed-cluster-diagnostics" in README
