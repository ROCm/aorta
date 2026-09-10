"""``aorta bundle`` must never package the ``aorta chat`` profile.

The profile holds an API key at rest (Decision 9b), and a bundle is a file the
operator emails to AMD. The redactor cannot cover this: its env-key removal is
driven by the recipe's ``redaction:`` globs, which say nothing about a TOML
file, so the writer refuses the file outright.

Deliberately in ``tests/bundle/`` rather than ``tests/chat/``: it is a property
of the bundle writer, it must hold on a base install where the chat extra is
absent, and it imports nothing from ``aorta.chat``.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from aorta._user_paths import CHAT_CONFIG_FILENAME
from aorta.bundle import bundle_run_dir
from aorta.bundle.manifest import MANIFEST_FILENAME
from aorta.bundle.writer import _iter_source_files


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    """A minimal tree the writer accepts as an ``aorta probe`` output."""
    root = tmp_path / "TICKET-1"
    trial = root / "cell_a" / "trial_0"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    return root


def test_the_filename_is_the_one_chat_actually_writes():
    """Guards the constant the writer matches on against a rename."""
    from aorta._user_paths import chat_config_path

    assert chat_config_path().name == CHAT_CONFIG_FILENAME


def test_a_profile_copy_at_the_top_level_is_skipped(run_dir: Path):
    (run_dir / CHAT_CONFIG_FILENAME).write_text(
        'remote_llm_api_key = "sk-must-not-ship"\n', encoding="utf-8"
    )
    collected = {p.name for p in _iter_source_files(run_dir)}
    assert CHAT_CONFIG_FILENAME not in collected


def test_a_profile_copy_nested_in_a_trial_is_skipped(run_dir: Path):
    """Matched by basename at any depth, unlike the other skips.

    The other entries in the skip set are relative-path matches, because a
    workload emitting its own ``manifest.json`` deep in the tree is a
    legitimate artifact. A credential file is not.
    """
    nested = run_dir / "cell_a" / "trial_0" / CHAT_CONFIG_FILENAME
    nested.write_text('remote_llm_api_key = "sk-must-not-ship"\n', encoding="utf-8")
    collected = set(_iter_source_files(run_dir))
    assert nested not in collected


def test_the_skip_is_logged_rather_than_silent(run_dir: Path, caplog):
    """An operator who put the file there deliberately deserves to know."""
    (run_dir / CHAT_CONFIG_FILENAME).write_text("chunk_size = 1\n", encoding="utf-8")
    with caplog.at_level("WARNING", logger="aorta.bundle.writer"):
        _iter_source_files(run_dir)
    assert any(CHAT_CONFIG_FILENAME in record.getMessage() for record in caplog.records)


def test_the_key_is_absent_from_the_written_tarball(run_dir: Path, tmp_path: Path):
    """End to end, because the skip list is not the only path into the archive."""
    (run_dir / CHAT_CONFIG_FILENAME).write_text(
        'remote_llm_api_key = "sk-must-not-ship"\n', encoding="utf-8"
    )
    output = tmp_path / "bundle.tar.gz"
    bundle_run_dir(run_dir, output=output)
    with tarfile.open(output, "r:gz") as tar:
        names = tar.getnames()
        assert not any(name.endswith(CHAT_CONFIG_FILENAME) for name in names), names
        blob = b"".join(
            tar.extractfile(name).read()
            for name in names
            if not name.endswith("/") and tar.extractfile(name) is not None
        )
    assert b"sk-must-not-ship" not in blob


class TestInTreeSymlinkAlias:
    """An alias inside the tree must not smuggle the profile past the skip.

    The escape check in ``_iter_source_files`` only rejects a link whose target
    leaves ``run_dir``; an in-tree link resolves inside it and is deliberately
    followed, so it passed that check. Matching the *walked* basename alone then
    missed it, because the alias is not called ``chat.toml`` -- and following it
    put the key in a bundle bound for a support ticket.
    """

    def test_an_alias_is_skipped(self, run_dir: Path):
        (run_dir / CHAT_CONFIG_FILENAME).write_text(
            'remote_llm_api_key = "sk-must-not-ship"\n', encoding="utf-8"
        )
        alias = run_dir / "profile-copy"
        alias.symlink_to(run_dir / CHAT_CONFIG_FILENAME)
        assert alias not in set(_iter_source_files(run_dir))

    def test_an_alias_nested_in_a_trial_is_skipped(self, run_dir: Path):
        """Depth is no more of an excuse for the alias than for the file."""
        (run_dir / CHAT_CONFIG_FILENAME).write_text(
            'remote_llm_api_key = "sk-must-not-ship"\n', encoding="utf-8"
        )
        alias = run_dir / "cell_a" / "trial_0" / "saved-settings.toml"
        alias.symlink_to(run_dir / CHAT_CONFIG_FILENAME)
        assert alias not in set(_iter_source_files(run_dir))

    def test_a_chain_of_aliases_is_skipped(self, run_dir: Path):
        """``resolve()`` collapses the chain, so the final target decides."""
        (run_dir / CHAT_CONFIG_FILENAME).write_text(
            'remote_llm_api_key = "sk-must-not-ship"\n', encoding="utf-8"
        )
        (run_dir / "link-b").symlink_to(run_dir / CHAT_CONFIG_FILENAME)
        alias = run_dir / "link-a"
        alias.symlink_to(run_dir / "link-b")
        collected = set(_iter_source_files(run_dir))
        assert alias not in collected
        assert (run_dir / "link-b") not in collected

    def test_a_directory_alias_does_not_help_either(self, run_dir: Path):
        """The profile reached through an aliased parent is still the profile."""
        real = run_dir / "cell_a" / "trial_0"
        (real / CHAT_CONFIG_FILENAME).write_text(
            'remote_llm_api_key = "sk-must-not-ship"\n', encoding="utf-8"
        )
        (run_dir / "trial-alias").symlink_to(real, target_is_directory=True)
        collected = {p.resolve().name for p in _iter_source_files(run_dir)}
        assert CHAT_CONFIG_FILENAME not in collected

    def test_the_alias_key_is_absent_from_the_written_tarball(
        self, run_dir: Path, tmp_path: Path
    ):
        """End to end: the bytes are what actually reach the support ticket."""
        (run_dir / CHAT_CONFIG_FILENAME).write_text(
            'remote_llm_api_key = "sk-must-not-ship"\n', encoding="utf-8"
        )
        (run_dir / "profile-copy").symlink_to(run_dir / CHAT_CONFIG_FILENAME)
        output = tmp_path / "bundle.tar.gz"
        bundle_run_dir(run_dir, output=output)
        with tarfile.open(output, "r:gz") as tar:
            names = tar.getnames()
            blob = b"".join(
                tar.extractfile(name).read()
                for name in names
                if not name.endswith("/") and tar.extractfile(name) is not None
            )
        assert not any(name.endswith("profile-copy") for name in names), names
        assert b"sk-must-not-ship" not in blob

    def test_the_alias_skip_names_the_target(self, run_dir: Path, caplog):
        """``profile-copy`` refused with no reason given is a puzzle, not a log."""
        (run_dir / CHAT_CONFIG_FILENAME).write_text("chunk_size = 1\n", encoding="utf-8")
        (run_dir / "profile-copy").symlink_to(run_dir / CHAT_CONFIG_FILENAME)
        with caplog.at_level("WARNING", logger="aorta.bundle.writer"):
            _iter_source_files(run_dir)
        messages = [r.getMessage() for r in caplog.records]
        assert any("profile-copy" in m and CHAT_CONFIG_FILENAME in m for m in messages), messages

    def test_an_unrelated_in_tree_symlink_is_still_followed(self, run_dir: Path):
        """The fix must not turn into a blanket refusal of in-tree links.

        Following them is documented behaviour -- only the credential filename
        is special.
        """
        target = run_dir / "cell_a" / "trial_0" / "result.json"
        alias = run_dir / "latest-result.json"
        alias.symlink_to(target)
        assert alias in set(_iter_source_files(run_dir))


def test_the_manifest_does_not_list_it_either(run_dir: Path, tmp_path: Path):
    """A manifest entry would name the file even with its bytes withheld."""
    (run_dir / CHAT_CONFIG_FILENAME).write_text("chunk_size = 1\n", encoding="utf-8")
    output = tmp_path / "bundle.tar.gz"
    bundle_run_dir(run_dir, output=output)
    with tarfile.open(output, "r:gz") as tar:
        member = next(n for n in tar.getnames() if n.endswith(MANIFEST_FILENAME))
        manifest = json.loads(tar.extractfile(member).read())
    assert all(not entry["path"].endswith(CHAT_CONFIG_FILENAME) for entry in manifest["files"])
