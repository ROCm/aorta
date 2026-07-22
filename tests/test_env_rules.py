"""Unit tests for the shared env-rule predicates (``aorta._env_rules``)."""

from aorta._env_rules import ENV_KEY_RE, is_valid_env_name, value_has_nul


def test_valid_env_names():
    for name in ("FOO", "_bar", "A1", "HIP_LAUNCH_BLOCKING", "_"):
        assert is_valid_env_name(name) is True


def test_invalid_env_names():
    for name in ("1BAD", "has space", "BAD-KEY", "a.b", "", "FOO=BAR", "é"):
        assert is_valid_env_name(name) is False


def test_env_key_re_is_fullmatch_anchored():
    # A trailing newline must NOT sneak through -- fullmatch semantics.
    assert ENV_KEY_RE.fullmatch("FOO\n") is None
    assert ENV_KEY_RE.fullmatch("FOO") is not None


def test_value_has_nul():
    assert value_has_nul("plain") is False
    assert value_has_nul("") is False
    assert value_has_nul("a\x00b") is True
    assert value_has_nul("\x00") is True


def test_module_is_dependency_free():
    # The whole point of this leaf module is that any layer can import it
    # without pulling in aorta subpackages. Its own imports must be stdlib only.
    import aorta._env_rules as mod

    src = mod.__file__
    with open(src, encoding="utf-8") as fh:
        text = fh.read()
    # No ``import aorta`` / ``from aorta`` inside the leaf module.
    assert "import aorta" not in text
    assert "from aorta" not in text
