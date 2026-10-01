# Contributing to AORTA

We are enthusiastic about contributions to our code and documentation. Please
feel free to file issues where documentation or functionality is lacking or,
even better, volunteer to contribute to help close these gaps!

______________________________________________________________________

> **Security vulnerabilities** — do not open a public GitHub Issue. See [SECURITY.md](SECURITY.md) for the private reporting process.

______________________________________________________________________

## Developer policies

These policies apply to all forms of activity and engagement in this project.

> [!IMPORTANT]
> AMD employees must also follow the ROCm open source software
> contributing policies at http://u.amd.com/rocm-oss-policies.

### Governance

This project is covered by the
[ROCm Project Governance](https://github.com/ROCm/ROCm/blob/develop/GOVERNANCE.md),
which also defines the code of conduct.

### Licensing

Code contributions to this project are covered under the terms of the
[LICENSE](LICENSE) file.

### Communication channels

Issue tracking, project planning, and code contributions are managed in GitHub.

______________________________________________________________________

## Development workflows

### Issue tracking

Before filing a new issue, search through
[existing issues](https://github.com/ROCm/aorta/issues) to avoid duplicates.

- If your issue is already listed, upvote it and add a comment with reproduction details.
- When in doubt, file a new issue — we'll mark duplicates accordingly.
- Provide as much information as possible: the `aorta` command and its output,
  GPU model, ROCm version, PyTorch version, and OS version. An environment
  snapshot (`aorta env probe -o env.json`, see
  [docs/env-probe.md](docs/env-probe.md)) answers most of these at once.

### Setting up

```bash
git clone https://github.com/ROCm/aorta.git
cd aorta
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
```

Optional features have their own extras (for example `chat-cli`, `cia`,
`hw-queue`); see [Installation](README.md#installation).

### Code style

black, isort and ruff are configured in [pyproject.toml](pyproject.toml). The
pre-commit hooks in [.pre-commit-config.yaml](.pre-commit-config.yaml) also run
in CI on every pull request.

### Testing

```bash
pytest tests/
```

Most of the suite runs on CPU. CI runs it on Python 3.10 through 3.14, and runs
the GPU tests on a self-hosted MI350 runner. Contributors are expected to:

- Run the tests relevant to the change and check the results.
- Add tests for new functionality and for fixed bugs.
- Document any known limitations or test exclusions.

### Pull requests

All contributions are submitted through a pull request against `main`.

- Create a feature branch; do not push directly to `main`.
- Keep changes focused, and update the documentation when behaviour,
  configuration, or the CLI changes.
- The required checks, `CPU tests` and `pytest (GPU, MI350)`, must pass before
  a PR can merge.
- The code owners in [.github/CODEOWNERS](.github/CODEOWNERS) are requested for
  review automatically. Resolve review feedback before merging.
- Use clear and descriptive commit messages.

### Security requirements

Contributors must not:

- Commit secrets, tokens, passwords, or credentials.
- Introduce vulnerable dependencies without justification.
- Bypass security controls or required security reviews.

All contributions may be subject to automated security scanning.
