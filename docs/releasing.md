# Releasing AORTA

AORTA is distributed to customers as a versioned, `pip install`-able package.
Stable releases are published to **PyPI** (`pip install amd-aorta`) and also
attached to a [GitHub Release](https://github.com/ROCm/aorta/releases);
pre-release nightlies are published to a rolling **`dev-wheels`** pre-release.
AORTA is a pure-Python package, so a single `py3-none-any` wheel installs on
every platform; PyTorch is intentionally **not** bundled. Workloads that need
PyTorch install a matching build separately.

The version is **derived from git tags** by
[`setuptools_scm`](https://setuptools-scm.readthedocs.io/) (see
`[tool.setuptools_scm]` in `pyproject.toml`) — it is not stored in
`pyproject.toml` or `src/aorta/__init__.py`, so there is a single source of
truth (the `vX.Y.Z` tag). A build sitting on a tag becomes `X.Y.Z`; a checkout
ahead of the latest tag becomes `X.Y.(Z+1).devN+g<sha>`, so a local
`pip install .` / `pip install git+...` always reports the most recent release
(plus how far past it the tree is) rather than a stale hard-coded value.
`aorta.__version__` reads this same value back from the installed package
metadata. Cutting a release is therefore just creating a tag.

## Maintainer release flow

Releases are automated by [`.github/workflows/release.yml`](../.github/workflows/release.yml).
Each run builds the artifacts for the resolved version and publishes them as a
new GitHub Release marked **Latest**. Each release needs a new version: on a
**manual run** the workflow refuses to release a version whose tag already
exists, and on a **tag push** Git itself rejects a tag that already exists
(force-updating an existing tag would re-release it). The release is entirely
**tag-driven** — the workflow never pushes to a protected branch.

Pick whichever trigger fits:

- **Manual run (recommended).** From the GitHub UI
  (*Actions -> Release -> Run workflow*), choose a `bump` of `patch`, `minor`,
  or `major` (or type an explicit version in the `version` field). The workflow
  computes the next version above the latest tag, builds it (via
  `setuptools_scm`'s pretend-version), and — only after the build succeeds —
  creates and pushes the new `vX.Y.Z` tag. You never edit a version by hand. The
  next version is computed by [`scripts/bump_version.py`](../scripts/bump_version.py),
  which reads the latest release tag and prints the next version (it writes no
  files); you can run it locally to preview:

  ```bash
  python scripts/bump_version.py patch        # latest v0.2.0 -> 0.2.1
  python scripts/bump_version.py minor        # latest v0.2.0 -> 0.3.0
  python scripts/bump_version.py --set 1.4.2  # an explicit version
  ```

- **Push a version tag**, for a fully manual flow (the tag *is* the version):

  ```bash
  git checkout main && git pull
  git tag vX.Y.Z
  git push origin vX.Y.Z
  ```

The workflow then:

- resolves the version (from the pushed tag, or the computed next version on a
  manual run) and pins it for the build via
  `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_AMD_AORTA`,
- builds the wheel + sdist with `python -m build`,
- **after building**, fails fast if the built wheel does not carry the resolved
  version (so a release can never disagree with the package metadata),
- on a manual run, creates and pushes the `vX.Y.Z` tag (only after a successful
  build, so a failed build leaves no orphaned tag),
- creates the GitHub Release named `AORTA X.Y.Z`, marks it **Latest**, and
  uploads the wheel + sdist as release assets with auto-generated notes,
- publishes the **same** wheel + sdist to PyPI (the `publish-pypi` job reuses the
  built artifacts via Trusted Publishing — see below).

### One-time PyPI Trusted Publishing setup

PyPI publishing uses [Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
(OIDC), so there is no API token stored in the repo. Before the first stable
release, a PyPI owner must register this repo as a trusted publisher once:

1. Create (or claim) the `amd-aorta` project on PyPI.
2. In the project's *Publishing* settings, add a GitHub trusted publisher:
   owner `ROCm`, repo `aorta`, workflow `release.yml`, environment `pypi`.
3. In the GitHub repo, create an Environment named `pypi` (optionally with
   required reviewers) so the `publish-pypi` job can run.

Until this is configured the `publish-pypi` job will fail; the GitHub Release
(with installable assets) is still created by the preceding job.

> **No branch pushes.** Because the version lives in git tags (not in a tracked
> file), the release workflow only ever *creates a tag* — it never pushes a
> commit to `main`, so branch-protection rules on `main` don't block it. Ensure
> your tag protection / rulesets allow the release actor to push `v*` tags.

After the workflow finishes, confirm the [latest release](https://github.com/ROCm/aorta/releases/latest)
shows `amd_aorta-X.Y.Z-py3-none-any.whl` plus the sdist, and run the customer
install command below in a clean virtualenv as a smoke test.

## Customer install flow

Install the core package first. Add only the extras and workload dependencies
the customer needs.

**Stable (recommended) — from PyPI:**

```bash
# Distribution name is amd-aorta (import package remains `aorta`)
pip install amd-aorta                  # latest stable
pip install "amd-aorta==X.Y.Z"         # a specific version
pip install "amd-aorta[hw-queue]"      # with optional extras
```

PyTorch is not part of the AORTA wheel. Install it only when the selected
workload or feature requires it, using the PyTorch index for that ROCm release:

```bash
PYTORCH_ROCM_INDEX=https://download.pytorch.org/whl/nightly/rocmX.Y/
pip install --pre torch --index-url "$PYTORCH_ROCM_INDEX"
```

**Stable — from the GitHub Release** (no PyPI; pin to the version you want, the
newest is tagged **Latest** on the [releases page](https://github.com/ROCm/aorta/releases)):

```bash
pip install "amd-aorta @ https://github.com/ROCm/aorta/releases/download/vX.Y.Z/amd_aorta-X.Y.Z-py3-none-any.whl"
```

## Nightly / pre-release channel

[`.github/workflows/nightly.yml`](../.github/workflows/nightly.yml) builds a
release candidate from `main` every night and uploads it to a single rolling
[`dev-wheels`](https://github.com/ROCm/aorta/releases/tag/dev-wheels)
pre-release (it is never marked **Latest**). The version is stamped as
`X.Y.ZrcYYYYMMDD` at build time — where `X.Y.Z` is the **next** release above
the latest tag (e.g. latest `v0.2.0` → `0.2.1rcYYYYMMDD`, so the rc sorts as a
pre-release of the version it will become) — via `setuptools_scm`'s
pretend-version and is not committed anywhere.

Customers who need a fix before the next stable release install a specific
nightly by pointing pip at the release's asset index:

```bash
pip install "amd-aorta==X.Y.ZrcYYYYMMDD" \
    -f https://github.com/ROCm/aorta/releases/expanded_assets/dev-wheels
```

[`.github/workflows/cleanup_releases.yml`](../.github/workflows/cleanup_releases.yml)
prunes `dev-wheels` assets older than 90 days (weekly; manual runs default to a
dry run) so the rolling release stays bounded.

## Out of scope (possible follow-ups)

- Publishing to an AMD-internal PyPI index (tracked on the aorta-internal side).
- Promoting a chosen nightly rc to a stable release by rewriting the embedded
  wheel version (instead of rebuilding at tag time).
- Signing / attestation of release artifacts.
