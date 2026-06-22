# Releasing AORTA

AORTA is distributed to customers as a versioned, `pip install`-able package.
Stable releases are published to **PyPI** (`pip install amd-aorta`) and also
attached to a [GitHub Release](https://github.com/ROCm/aorta/releases);
pre-release nightlies are published to a rolling **`dev-wheels`** pre-release.
AORTA is a pure-Python package, so a single `py3-none-any` wheel installs on
every platform; PyTorch is intentionally **not** bundled (customers install it
from the ROCm index — see below).

The release version is always read from `version` in `pyproject.toml` — it is
never hard-coded in the workflow — so cutting a new release is just a version
bump plus a trigger.

## Maintainer release flow

Releases are automated by [`.github/workflows/release.yml`](../.github/workflows/release.yml).
Each run builds the artifacts for the resolved `pyproject.toml` version and
publishes them as a new GitHub Release marked **Latest**. Each release needs a
new version: on a **manual run** the workflow refuses to release a version whose
tag already exists, and on a **tag push** Git itself rejects a tag that already
exists (force-updating an existing tag would re-release it).

Pick whichever trigger fits:

- **Manual run with a version bump (recommended).** From the GitHub UI
  (*Actions -> Release -> Run workflow*), choose a `bump` of `patch`, `minor`,
  or `major` (or type an explicit version in the `version` field). The workflow
  bumps `version` in `pyproject.toml`, commits that bump back to the branch it
  ran from, then tags and releases — so you never edit the version by hand. The
  bump is computed by [`scripts/bump_version.py`](../scripts/bump_version.py),
  which you can also run locally:

  ```bash
  python scripts/bump_version.py patch        # 0.2.0 -> 0.2.1
  python scripts/bump_version.py minor        # 0.2.0 -> 0.3.0
  python scripts/bump_version.py --set 1.4.2  # set an explicit version
  ```

- **Manual run without a bump.** Choose `bump = none` to release the current
  `pyproject.toml` version as-is (the workflow creates and pushes the matching
  `vX.Y.Z` tag for you).

- **Push a version tag** matching the current version, for a fully manual flow:

  ```bash
  git checkout main && git pull
  git tag vX.Y.Z      # X.Y.Z must equal the pyproject.toml version
  git push origin vX.Y.Z
  ```

The workflow then:

- (manual bump only) bumps `pyproject.toml` and pushes the bump commit,
- reads the version from `pyproject.toml`,
- **before building**, fails fast if a pushed tag does not match that version
  (so a release can never disagree with the package metadata),
- builds the wheel + sdist with `python -m build`,
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

> **Branch protection note.** A manual bump run pushes the version-bump commit
> to the branch it ran from. If you run it against a protected branch (e.g.
> `main`) whose rules block the `GITHUB_TOKEN`, either allow the
> `github-actions` bot to push, or bump the version in a normal PR and use the
> tag-push trigger instead.

After the workflow finishes, confirm the [latest release](https://github.com/ROCm/aorta/releases/latest)
shows `amd_aorta-X.Y.Z-py3-none-any.whl` plus the sdist, and run the customer
install command below in a clean virtualenv as a smoke test.

## Customer install flow

PyTorch is installed separately from the ROCm index (it is not part of the
wheel), so customers always install it first:

```bash
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/rocm7.1/
```

**Stable (recommended) — from PyPI:**

```bash
# Distribution name is amd-aorta (import package remains `aorta`)
pip install amd-aorta                  # latest stable
pip install "amd-aorta==X.Y.Z"         # a specific version
pip install "amd-aorta[hw-queue]"      # with optional extras
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
`X.Y.ZrcYYYYMMDD` at build time (via `scripts/bump_version.py --suffix`) and is
not committed back to the repo.

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
