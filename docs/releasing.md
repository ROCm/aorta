# Releasing AORTA

AORTA is distributed to customers as a versioned, `pip install`-able artifact
attached to a [GitHub Release](https://github.com/ROCm/aorta/releases). AORTA is
a pure-Python package, so a single `py3-none-any` wheel installs on every
platform; PyTorch is intentionally **not** bundled (customers install it from
the ROCm index — see below).

The release version is always read from `version` in `pyproject.toml` — it is
never hard-coded in the workflow — so cutting a new release is just a version
bump plus a trigger.

## Maintainer release flow

Releases are automated by [`.github/workflows/release.yml`](../.github/workflows/release.yml).
Each run builds the artifacts for the resolved `pyproject.toml` version and
publishes them as a new GitHub Release. A strict `X.Y.Z` version is marked
**Latest**; a version carrying a pre-release suffix (e.g. `0.3.0rc1`) is
published as a **pre-release** and is *not* marked Latest, so the releases-page
Latest pin always points at the last stable cut. Each release needs a
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
- creates the GitHub Release named `AORTA X.Y.Z`, marks a strict `X.Y.Z` cut
  **Latest** (a pre-release suffix is published as a pre-release, not Latest),
  and uploads the wheel + sdist as release assets with auto-generated notes.

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
wheel), so customers install in two steps. Replace `X.Y.Z` with the release
version you want — browse [the releases page](https://github.com/ROCm/aorta/releases)
(the newest is tagged **Latest**) to find it:

```bash
# 1. PyTorch for the target ROCm (adjust the index URL to your ROCm version)
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/rocm7.1/

# 2. AORTA from the release (pin to the version you want)
pip install "amd-aorta @ https://github.com/ROCm/aorta/releases/download/vX.Y.Z/amd_aorta-X.Y.Z-py3-none-any.whl"
```

To install with optional extras (for example the hardware-queue tools):

```bash
pip install "amd-aorta[hw-queue] @ https://github.com/ROCm/aorta/releases/download/vX.Y.Z/amd_aorta-X.Y.Z-py3-none-any.whl"
```

## Out of scope (possible follow-ups)

- Publishing to PyPI or a private index — can be layered onto the same workflow
  later without changing the customer-facing package name.
- Signing / attestation of release artifacts.
