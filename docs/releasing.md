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

A separate job then builds the `aorta chat` retrieval index for the same tag and
attaches it to the release; see [the index asset](#the-aorta-chat-index-asset)
below.

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

### The `aorta chat` index asset

Every release also carries a prebuilt retrieval index for
[`aorta chat`](chat/README.md), so a customer who installed the wheel can run
`aorta chat index fetch` instead of embedding the tree themselves. Three assets
are attached alongside the wheel and sdist:

```text
aorta-chat-index.sqlite
aorta-chat-index.sqlite.manifest.json
aorta-chat-index.sqlite.sha256
```

`index fetch` resolves by **installed version**: an exact `X.Y.Z` fetches that
release's tag, so the index describes the code the customer actually has, while
a `.dev` build falls back to the rolling `main` asset the nightly publishes (see
below). Release assets are immutable and never expire, which is what makes the
version-matched fetch work. Unlike the nightly there is deliberately **no digest
skip** here: a release must carry its own asset even when the corpus is
byte-identical to the previous release's, because `index fetch` resolves by tag.

The index is built by a **separate `chat-index` job**, ordered after the release
job rather than being a step inside it, so it cannot delay or fail the wheel
publish or the PyPI upload. If it fails, the release still stands with the wheel
attached and only the index asset is missing.

**The corpus is restricted to git-tracked files of the public repository**, by
three overlapping guards. The index stores every chunk's source text verbatim,
so the published asset is a redistribution of the tree it was built from:

1. the job is gated on the repository being `ROCm/aorta`, so it does not run in
   a fork;
2. `aorta chat index build --public-only` re-checks that `origin` resolves to
   that repository *and* restricts the corpus to what `git ls-files` reports —
   the tracked-file half is what catches an untracked reproducer or customer
   bundle left in the working directory, which no remote check can see;
3. [`scripts/verify_chat_index_sources.py`](../scripts/verify_chat_index_sources.py)
   then scans the built index and fails the job if any chunk's recorded source
   path is not tracked.

All three raise rather than warn. The job additionally fails if the `chat-cli`
extra resolved a `torch`, `nvidia-*` or `chromadb` wheel, and scores retrieval
against the shipped question set — logged, not gated, so an immature baseline
cannot block a release.

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

The nightly also publishes the rolling `aorta chat` index built from `main`, to
the same `dev-wheels` tag and under the same three asset names as a release.
That is what `aorta chat index fetch` resolves to from a `.dev` install — which
is every editable and git install, since `setuptools_scm` stamps them that way —
and it prints how far past the last release the running build is. Two
differences from the release job are worth knowing when reading a nightly run:

- It is **ordered after** the wheel job rather than running beside it. Both write
  to `dev-wheels` and the wheel job force-moves that tag, so running them in
  parallel would race a tag move against an asset upload. A failed wheel build
  therefore publishes no index either, which is the right way round.
- It **skips the rebuild when the corpus digest is unchanged**
  (`aorta chat index digest`), because most nights touch no indexed file and
  re-uploading a byte-identical asset is churn. The digest covers the corpus
  content plus the chunking and model parameters, deliberately not the git SHA.
  A missing or unreadable published manifest counts as "no baseline" and
  rebuilds; only an explicit digest match skips.

The same three public-tree guards apply, with the repository gate additionally
narrowed to scheduled runs and manual runs from `main`.

## Out of scope (possible follow-ups)

- Publishing to an AMD-internal PyPI index (tracked on the aorta-internal side).
- Promoting a chosen nightly rc to a stable release by rewriting the embedded
  wheel version (instead of rebuilding at tag time).
- Signing / attestation of release artifacts.
