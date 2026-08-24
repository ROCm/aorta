# Keeping ConSan results tied to a known rocjitsu build

A sanitizer verdict only means something relative to the hook that produced it.
Right now a `sanitizer_report.json` records the hook's path and SHA-256 but not
which rocjitsu commit it was built from, and there are two ways to end up running
an older hook than you think. Both were hit while verifying the upstream fixes
claimed for ROCm/rocm-systems#9964, #9970 and #9972 — a verification that only
reached a trustworthy answer *because* the stale hook was caught, and which found
#9964/#9970 fixed but #9972 unchanged.

## The two staleness paths

**1. A local prebuilt directory outlives the branch it came from.**
`resolve_consan_hook()` in
`src/aorta/instrumentation/rocjitsu_sanitizers/consan.py` prefers
`ROCJITSU_PREBUILT` over `ROCJITSU_BUILD` and returns the first hook it finds,
with no freshness check:

```python
prebuilt = os.environ.get("ROCJITSU_PREBUILT", "").strip()
if prebuilt:
    candidate = Path(prebuilt) / "lib" / "librocjitsu_dbi_hooks.so"
    if candidate.is_file():
        return candidate
```

The workspace's own `.sanitizer-nightly/rocjitsu-prebuilt/` was three commits and
several days behind the branch (`7d2c61e7`, downloaded Aug 10, versus `db0c47df`
on the branch). Anything pointing `ROCJITSU_PREBUILT` at it silently ran the
pre-fix hook and would have reported the bugs as unfixed. CI is not exposed to
this — the nightly does `rm -rf` plus `--force` before every download — so it is
purely a local-development trap, which is what makes it easy to miss.

**2. `--run latest` is a moving target.**
`download_sanitizer_artifacts.py` defaults to the newest successful run on
`shared/rocjitsu/sanitizers`. Two nightlies a week apart can therefore use
different sanitizers, and nothing in the published dashboard says so. Comparing
this week's Tab 2 against last week's can silently compare two different tools.

There is a third, time-boxed variant: these are 30-day GitHub Actions artifacts.
Run 31716952381 (the one carrying all three fixes) expires around 2026-09-12,
after which `--run latest` resolves to something else entirely.

## Recommendations, cheapest first

**1. Stop the local cache from lingering unnoticed.** Adding `.sanitizer-nightly/`
to `.gitignore` is commit hygiene only — it keeps a ~53 MB bundle plus its run
output from being committed by accident, and drops it out of `git status`. It
does **not** mitigate
staleness, and it slightly works against you: an ignored directory survives an
ordinary `git clean -fd` (you need `-x`), so the cache can now outlive a cleanup
that people assume is thorough.

The actual mitigation has to be explicit. For local runs, either re-download
before use the way CI does (`rm -rf "${dest}"` then `--force`, as in
`sanitizers-nightly.yml`), or prefer a throwaway `--dest` per session over a
long-lived directory. Reusing a directory is only safe with the freshness check
in recommendation 5.

**2. Record bundle provenance in the report.** The bundles published by the
rocjitsu build workflow ship a `MANIFEST.json` alongside `lib/` and `bin/`, and it
already carries `commit`, `run_id` and `run_url`. Note this comes from the
artifact, not from us: `download_sanitizer_artifacts.py` only extracts the zip and
verifies `sha256sums.txt`, so a bundle built without a manifest — or a hook
supplied via `ROCJITSU_BUILD` from a local source build — has no provenance to
read.

Reading it in `run_consan()` and storing those three fields in the `backend` dict
alongside the existing `hook_sha256` is a small, self-contained change, and it is
what turns `hook_sha256: ca39f6c6…` from an opaque digest into something a reader
can act on. Treat the manifest as optional input: record the fields when present
and leave them absent otherwise, rather than failing the run.

**Partly delivered by #386, at a different layer.** The nightly now copies
`commit`/`run_url` out of the bundle manifest into `rocjitsu.json`, and the
publish job exports them so the dashboard renders a `rocjitsu bundle: <commit>`
fact per run. That covers the CI reader, and it is the reason recommendation 3
below is already done. It does not replace per-report provenance: the fields
travel beside the reports rather than inside them, they describe the run as a
whole rather than each report, and a locally produced `sanitizer_report.json` —
the case where a stale `ROCJITSU_PREBUILT` actually bites, per path 1 above —
still carries nothing. Recording them in the report keeps the answer attached to
the artifact that gets copied, archived and compared.

**3. Show it on the dashboard.** ✅ Shipped in #386: `gen_sanitizer_dashboard.py`
renders the bundle commit per run in both the HTML and Markdown outputs, sourced
from `AORTA_ROCJITSU_COMMIT`. Recommendation 2 would let that caption come from
the report itself rather than from run-scoped environment.

**4. Pin `--run` for the gate.** This is the existing `#330` TODO in
`sanitizers-nightly.yml`. Pinning an explicit reviewed run ID makes the gate
immutable and turns a rocjitsu bump into a reviewed change rather than something
that happens overnight. The 30-day retention means a real pin also needs the
bundle promoted to durable storage (a GitHub Release or an internal artifact
store); pinning alone just converts silent drift into a hard failure once the
artifact expires. Worth deciding deliberately, since either outcome beats a gate
whose meaning changes without anyone noticing.

**5. Optionally, guard against staleness in dev.** If `ROCJITSU_PREBUILT`
contains a `MANIFEST.json` whose commit differs from the expected pin, warn.
A single line of output at run start would have made path 1 obvious immediately.
