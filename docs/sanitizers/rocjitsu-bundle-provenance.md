# Keeping ConSan results tied to a known rocjitsu build

A sanitizer verdict only means something relative to the hook that produced it.
Right now a `sanitizer_report.json` records the hook's path and SHA-256 but not
which rocjitsu commit it was built from, and there are two ways to end up running
an older hook than you think. Both were hit while verifying the fixes for
ROCm/rocm-systems#9964, #9970 and #9972.

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

**1. Stop the local cache from lingering unnoticed.** `.sanitizer-nightly/` is
now in `.gitignore`, so it no longer shows up as untracked noise that people
learn to skip past. For local runs, prefer a throwaway `--dest` per session over
a long-lived directory, and re-download rather than reuse.

**2. Record bundle provenance in the report.** The downloader already writes a
`MANIFEST.json` next to the hook containing `commit`, `run_id` and `run_url`.
Reading it in `run_consan()` and storing those three fields in the `backend` dict
alongside the existing `hook_sha256` is a small, self-contained change, and it is
what turns `hook_sha256: ca39f6c6…` from an opaque digest into something a reader
can act on. Without it, answering "which rocjitsu produced this row?" means
keeping an out-of-band digest-to-commit table.

**3. Show it on the dashboard.** Once the report carries the commit, render a
short `rocjitsu db0c47df` caption per run. That makes each row self-describing
and makes run-to-run comparisons honest.

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
