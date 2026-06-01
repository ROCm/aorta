# `aorta bundle` — design + reference (issue #196)

> Tracking issue: [`ROCm/aorta#196`](https://github.com/ROCm/aorta/issues/196).
> This command is the prerequisite for `aorta probe` Phase 3 (issue #188).

`aorta bundle` packages a probe run directory into a single
shareable tarball, applying recipe-specified redaction via
`aorta.probe.redaction.RedactingRedactor` when the recipe (or
`<run-dir>/recipe.resolved.yaml`) includes a `redaction:` block.
Otherwise the built-in `IdentityRedactor` copies bytes through.

## CLI

```
aorta bundle <run-dir>
    [--ticket TICKET]
    [--review]
    [--output BUNDLE_PATH]
    [--redaction-from RECIPE]
```

`<run-dir>` is the per-ticket leaf written by `aorta probe`'s
`flat_resume` layout (`<probe-output>/<safe_slug(ticket)>/`). The
command does not look inside the cell directories beyond what the
redactor consumes — every file under `<run-dir>` (recursively) is
streamed through the redactor and copied into the staging tree.

### Flag reference

| Flag                  | Default                                      | Purpose                                                                                         |
|-----------------------|----------------------------------------------|-------------------------------------------------------------------------------------------------|
| `--ticket TICKET`     | inferred from `<run-dir>` basename           | Cross-check against the probe artifact tree; required when the basename is `_no_ticket_`.       |
| `--review`            | off                                          | Print the manifest summary and pause for `[y/N]` confirmation before writing the tarball.       |
| `--output PATH`       | `./<safe_slug(ticket)>-<UTC-timestamp>.tar.gz` | Where to write the bundle tarball. An *existing* directory drops the default filename inside it; any other PATH is used verbatim as the tarball filename. The ticket is slugified for filesystem safety (spaces/slashes → `_`). |
| `--redaction-from F`  | auto: `<run-dir>/recipe.resolved.yaml`       | Recipe whose `redaction:` block governs scrubbers. Explicit path overrides the auto-fallback. Without a `redaction:` block, `IdentityRedactor` runs (no scrubbing). |

### Ticket resolution

`aorta bundle` refuses to write a bundle that has no real ticket.
That guarantee comes in two halves:

1. If `<run-dir>` basename is `_no_ticket_`, the command exits
   non-zero with a `ClickException` pointing at
   `aorta probe --ticket TICKET ...`. This matches the rubric §3.B
   FR 3.1 contract for #188 Phase 3.
2. If `--ticket TICKET` is passed and the basename does not match
   `safe_slug(TICKET)`, the command **proceeds** but logs a
   warning. Operators legitimately move probe artifact trees
   between machines (e.g. NFS handoff) and the basename is the
   strict source of truth only when the operator did not override
   it.

If neither condition triggers a refusal, the resolved ticket is the
`--ticket` value (if passed) or the basename (otherwise). The
resolved value lands in the manifest's `ticket` field.

## Output layout

```
<bundle-name>.tar.gz
└── <bundle-name>/
    ├── manifest.json
    ├── recipe.resolved.yaml      # copied if present in source
    ├── matrix.md                 # copied if present in source
    ├── matrix.json               # copied if present in source
    ├── host_env.json             # copied if present in source
    └── <cell>/
        └── trial_<n>/
            ├── stdout.log
            ├── stderr.log
            ├── result.json
            └── probe.env         # only when env_passthrough_mode == 'file'
```

`<bundle-name>` defaults to `<safe_slug(ticket)>-<UTC-timestamp>`
(the ticket is slugified for filesystem safety, so spaces/slashes
become `_`) and is also the tarball's top-level directory. The manifest lives at
`bundle/manifest.json` so a downstream consumer can extract a single
file (`tar -xzOf <bundle> <bundle-name>/manifest.json`) without
unpacking the whole tree.

## Manifest schema

```json
{
  "schema_version": 1,
  "ticket": "TICKET-1234",
  "created_at": "2026-05-25T10:00:00Z",
  "aorta_version": "0.2.0",
  "source_run_dir": "TICKET-1234",
  "redaction_applied": false,
  "redactor_kind": "identity",
  "files": [
    {
      "path": "none-none/trial_0/stdout.log",
      "env_keys_removed": 0,
      "paths_rewritten": 0,
      "ips_rewritten": 0,
      "bytes_in": 12345,
      "bytes_out": 12345
    }
  ]
}
```

* `path` is **relative to `<bundle-name>/`** (matches the path the
  reader gets after `tar -xzf ...`). Forward slashes regardless of
  host OS.
* `redaction_applied` is `true` when any file had non-zero env/path/ip counts.
* `redactor_kind` is `"identity"` (no scrubbing) or `"probe.v1"` (Phase 3 redactor).
* `source_run_dir` records **only the leaf directory name** (the
  per-ticket segment), never the operator's absolute path. A bundle
  is a shareable artifact, so the full path is deliberately withheld
  to avoid leaking workstation usernames, mount points, or customer
  directory names off the source machine.

## Trust boundary: symlinks

`aorta bundle` walks the run dir and includes regular files only.
A symlink (or a symlinked parent component) whose resolved target
lands **outside** the run dir is refused with `UnsafeSymlinkError`
rather than dereferenced — otherwise a link such as
`cell/trial_0/leak -> ../../secret.txt` would pull unrelated local
bytes into a "shareable" tarball and break the trust boundary.
Symlinks whose target stays inside the run dir are still followed.

## Redactor contract

`aorta.bundle.redactor.Redactor` is an `ABC` with one method:

```python
def scrub_file(self, src: Path, dst: Path) -> RedactionCounts: ...
```

* `src` is a regular file inside the source `<run-dir>`.
* `dst` is the destination path in the staging tree. **The
  implementation creates `dst.parent` itself** -- the bundle
  writer only creates the top-level bundle root, not the
  per-file parent dirs. `IdentityRedactor.scrub_file` calls
  `dst.parent.mkdir(parents=True, exist_ok=True)` before copying;
  Phase 3's `RedactingRedactor` must do the same. (Earlier
  revisions of this doc said the parent was pre-created;
  that was wrong -- see PR #199.)
* `RedactionCounts` is a frozen dataclass with the three documented
  counters (`env_keys_removed`, `paths_rewritten`, `ips_rewritten`)
  plus `bytes_in` / `bytes_out`.

The default `IdentityRedactor` calls `shutil.copyfile(src, dst)`
and returns zeros. `RedactingRedactor` in `aorta.probe.redaction`
implements the scrubbers described in [`redaction.md`](redaction.md).
**`aorta bundle` does not own the scrubber logic** — it only invokes
the `Redactor` protocol.

## Originals are never modified

Every file in `<run-dir>` is read, never written. The staging tree
is built under a `tempfile.TemporaryDirectory`; the tarball is
written to `--output` and the staging tree is cleaned up. The
existing `aorta probe` `flat_resume` lockfile is left alone — the
bundle command does not acquire it (bundle is a read-only consumer
of the run directory).

## Errors (operator-visible)

| Class                  | Trigger                                                                |
|------------------------|------------------------------------------------------------------------|
| `RunDirNotFoundError`  | `<run-dir>` does not exist or is not a directory.                       |
| `NoTicketError`        | basename is `_no_ticket_` and `--ticket` was not passed.                |
| `EmptyRunDirError`     | `<run-dir>` exists but contains no `trial_*/result.json` artifacts.     |
| `BundleAbortedError`   | `--review` was passed and the operator answered `n`.                    |
| `BundleIOError`        | An `OSError` (permissions, ENOSPC, etc.) escaped staging or tarball writing. The original exception is preserved on `.cause`. |

All bridge to `click.ClickException` in the CLI handler so operators
see a clean error message instead of a Python traceback.

## What this command does NOT do (issue #196 out-of-scope)

* Network upload of bundles.
* Bundle decryption / unpacking utilities.
* Auto-detection of secrets beyond what the recipe's `redaction:`
  block specifies — see [`redaction.md`](redaction.md).
* Re-implementing scrubbers inside `aorta.bundle` (they live in
  `aorta.probe.redaction`).
