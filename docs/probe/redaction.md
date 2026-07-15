# Redaction engine (`aorta.probe.redaction`)

> Issue [#188 Phase 3](https://github.com/ROCm/aorta/issues/188). Consumed by
> [`aorta bundle`](bundle.md) via the `Redactor` ABC.

The redaction engine scrubs probe artifacts **before** they land in a shareable
bundle tarball. Scrubbing always operates on **copies** staged under a
temporary directory; the original `<probe-output>/<ticket>/` tree is never
modified.

## Recipe block

Probe-mode recipes may include:

```yaml
redaction:
  scrub_env_keys: ["AWS_*", "GCP_*", "*_TOKEN", "*_KEY", "USER", "HOME"]
  scrub_paths: true
  scrub_ip_addresses: true
```

| Key | Type | Effect |
|---|---|---|
| `scrub_env_keys` | `list[str]` | Remove env keys matching any glob (case-sensitive `fnmatch`) |
| `scrub_paths` | `bool` | Rewrite absolute POSIX paths to `<PATH:N>` |
| `scrub_ip_addresses` | `bool` | Rewrite IPv4/IPv6 to `<IPV4:N>` / `<IPV6:N>` |

Unknown keys under `redaction:` are rejected at recipe load time.

## Where each scrubber runs

| Artifact | Env keys | Paths | IPs |
|---|---|---|---|
| `result.json` (`env`, `argv`, `capture`, string leaves) | yes | yes | yes |
| `probe.env` | yes | no | no |
| `host_env.json` (`env` block) | yes | yes | yes |
| `stdout.log`, `stderr.log`, `matrix.md`, `recipe.resolved.yaml`, other text | no | yes | yes |
| Binary / unknown extensions | no | no | no (byte copy) |

### `result.json::env`

Phase 3 adds an `env: {}` block to every probe trial's `result.json` recording
the cell's resolved mitigation + diagnostic env bundle. This is the canonical
env snapshot for bundling — it is scrubbed with `scrub_env_keys` before the
bundle is written.

## Placeholder semantics

* **Paths:** `/(?:[A-Za-z0-9_.\-]+/)+[A-Za-z0-9_.\-]+` → `<PATH:N>`. The index
  `N` deduplicates within a single file (restarts per bundled file). **No reverse
  mapping** from `<PATH:N>` back to the original path is written anywhere.
* **IPv4:** validated with `ipaddress.ip_address` before rewrite → `<IPV4:N>`.
* **IPv6:** bracketed literals (`[::1]`, `[2001:db8::1]`) and compressed
  unbracketed forms (`::1`, `fe80::1`) are matched without relying on word
  boundaries; each candidate is validated with `ipaddress` before rewrite →
  `<IPV6:N>`.
* IPv4 and IPv6 counters are summed into the manifest's `ips_rewritten` field.

## DoS bound

Text scrubbers process input in `MAX_LOG_BYTES` (10 MiB) windows — the same
cap used by the Phase 2 classifier sandbox. A hostile log cannot force unbounded
regex work per file.

## Bundle integration

`aorta bundle` resolves the recipe via:

1. `--redaction-from <recipe>` when passed, else
2. `<run-dir>/recipe.resolved.yaml` when present.

When the resolved recipe has no `redaction:` block, `IdentityRedactor` copies
bytes through (zero counts in `manifest.json`).

When a block is present, `RedactingRedactor` (`redactor_kind: "probe.v1"`) runs
and `manifest.json` records per-file counts:

```json
{
  "path": "none-none/trial_0/stdout.log",
  "env_keys_removed": 0,
  "paths_rewritten": 3,
  "ips_rewritten": 2,
  "bytes_in": 12345,
  "bytes_out": 12200
}
```

See [`bundle.md`](bundle.md) for the full manifest schema.

## Security review

> **Sign-off block (Open Question #1 from issue #188):**
>
> Redaction semantics in this document MUST be reviewed by a security owner
> outside the AORTA team before external customers run `aorta bundle` on
> real probe artifacts. Recommended reviewer: issue author (`@oyazdanb`) or
> their designated security delegate.
>
> Review checklist:
> - [ ] Env-key glob list is recipe-authoritative (no auto-detect heuristics).
> - [ ] Path/IP placeholders carry no reverse mapping.
> - [ ] Original probe tree is never modified in place.
> - [ ] `condition` sandbox (Phase 2) and redaction engine (Phase 3) reviewed together.
