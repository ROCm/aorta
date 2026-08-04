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
| `result.json`, dispatcher `trial_d*_m*_t*.json` (schema-owned env mappings, plus all string leaves) | yes | yes | yes |
| `probe.env` | yes | yes | yes |
| `host_env.json`, environment `env.json` (`env_vars` and catalog `env_overrides`) | yes | yes | yes |
| `matrix.json` (`resolved_env_vars` and `extra_env`) | yes | yes | yes |
| `recipe.resolved.yaml` (`extra_env` at recipe and cell scope, inline `environment.env`) | yes | yes | yes |
| `inline_environments.sidecar.json`, copied `sidecars/*.json` (environment / mitigation maps) | yes | yes | yes |
| `stdout.log`, `stderr.log`, `matrix.md`, other text | no | yes | yes |
| Binary / unknown extensions | no | no | no (byte copy) |

Every artifact that carries a *copy* of a cell's environment mapping is
scrubbed the same way, so a value cannot be a placeholder in one file and a
customer secret in another. `probe.env` and `recipe.resolved.yaml` are the two
non-JSON members of that set: the first is the env file handed to the child
process, the second re-emits the same `extra_env` overlay `matrix.json`
records.

### Structured JSON

Probe trials may use a flat `env: {NAME: value}` mapping, while `aorta run`
stores the full nested EnvSnapshot in dispatcher trial JSON and the process
mapping under `env.env_vars`. The redactor identifies known artifacts by path,
then locates environment mappings from that artifact's schema. This handles
mixed-type legacy flat maps without failing open and preserves nested JSON types.
Unrelated collector files merely named `env.json` or `result.json` are not
treated as platform artifacts, and an arbitrary nested object named `env_vars`
is not deleted. Their JSON *string tokens* are still scrubbed for paths and IPs,
so the semantic string is rewritten even when `/` was encoded as `\u002f` or
`\/`, and a valid document stays valid. Everything outside a string token is
copied byte for byte: number literals keep the precision and the spelling the
collector wrote, duplicate keys are not collapsed, and the document is never
re-serialized. Truncation therefore survives: a file the crash cut short is
staged scrubbed and still truncated. Only a *string token* that cannot be scanned
at all — a cut mid-string, an invalid escape — falls back to the text scrubber,
logged at WARNING. Either way the file is bundled: it owns no env mappings to
protect, and refusing it would deny a handout for the run that most needs one. Artifacts are identified by their exact path relative
to the canonical bundle run root, which `aorta bundle` supplies, so a nested
collector tree cannot spoof one and invoking bundle through a symlink does not
disable schema handling. In a *schema-owned* document — where structure is what
locates the env mappings — a parse failure or wrong top-level shape fails the
bundle closed rather than copying the file through unredacted.

`recipe.resolved.yaml` is parsed and re-emitted through
`aorta.triage.recipe`; `aorta.probe.redaction` itself stays stdlib-only per
rubric §3.F. The staged copy remains a loadable recipe — only the matched env
keys are gone. A non-mapping recipe is malformed and fails closed.

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
* Structured JSON applies path/IP rewriting to object keys as well as values.
  If two source keys would collapse to the same redacted key, bundling fails
  closed rather than silently dropping one value.

## DoS bound

Text scrubbers process input in `MAX_LOG_BYTES` (10 MiB) windows — the same
cap used by the Phase 2 classifier sandbox. A hostile log cannot force unbounded
regex work per file.

## Bundle integration

`aorta bundle` resolves the block via:

1. `--redaction-from <recipe>` when passed. An explicit path is authoritative —
   no fallback is consulted if it carries no `redaction:` block.
2. Otherwise the first of `<run-dir>/recipe.resolved.yaml` and
   `<run-dir>/matrix.json` that carries one.

`matrix.json` is the load-bearing entry for a real run: `recipe.resolved.yaml`
is emitted in the *triage* shape so it stays loadable by `load_recipe`, and
`redaction:` is a probe-mode-only key, so the resolved recipe never carries it.
`matrix.json` records the rule that was actually in force, alongside the rest of
the run-time state. The key is omitted (not `null`) when the run declared no
redaction.

When no source yields a block, `IdentityRedactor` copies bytes through (zero
counts in `manifest.json`).

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
