# Probe handout templates

Generic `mode: probe` recipes for customer-facing handouts. Each template ships
with a `redaction:` block so `aorta bundle` can produce a shareable tarball
without manual scrubbing.

## When to use which template

| Template | Customer launch pattern | Example trailing argv |
|---|---|---|
| [`probe-template-torchrun.yaml`](../../recipes/probe-template-torchrun.yaml) | PyTorch distributed / `torchrun` | `torchrun --nproc_per_node=8 train.py --config cfg.yaml` |
| [`probe-template-buck2.yaml`](../../recipes/probe-template-buck2.yaml) | Buck2 monorepo targets | `buck2 run //models:train -- --steps 1000` |
| [`probe-template-bash.yaml`](../../recipes/probe-template-bash.yaml) | Shell script / Makefile wrapper | `bash launch.sh --profile production` |

All three templates use a 2×2 mitigation × diagnostic matrix (`none` /
`tf32_off` × `none` / `xnack`), three trials, and a 30-minute
`timeout_per_trial` (1800 seconds).

## Workflow

1. Copy or symlink the template; set `ticket:` to the customer's ticket id.
2. Run probe (customer prepends `aorta probe`, keeps their argv after `--`):

   ```bash
   aorta probe --recipe recipes/probe-template-bash.yaml \
       --output ./probe-out --ticket TICKET-1234 -- \
       bash launch.sh
   ```

3. Dry-run first when validating the recipe on your side:

   ```bash
   aorta probe --recipe recipes/probe-template-bash.yaml --dry-run -- echo hi
   ```

4. After the matrix completes, bundle the ticket leaf:

   ```bash
   aorta bundle ./probe-out/TICKET-1234/ --review
   ```

   Redaction is loaded automatically from `./probe-out/TICKET-1234/recipe.resolved.yaml`
   when that file contains a `redaction:` block (the runner writes it during
   `aorta probe`).

5. Share the resulting `<ticket>-<timestamp>.tar.gz` with AMD support.

## Customising axes

Replace `mitigation_axis` / `diagnostic_axis` entries with any names registered
in `aorta.registry` (run `aorta mitigations list` on your host). The template
defaults use only mitigations present in the public registry snapshot.

See [`redaction.md`](redaction.md) for scrubber semantics and
[`usage.md`](usage.md) for env-passthrough and artifact layout.
