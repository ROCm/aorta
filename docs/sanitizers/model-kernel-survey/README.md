# Model-kernel survey snapshot

Rendered 2026-10-15 from the local waitcheck and ConSan reports for gpt-oss,
Qwen3, DeepSeek-V4, and GLM-5.3. `summary.md` is the cross-check: the workload
roll-up and one table per sanitizer with every selected kernel. `index.html`
is the same render as a page; it is gitignored, and this copy opens on the
Workload survey tab.

This is a dated cross-check of Tab 2, not a nightly publish and not a file
that CI refreshes. Nothing else in the repo points at this directory. The
guardrail tab is empty because no gate reports were supplied, which is why
`summary.md` leads with an **INCOMPLETE** banner. The reproduce lines name
`daily-<sanitizer>-<model>.yaml`; those recipes are not in this tree.
Code-object paths are filenames only.

ConSan rows still include runs from before the no-site `not_checked`
classification, so an `exit 86` here is not the final label for every kernel.

The Kernels column counts selections. One symbol compiled into two code
objects counts as two. A symbol that both waitcheck and ConSan checked counts
once, as the larger of the two reports, so the headline (163) is smaller than
the sum of the per-sanitizer tables. A Note such as "8 of 11 kernels" is that
one run's own completion line, not the Kernels column.

## Regenerating

The reports are not in the tree. The copy that produced this file is
`/tmp/san-sample/models-in` (eight `sanitizer_report.json` files, one per
case directory). `runs/model_survey/informational` is a later copy and does
not hash-match. The renderer revision recorded in `summary.md` is the parent
of the commit that added this regeneration; re-running the command below
rewrites the generator's body and drops the hand-added blockquote at the top
of `summary.md`.

```bash
python scripts/sanitizers/gen_sanitizer_dashboard.py \
  --results-dir <empty-dir> \
  --baselines recipes/sanitizers/fixtures/expected/verdict_baselines.json \
  --informational-results-dir /tmp/san-sample/models-in \
  --out-dir <out> \
  --commit "$(git rev-parse HEAD)" \
  --run-label model-kernel-survey
```

`<empty-dir>` is an empty results directory, so the three gate rows stay
"report missing". SHA-256 of the eight inputs:

| Case | SHA-256 |
|---|---|
| `consan-deepseek-v4` | `3912015cd4bc358f77a4675526ed67cb56d97bda4a02c5b45894c93451e23dff` |
| `consan-glm-5.3` | `bb6c2aadcc7cd5b33895cb651e23c7e470ce7ae44ac7dc85c5f9e0153665b592` |
| `consan-gpt-oss` | `873775f29a28347b5889ce591e55837cf86529c49f66bc23c3ec056b87040044` |
| `consan-qwen3` | `f1d530538bb7ce59d12a28429ac1536b680bea4058f6be4f10e330099d7b37cf` |
| `waitcheck-deepseek-v4` | `d708a0165449f799302fdc88c87334bb2c04d5f9c5e0de5fb1b4d22f603fde48` |
| `waitcheck-glm-5.3` | `13e4446c959c2bfda845616354fc760d471d1b00e95a6453f65da3034e18ebbd` |
| `waitcheck-gpt-oss` | `2dabdb4eb6690dcd30c3cbfa654cc7de4cc621f32b50e70168c708c150afb764` |
| `waitcheck-qwen3` | `8b873a982615bb95e5460eb14a3d10f98561e8198468c89a453c67e22b17d2a0` |
