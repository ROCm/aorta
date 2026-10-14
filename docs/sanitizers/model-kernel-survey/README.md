# Model-kernel survey snapshot

Rendered 2026-10-14 from the local waitcheck and ConSan reports for gpt-oss,
Qwen3, DeepSeek-V4, and GLM-5.3. Open `summary.md` for the tables, or
`index.html` for the dashboard page.

This is a cross-check of Tab 2, not a nightly publish. The guardrail tab is
empty because no gate reports were supplied. The reproduce lines name
`daily-<sanitizer>-<model>.yaml`; those recipes are not in this tree.
Code-object paths are filenames only.

ConSan rows still include runs from before the no-site `not_checked`
classification, so an `exit 86` here is not the final label for every kernel.
