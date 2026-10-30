# Sanitizers Nightly · gfx950

Run `2026-10-30T141145-41054343941` · commit `b78967101935` · 2026-10-30 14:11:45 UTC

✅ **HEALTHY** — 3/3 sanitizer outcomes match their baselines

Observed `WARN` or `FAIL` verdicts may be expected positive-control outcomes. Baseline status is the regression-health signal.

| Recipe | Backend | Baseline status | Observed | Expected | Execution | Findings | Coverage |
|---|---|---|---|---|---|--:|---|
| daily-waitcheck-gemm | waitcheck (static) | ✅ **Expected outcome** | `warn` | `warn` | complete | 64 | — |
| daily-consan-clean | consan (dynamic) | ✅ **Expected outcome** | `pass` | `pass` | complete | 0 | 0/0, 2/2 |
| daily-consan-racy | consan (dynamic) | ✅ **Expected outcome** | `fail` | `fail` | complete | 1 | 0/0, 2/2 |

Two views below: **Expected behavior (guardrails)** (baseline-checked, the gate) and **Workload survey (observed-only)** (non-gating).

## Expected behavior (guardrails) · Kernel details

<details><summary><b>daily-waitcheck-gemm</b> — ✅ **Expected outcome**</summary>

Observed sanitizer verdict `warn` · expected `warn`
Observation: waitcheck warn; 64 finding(s) (wait_hazard)
backend `rj_waitcheck` `2c406bcc3be3` · selection `top_dispatch_count` top-3 · 3 kernel(s) · execution complete

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `gemm_NT_M256_N4096_K1024` | 479 | `warn` | 32 | `sol_126578.hsaco` | `57c5d8efa4` | — |
| `gemm_NT_M128_N4096_K1280` | 471 | `warn` | 0 | `sol_175415.hsaco` | `57c5d8efa4` | same code object as gemm_NT_M256_N4096_K1024; scanned once |
| `gemm_TT_M64_N64_K1280` | 440 | `warn` | 32 | `sol_137678.hsaco` | `aeb46fded1` | — |

| Sanitizer | Code | Severity | Count | Example |
|---|---|---|--:|---|
| waitcheck | `wait_hazard` | warning | 64 | sol_126578.hsaco:gfx950[0]:.text+0x293c: missing s_waitcnt lgkmcnt(14) before use of v20 |

</details>

<details><summary><b>daily-consan-clean</b> — ✅ **Expected outcome**</summary>

Observed sanitizer verdict `pass` · expected `pass`
Observation: consan pass; preflight pass
backend `—` · selection `top_dispatch_count` top-1 · 1 kernel(s) · execution complete

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `consan_lds_race` | 1 | `pass` | 0 | `—` | `—` | — |

</details>

<details><summary><b>daily-consan-racy</b> — ✅ **Expected outcome**</summary>

Observed sanitizer verdict `fail` · expected `fail`
Observation: consan fail; 1 finding(s) (sampled_conflict); preflight pass
backend `—` · selection `top_dispatch_count` top-1 · 1 kernel(s) · execution complete

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `consan_lds_race_2wave` | 1 | `fail` | 1 | `—` | `—` | — |

| Sanitizer | Code | Severity | Count | Example |
|---|---|---|--:|---|
| consan | `sampled_conflict` | race | 1 | [rocjitsu-dbi-hooks] ConSan conflict reader=110266711188896 first_index=3 second_index=8 first_kind=2 second_kind=1 first_owner=0 second_owner=1 epoch=0 genera… |

</details>

## Workload survey (observed-only)

How real GPU kernels behave under AMD's sanitizers — **waitcheck** (static `s_waitcnt` wait-count scan) and **ConSan** (dynamic data-race check); where both produced a report the kernel is shown under each, and a scan that was skipped or whose report is missing still appears, marked report missing with no verdict. **No expected-behavior comparison on this tab**; an `error` / `fail` / `warn` here is an observation of how the kernel behaved, not a regression. Each case lists a copy-paste command to reproduce the run.

Surveyed 3 workloads (59 kernels) across 6 sanitizer runs — 2 pass · 1 warn · 1 fail · 1 error · 1 not_checked

| Workload | Kernels | waitcheck | ConSan | Findings | Note |
|---|--:|---|---|--:|---|
| gemm | 1 | `warn` | `error` | 32 | combined_hook_timeout |
| gpt oss | 55 | `pass` | `fail` | 50 | sampled_conflict |
| qwen3 | 3 | `pass` | `not_checked` | 0 | 1 of 2 kernels not checked (1x consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object) |

<details><summary><b>consan-gemm</b> — observed `error`</summary>

Observation: consan error; reason combined_hook_timeout; gemm_f32_ss: combined_hook_timeout; preflight error

Reason: `combined_hook_timeout — gemm_f32_ss: combined_hook_timeout`

Reproduce: `aorta sweep run --recipe recipes/sanitizers/daily-consan-gemm.yaml`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `gemm_f32_ss` | 1 | `error` | 0 | `consan_gemm_f32.hsaco` | `57c5d8efa4` | combined_hook_timeout |

</details>

<details><summary><b>consan-gpt-oss</b> — observed `fail`</summary>

Observation: consan fail; _post_update_kernel: consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object, _prepare_pos_seq_lens_kernel: consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object, _prepare_prefill_inputs_kernel: consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object, _rope_kernel_cached_thd_2c_gqa_onehead_fwd: consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object, _combine_sampled_and_draft_tokens_kernel: consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object, _compute_slot_mappings_kernel: consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object, _gather_block_tables_kernel: consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object, _get_num_sampled_and_rejected_kernel: consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object, _masked_compaction: consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object; 50 finding(s) (sampled_conflict); preflight pass

Finding: `[rocjitsu-dbi-hooks] ConSan conflict reader=446518800 first_index=16 second_index=16 first_kind=2 second_kind=2 first_owner=1 second_owner=1 epoch=1 generation…`

Reproduce: `python scripts/sanitizers/model_survey.py run --inventory recipes/sanitizers/models/gpt-oss.yaml --out results`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `_post_update_kernel` | 1 | `not_checked` | 0 | `dispatched.hsaco` | `d39ae3b1c4` | consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object |
| `_prepare_pos_seq_lens_kernel` | 1 | `not_checked` | 0 | `dispatched.hsaco` | `6eb8b8627f` | consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object |
| `_prepare_prefill_inputs_kernel` | 1 | `not_checked` | 0 | `dispatched.hsaco` | `fa4bd10847` | consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object |
| `_rms_norm_kernel` | 1 | `fail` | 8 | `dispatched.hsaco` | `b6dcdd21dd` | — |
| `_rope_kernel_cached_thd_2c_gqa_fwd` | 1 | `pass` | 0 | `dispatched.hsaco` | `44472db069` | — |
| `_rope_kernel_cached_thd_2c_gqa_onehead_fwd` | 1 | `not_checked` | 0 | `dispatched.hsaco` | `d5458dc446` | consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object |
| `_stage2_pow2` | 1 | `fail` | 5 | `dispatched.hsaco` | `e6acd0c33b` | — |
| `_swiglu_bf16xbf16_16x128` | 1 | `pass` | 0 | `dispatched.hsaco` | `6860ae4cbc` | — |
| `_topk_forward` | 1 | `fail` | 3 | `dispatched.hsaco` | `d46a6cefb7` | — |
| `_topk_topp_kernel` | 1 | `fail` | 9 | `dispatched.hsaco` | `b7b68ee37b` | — |
| `kernel_unified_attention_2d` | 1 | `pass` | 0 | `dispatched.hsaco` | `05e205ab8f` | — |
| `kernel_unified_attention_3d` | 1 | `pass` | 0 | `dispatched.hsaco` | `6d038c64ca` | — |
| `pack_bitmatrix` | 1 | `pass` | 0 | `dispatched.hsaco` | `62b879e054` | — |
| `reduce_segments` | 1 | `fail` | 9 | `dispatched.hsaco` | `7bcee0d9ae` | — |
| `_combine_sampled_and_draft_tokens_kernel` | 1 | `not_checked` | 0 | `dispatched.hsaco` | `44f4a73c86` | consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object |
| `_compute_slot_mappings_kernel` | 1 | `not_checked` | 0 | `dispatched.hsaco` | `597c8b991d` | consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object |
| `_fused_add_rmsnorm_kernel` | 1 | `fail` | 8 | `dispatched.hsaco` | `b7cbcd5bb8` | — |
| `_fused_add_rmsnorm_pad` | 1 | `fail` | 8 | `dispatched.hsaco` | `dc6e5bea86` | — |
| `_gather_block_tables_kernel` | 1 | `not_checked` | 0 | `dispatched.hsaco` | `ffba6285b6` | consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object |
| `_get_num_sampled_and_rejected_kernel` | 1 | `not_checked` | 0 | `dispatched.hsaco` | `cb7a82decf` | consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object |
| `_masked_compaction` | 1 | `not_checked` | 0 | `dispatched.hsaco` | `a3e32aef6c` | consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object |
| `_matmul_ogs_NNT_bf16xbf16xmxfp4_32x256x128x1_swiglu` | 1 | `pass` | 0 | `dispatched.hsaco` | `02170daa86` | — |
| `_matmul_ogs_NNT_bf16xbf16xmxfp4_32x256x128x1` | 1 | `pass` | 0 | `dispatched.hsaco` | `8b64fa68f3` | — |

</details>

<details><summary><b>consan-qwen3</b> — observed `not_checked`</summary>

Observation: consan not_checked; reason 1 of 2 kernels not checked (1x consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object); _compute_slot_mapping_kernel: consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object; preflight pass

Reason: `1 of 2 kernels not checked (1x consan_not_applicable: no LDS access, barrier, a… — _compute_slot_mapping_kernel: consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object`

Reproduce: `python scripts/sanitizers/model_survey.py run --inventory recipes/sanitizers/models/qwen3.yaml --out results`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `_fwd_kernel` | 1 | `pass` | 0 | `dispatched.hsaco` | `336ef011aa` | — |
| `_compute_slot_mapping_kernel` | 1 | `not_checked` | 0 | `dispatched.hsaco` | `2b4432007a` | consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object |

</details>

<details><summary><b>waitcheck-gemm</b> — observed `warn`</summary>

Observation: waitcheck warn; 32 finding(s) (wait_hazard)

Finding: `consan_gemm_f32.hsaco:gfx950[0]:.text+0x293c: missing s_waitcnt lgkmcnt(14) before use of v20`

Reproduce: `aorta sweep run --recipe recipes/sanitizers/daily-waitcheck-gemm-object.yaml`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `gemm_f32_ss` | 1 | `warn` | 32 | `consan_gemm_f32.hsaco` | `57c5d8efa4` | — |

</details>

<details><summary><b>waitcheck-gpt-oss</b> — observed `pass`</summary>

Observation: waitcheck pass

Reproduce: `python scripts/sanitizers/model_survey.py run --inventory recipes/sanitizers/models/gpt-oss.yaml --out results`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `_matmul_ogs_NNT_bf16xbf16xmxfp4_32x256x128x1_swiglu` | 1 | `pass` | 0 | `dispatched.hsaco` | `02170daa86` | — |
| `_rope_kernel_cached_thd_2c_gqa_onehead_fwd` | 1 | `pass` | 0 | `dispatched.hsaco` | `02f9a6e283` | — |
| `_fused_add_rmsnorm_kernel` | 1 | `pass` | 0 | `dispatched.hsaco` | `0429ecf4ed` | — |
| `kernel_unified_attention_2d` | 1 | `pass` | 0 | `dispatched.hsaco` | `05e205ab8f` | — |
| `_stage2_pow2` | 1 | `pass` | 0 | `dispatched.hsaco` | `1192e120f5` | — |
| `_fused_add_rmsnorm_kernel` | 1 | `pass` | 0 | `dispatched.hsaco` | `136cc91e21` | — |
| `_rms_norm_kernel` | 1 | `pass` | 0 | `dispatched.hsaco` | `1dcf3e3765` | — |
| `kernel_unified_attention_2d` | 1 | `pass` | 0 | `dispatched.hsaco` | `1e85e67552` | — |
| `_fused_add_rmsnorm_kernel` | 1 | `pass` | 0 | `dispatched.hsaco` | `26b4c78f52` | — |
| `_fused_add_rmsnorm_kernel` | 1 | `pass` | 0 | `dispatched.hsaco` | `28c7fc9948` | — |
| `_matmul_ogs_NNT_bf16xbf16xmxfp4_64x256x128x1` | 1 | `pass` | 0 | `dispatched.hsaco` | `357789577b` | — |
| `_rope_kernel_cached_thd_2c_gqa_fwd` | 1 | `pass` | 0 | `dispatched.hsaco` | `44472db069` | — |
| `_combine_sampled_and_draft_tokens_kernel` | 1 | `pass` | 0 | `dispatched.hsaco` | `44f4a73c86` | — |
| `_fused_add_rmsnorm_pad` | 1 | `pass` | 0 | `dispatched.hsaco` | `494126cd1a` | — |
| `_topk_topp_kernel` | 1 | `pass` | 0 | `dispatched.hsaco` | `50fe4fea9d` | — |
| `_matmul_ogs_NNT_bf16xbf16xmxfp4_32x256x128x1_swiglu` | 1 | `pass` | 0 | `dispatched.hsaco` | `5111186046` | — |
| `reduce_segments` | 1 | `pass` | 0 | `dispatched.hsaco` | `53c2bad898` | — |
| `_compute_slot_mappings_kernel` | 1 | `pass` | 0 | `dispatched.hsaco` | `597c8b991d` | — |
| `_topk_forward` | 1 | `pass` | 0 | `dispatched.hsaco` | `5c4d4d519d` | — |
| `_fused_add_rmsnorm_kernel` | 1 | `pass` | 0 | `dispatched.hsaco` | `61903965e1` | — |
| `pack_bitmatrix` | 1 | `pass` | 0 | `dispatched.hsaco` | `62b879e054` | — |
| `_rms_norm_kernel` | 1 | `pass` | 0 | `dispatched.hsaco` | `63fa64b777` | — |
| `_swiglu_bf16xbf16_16x128` | 1 | `pass` | 0 | `dispatched.hsaco` | `6860ae4cbc` | — |
| `kernel_unified_attention_2d` | 1 | `pass` | 0 | `dispatched.hsaco` | `699e09c798` | — |
| `kernel_unified_attention_2d` | 1 | `pass` | 0 | `dispatched.hsaco` | `6a27646a37` | — |
| `kernel_unified_attention_3d` | 1 | `pass` | 0 | `dispatched.hsaco` | `6d038c64ca` | — |
| `_prepare_pos_seq_lens_kernel` | 1 | `pass` | 0 | `dispatched.hsaco` | `6eb8b8627f` | — |
| `_rope_kernel_cached_thd_2c_gqa_onehead_fwd` | 1 | `pass` | 0 | `dispatched.hsaco` | `6f97774106` | — |
| `kernel_unified_attention_2d` | 1 | `pass` | 0 | `dispatched.hsaco` | `7433030bb8` | — |
| `reduce_segments` | 1 | `pass` | 0 | `dispatched.hsaco` | `7bcee0d9ae` | — |
| `pack_bitmatrix` | 1 | `pass` | 0 | `dispatched.hsaco` | `7da1e46e4d` | — |
| `_matmul_ogs_NNT_bf16xbf16xmxfp4_32x256x128x1` | 1 | `pass` | 0 | `dispatched.hsaco` | `8b64fa68f3` | — |
| `_rope_kernel_cached_thd_2c_gqa_fwd` | 1 | `pass` | 0 | `dispatched.hsaco` | `8f39bbd1bd` | — |
| `_matmul_ogs_NNT_bf16xbf16xmxfp4_64x256x128x1_swiglu` | 1 | `pass` | 0 | `dispatched.hsaco` | `99783dba6b` | — |
| `_masked_compaction` | 1 | `pass` | 0 | `dispatched.hsaco` | `a3e32aef6c` | — |
| `kernel_unified_attention_2d` | 1 | `pass` | 0 | `dispatched.hsaco` | `a5497ec8d0` | — |
| `_matmul_ogs_NNT_bf16xbf16xmxfp4_32x256x128x1` | 1 | `pass` | 0 | `dispatched.hsaco` | `b60dc71885` | — |
| `_rms_norm_kernel` | 1 | `pass` | 0 | `dispatched.hsaco` | `b6dcdd21dd` | — |
| `_topk_topp_kernel` | 1 | `pass` | 0 | `dispatched.hsaco` | `b7b68ee37b` | — |
| `_fused_add_rmsnorm_kernel` | 1 | `pass` | 0 | `dispatched.hsaco` | `b7cbcd5bb8` | — |
| `_get_num_sampled_and_rejected_kernel` | 1 | `pass` | 0 | `dispatched.hsaco` | `cb7a82decf` | — |
| `_rms_norm_kernel` | 1 | `pass` | 0 | `dispatched.hsaco` | `cca3ceda2f` | — |
| `_post_update_kernel` | 1 | `pass` | 0 | `dispatched.hsaco` | `d39ae3b1c4` | — |
| `_topk_forward` | 1 | `pass` | 0 | `dispatched.hsaco` | `d46a6cefb7` | — |
| `_rope_kernel_cached_thd_2c_gqa_onehead_fwd` | 1 | `pass` | 0 | `dispatched.hsaco` | `d5458dc446` | — |
| `_fused_add_rmsnorm_pad` | 1 | `pass` | 0 | `dispatched.hsaco` | `dc6e5bea86` | — |
| `pack_bitmatrix` | 1 | `pass` | 0 | `dispatched.hsaco` | `de085311a4` | — |
| `_fused_add_rmsnorm_kernel` | 1 | `pass` | 0 | `dispatched.hsaco` | `df48165cc5` | — |
| `_rms_norm_kernel` | 1 | `pass` | 0 | `dispatched.hsaco` | `e189a11509` | — |
| `_rms_norm_kernel` | 1 | `pass` | 0 | `dispatched.hsaco` | `e5d86d5d3e` | — |
| `_stage2_pow2` | 1 | `pass` | 0 | `dispatched.hsaco` | `e6acd0c33b` | — |
| `_rms_norm_kernel` | 1 | `pass` | 0 | `dispatched.hsaco` | `ec7c47196f` | — |
| `kernel_unified_attention_3d` | 1 | `pass` | 0 | `dispatched.hsaco` | `f437d8850a` | — |
| `_prepare_prefill_inputs_kernel` | 1 | `pass` | 0 | `dispatched.hsaco` | `fa4bd10847` | — |
| `_gather_block_tables_kernel` | 1 | `pass` | 0 | `dispatched.hsaco` | `ffba6285b6` | — |

</details>

<details><summary><b>waitcheck-qwen3</b> — observed `pass`</summary>

Observation: waitcheck pass

Reproduce: `python scripts/sanitizers/model_survey.py run --inventory recipes/sanitizers/models/qwen3.yaml --out results`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `_topk_topp_kernel` | 1 | `pass` | 0 | `dispatched.hsaco` | `0673f02020` | — |
| `_compute_slot_mapping_kernel` | 1 | `pass` | 0 | `dispatched.hsaco` | `2b4432007a` | — |
| `_fwd_kernel` | 1 | `pass` | 0 | `dispatched.hsaco` | `336ef011aa` | — |

</details>

## History / trend

| Run | Commit | daily-waitcheck-gemm | daily-consan-clean | daily-consan-racy | Gate |
|---|---|---|---|---|---|
| 2026-10-30T141145-41054343941 | `b78967101935` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-10-29T141105-40872193672 | `b78967101935` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-10-28T140907-40686657660 | `4fb536db06a8` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-10-27T155149-40503613953 | `4fb536db06a8` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-10-27T135759-40500801889 | `71099ef28e19` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-10-26T135357-40333939608 | `53636046fa47` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-10-25T122557-40192731938 | `a347a262ea51` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-10-24T122537-40093556970 | `a347a262ea51` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-10-23T122811-39952936160 | `7d567f0262b1` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-10-22T175155-39822417864 | `7d567f0262b1` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-10-22T141431-39778180398 | `3c6a345ae84f` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-10-21T141345-39599326762 | `3c6a345ae84f` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-10-20T141440-39422086566 | `3c6a345ae84f` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-10-19T141550-39252050781 | `3c6a345ae84f` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-10-18T141043-39126948533 | `3c6a345ae84f` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-10-17T140956-39032141956 | `3c6a345ae84f` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-10-16T160650-38922506989 | `3c6a345ae84f` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-10-16T123432-38900327093 | `bf117600ca07` | ✅ **Match**<br>Observed: `warn` | ❌ **Mismatch**<br>Observed: `error`; expected `pass` | ❌ **Mismatch**<br>Observed: `error`; expected `fail` | Regression |
| 2026-10-15T123427-38734461239 | `bf117600ca07` | ✅ **Match**<br>Observed: `warn` | ❌ **Mismatch**<br>Observed: `error`; expected `pass` | ❌ **Mismatch**<br>Observed: `error`; expected `fail` | Regression |
| 2026-10-14T195118-38563696959 | `bf117600ca07` | ✅ **Match**<br>Observed: `warn` | ❌ **Mismatch**<br>Observed: `error`; expected `pass` | ❌ **Mismatch**<br>Observed: `error`; expected `fail` | Regression |
| 2026-10-14T194155-38397494175 | `bf117600ca07` | ❌ **Report missing**<br>Observed: `—` | ❌ **Report missing**<br>Observed: `—` | ❌ **Report missing**<br>Observed: `—` | Incomplete |
| 2026-10-12T142744-38250297996 | `bf117600ca07` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-10-11T141417-38139192341 | `224d19f4a371` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-10-10T141315-38051823495 | `224d19f4a371` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-10-09T141446-37929907291 | `224d19f4a371` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-10-08T141657-37776980403 | `224d19f4a371` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-10-07T142601-37621140131 | `cab08a1abc4c` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-10-06T141219-37463222089 | `1f59e2040535` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-10-05T163143-37309782694 | `b306c32cf2b8` | ✅ **Match**<br>Observed: `warn` | ❌ **Mismatch**<br>Observed: `error`; expected `pass` | ❌ **Mismatch**<br>Observed: `error`; expected `fail` | Regression |
| 2026-10-05T161430-37207730800 | `77d4fbb48a5d` | ❌ **Report missing**<br>Observed: `—` | ❌ **Report missing**<br>Observed: `—` | ❌ **Report missing**<br>Observed: `—` | Incomplete |
