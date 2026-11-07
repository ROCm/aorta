# Sanitizers Nightly · gfx950

Run `2026-11-07T140920-42371579575` · commit `ac11f84616a8` · 2026-11-07 14:09:20 UTC

🟢 **Healthy** — 5/5 matched

Observed `warn` or `fail` verdicts may be expected positive-control outcomes. A matched row is healthy even when the verdict is fail. Baseline status is matched or unmatched.

| Recipe | Backend | Observed | Expected | Execution | Findings | Coverage | Baseline |
|---|---|---|---|---|--:|---|---|
| daily-waitcheck-gemm | waitcheck (static) | 🟡 warn | 🟡 warn | complete | 64 | — | **🟢 matched** |
| daily-consan-clean | consan (dynamic) | 🟢 pass | 🟢 pass | complete | 0 | 0/0, 2/2 | **🟢 matched** |
| daily-consan-racy | consan (dynamic) | 🔴 fail | 🔴 fail | complete | 1 | 0/0, 2/2 | **🟢 matched** |
| daily-fpsan-match | fpsan (digest) | 🟢 pass | 🟢 pass | complete | 0 | — | **🟢 matched** |
| daily-fpsan-mismatch | fpsan (digest) | 🔴 fail | 🔴 fail | complete | 1 | — | **🟢 matched** |

Three views below: **Nightly gate** (baseline-checked, the gate), **Workload survey** and **FpSan** (both observed only, non-gating).

## Nightly gate · Kernel details

<details><summary><b>daily-waitcheck-gemm</b> — 🟡 warn — **🟢 matched**</summary>

Observed 🟡 warn · expected 🟡 warn
Observation: waitcheck warn; 64 finding(s) (wait_hazard)
backend `rj_waitcheck` `2c406bcc3be3` · selection `top_dispatch_count` top-3 · 3 kernel(s) · execution complete

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `gemm_NT_M256_N4096_K1024` | 479 | 🟡 warn | 32 | `sol_126578.hsaco` | `57c5d8efa4` | — |
| `gemm_NT_M128_N4096_K1280` | 471 | 🟡 warn | 0 | `sol_175415.hsaco` | `57c5d8efa4` | same code object as gemm_NT_M256_N4096_K1024; scanned once |
| `gemm_TT_M64_N64_K1280` | 440 | 🟡 warn | 32 | `sol_137678.hsaco` | `aeb46fded1` | — |

| Sanitizer | Code | Severity | Count | Example |
|---|---|---|--:|---|
| waitcheck | `wait_hazard` | warning | 64 | sol_126578.hsaco:gfx950[0]:.text+0x293c: missing s_waitcnt lgkmcnt(14) before use of v20 |

</details>

<details><summary><b>daily-consan-clean</b> — 🟢 pass — **🟢 matched**</summary>

Observed 🟢 pass · expected 🟢 pass
Observation: consan pass; preflight pass
backend `—` · selection `top_dispatch_count` top-1 · 1 kernel(s) · execution complete

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `consan_lds_race` | 1 | 🟢 pass | 0 | `—` | `—` | — |

</details>

<details><summary><b>daily-consan-racy</b> — 🔴 fail — **🟢 matched**</summary>

Observed 🔴 fail · expected 🔴 fail
Observation: consan fail; 1 finding(s) (sampled_conflict); preflight pass
backend `—` · selection `top_dispatch_count` top-1 · 1 kernel(s) · execution complete

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `consan_lds_race_2wave` | 1 | 🔴 fail | 1 | `—` | `—` | — |

| Sanitizer | Code | Severity | Count | Example |
|---|---|---|--:|---|
| consan | `sampled_conflict` | race | 1 | [rocjitsu-dbi-hooks] ConSan conflict reader=105804328241424 first_index=2 second_index=13 first_kind=2 second_kind=1 first_owner=0 second_owner=1 epoch=0 gener… |

</details>

<details><summary><b>daily-fpsan-match</b> — 🟢 pass — **🟢 matched**</summary>

Observed 🟢 pass · expected 🟢 pass
Observation: fpsan pass
backend `—` · selection `top_dispatch_count` top-1 · 1 kernel(s) · execution complete

| Kernel pair | Observed | Reference digest | Candidate digest | Instrumented | Detail |
|---|---|---|---|---|---|
| `row_dot` | 🟢 pass | `53b41a29b1abfb54…` | `53b41a29b1abfb54…` | both | — |

</details>

<details><summary><b>daily-fpsan-mismatch</b> — 🔴 fail — **🟢 matched**</summary>

Observed 🔴 fail · expected 🔴 fail
Observation: fpsan fail; reason fpsan_digest_mismatch; 1 finding(s) (fpsan_digest_mismatch)
backend `—` · selection `top_dispatch_count` top-1 · 1 kernel(s) · execution complete

| Kernel pair | Observed | Reference digest | Candidate digest | Instrumented | Detail |
|---|---|---|---|---|---|
| `row_dot_no_accumulate` | 🔴 fail | `53b41a29b1abfb54…` | `b85f98818a3ebb61…` | both | fpsan_digest_mismatch |

| Sanitizer | Code | Severity | Count | Example |
|---|---|---|--:|---|
| fpsan | `fpsan_digest_mismatch` | error | 1 | reference and candidate kernels disagree under FpSan: reference digest 53b41a29b1abfb54... != candidate digest b85f98818a3ebb61... |

</details>

## Workload survey

How real GPU kernels behave under AMD's sanitizers — **waitcheck** (static `s_waitcnt` wait-count scan) and **ConSan** (dynamic data-race check); where both produced a report the kernel is shown under each, and a scan that was skipped or whose report is missing still appears, marked report missing with no verdict. **No expected-behavior comparison on this tab**; an `error` / `fail` / `warn` here is an observation of how the kernel behaved, not a gate result. Each case lists a copy-paste command to reproduce the run.

Surveyed 3 workloads (59 kernels) across 6 sanitizer runs — 🟢 2 pass · 🟡 1 warn · 🔴 1 fail · 🔴 1 error · ⚪ 1 not_checked

| Workload | Kernels | waitcheck | ConSan | Findings | Note |
|---|--:|---|---|--:|---|
| gemm | 1 | 🟡 warn | 🔴 error | 32 | combined_hook_timeout |
| gpt oss | 55 | 🟢 pass | 🔴 fail | 48 | sampled_conflict |
| qwen3 | 3 | 🟢 pass | ⚪ not_checked | 0 | 1 of 2 kernels not checked (1x consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object) |

<details><summary><b>consan-gemm</b> — observed 🔴 error</summary>

Observation: consan error; reason combined_hook_timeout; gemm_f32_ss: combined_hook_timeout; preflight error

Reason: `combined_hook_timeout — gemm_f32_ss: combined_hook_timeout`

Reproduce: `aorta sweep run --recipe recipes/sanitizers/daily-consan-gemm.yaml`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `gemm_f32_ss` | 1 | 🔴 error | 0 | `consan_gemm_f32.hsaco` | `57c5d8efa4` | combined_hook_timeout |

</details>

<details><summary><b>consan-gpt-oss</b> — observed 🔴 fail</summary>

Observation: consan fail; _combine_sampled_and_draft_tokens_kernel: consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object, _compute_slot_mappings_kernel: consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object, _gather_block_tables_kernel: consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object, _get_num_sampled_and_rejected_kernel: consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object, _masked_compaction: consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object, _post_update_kernel: consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object, _prepare_pos_seq_lens_kernel: consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object, _prepare_prefill_inputs_kernel: consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object, _rope_kernel_cached_thd_2c_gqa_onehead_fwd: consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object; 48 finding(s) (sampled_conflict); preflight pass

Finding: `[rocjitsu-dbi-hooks] ConSan conflict reader=639974640 first_index=112 second_index=118 first_kind=2 second_kind=2 first_owner=1 second_owner=3 epoch=8 generati…`

Reproduce: `python scripts/sanitizers/model_survey.py run --inventory recipes/sanitizers/models/gpt-oss.yaml --out results`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `_topk_forward` | 1 | 🔴 fail | 3 | `dispatched.hsaco` | `d46a6cefb7` | — |
| `_topk_topp_kernel` | 1 | 🔴 fail | 9 | `dispatched.hsaco` | `b7b68ee37b` | — |
| `kernel_unified_attention_2d` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `05e205ab8f` | — |
| `kernel_unified_attention_3d` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `6d038c64ca` | — |
| `pack_bitmatrix` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `62b879e054` | — |
| `reduce_segments` | 1 | 🔴 fail | 9 | `dispatched.hsaco` | `7bcee0d9ae` | — |
| `_combine_sampled_and_draft_tokens_kernel` | 1 | ⚪ not_checked | 0 | `dispatched.hsaco` | `44f4a73c86` | consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object |
| `_compute_slot_mappings_kernel` | 1 | ⚪ not_checked | 0 | `dispatched.hsaco` | `597c8b991d` | consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object |
| `_fused_add_rmsnorm_kernel` | 1 | 🔴 fail | 8 | `dispatched.hsaco` | `b7cbcd5bb8` | — |
| `_fused_add_rmsnorm_pad` | 1 | 🔴 fail | 8 | `dispatched.hsaco` | `dc6e5bea86` | — |
| `_gather_block_tables_kernel` | 1 | ⚪ not_checked | 0 | `dispatched.hsaco` | `ffba6285b6` | consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object |
| `_get_num_sampled_and_rejected_kernel` | 1 | ⚪ not_checked | 0 | `dispatched.hsaco` | `cb7a82decf` | consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object |
| `_masked_compaction` | 1 | ⚪ not_checked | 0 | `dispatched.hsaco` | `a3e32aef6c` | consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object |
| `_matmul_ogs_NNT_bf16xbf16xmxfp4_32x256x128x1_swiglu` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `02170daa86` | — |
| `_matmul_ogs_NNT_bf16xbf16xmxfp4_32x256x128x1` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `8b64fa68f3` | — |
| `_post_update_kernel` | 1 | ⚪ not_checked | 0 | `dispatched.hsaco` | `d39ae3b1c4` | consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object |
| `_prepare_pos_seq_lens_kernel` | 1 | ⚪ not_checked | 0 | `dispatched.hsaco` | `6eb8b8627f` | consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object |
| `_prepare_prefill_inputs_kernel` | 1 | ⚪ not_checked | 0 | `dispatched.hsaco` | `fa4bd10847` | consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object |
| `_rms_norm_kernel` | 1 | 🔴 fail | 8 | `dispatched.hsaco` | `b6dcdd21dd` | — |
| `_rope_kernel_cached_thd_2c_gqa_fwd` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `44472db069` | — |
| `_rope_kernel_cached_thd_2c_gqa_onehead_fwd` | 1 | ⚪ not_checked | 0 | `dispatched.hsaco` | `d5458dc446` | consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object |
| `_stage2_pow2` | 1 | 🔴 fail | 3 | `dispatched.hsaco` | `e6acd0c33b` | — |
| `_swiglu_bf16xbf16_16x128` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `6860ae4cbc` | — |

</details>

<details><summary><b>consan-qwen3</b> — observed ⚪ not_checked</summary>

Observation: consan not_checked; reason 1 of 2 kernels not checked (1x consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object); _compute_slot_mapping_kernel: consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object; preflight pass

Reason: `1 of 2 kernels not checked (1x consan_not_applicable: no LDS access, barrier, a… — _compute_slot_mapping_kernel: consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object`

Reproduce: `python scripts/sanitizers/model_survey.py run --inventory recipes/sanitizers/models/qwen3.yaml --out results`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `_fwd_kernel` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `336ef011aa` | — |
| `_compute_slot_mapping_kernel` | 1 | ⚪ not_checked | 0 | `dispatched.hsaco` | `2b4432007a` | consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object |

</details>

<details><summary><b>waitcheck-gemm</b> — observed 🟡 warn</summary>

Observation: waitcheck warn; 32 finding(s) (wait_hazard)

Finding: `consan_gemm_f32.hsaco:gfx950[0]:.text+0x293c: missing s_waitcnt lgkmcnt(14) before use of v20`

Reproduce: `aorta sweep run --recipe recipes/sanitizers/daily-waitcheck-gemm-object.yaml`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `gemm_f32_ss` | 1 | 🟡 warn | 32 | `consan_gemm_f32.hsaco` | `57c5d8efa4` | — |

</details>

<details><summary><b>waitcheck-gpt-oss</b> — observed 🟢 pass</summary>

Observation: waitcheck pass

Reproduce: `python scripts/sanitizers/model_survey.py run --inventory recipes/sanitizers/models/gpt-oss.yaml --out results`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `_matmul_ogs_NNT_bf16xbf16xmxfp4_32x256x128x1_swiglu` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `02170daa86` | — |
| `_rope_kernel_cached_thd_2c_gqa_onehead_fwd` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `02f9a6e283` | — |
| `_fused_add_rmsnorm_kernel` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `0429ecf4ed` | — |
| `kernel_unified_attention_2d` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `05e205ab8f` | — |
| `_stage2_pow2` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `1192e120f5` | — |
| `_fused_add_rmsnorm_kernel` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `136cc91e21` | — |
| `_rms_norm_kernel` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `1dcf3e3765` | — |
| `kernel_unified_attention_2d` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `1e85e67552` | — |
| `_fused_add_rmsnorm_kernel` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `26b4c78f52` | — |
| `_fused_add_rmsnorm_kernel` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `28c7fc9948` | — |
| `_matmul_ogs_NNT_bf16xbf16xmxfp4_64x256x128x1` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `357789577b` | — |
| `_rope_kernel_cached_thd_2c_gqa_fwd` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `44472db069` | — |
| `_combine_sampled_and_draft_tokens_kernel` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `44f4a73c86` | — |
| `_fused_add_rmsnorm_pad` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `494126cd1a` | — |
| `_topk_topp_kernel` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `50fe4fea9d` | — |
| `_matmul_ogs_NNT_bf16xbf16xmxfp4_32x256x128x1_swiglu` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `5111186046` | — |
| `reduce_segments` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `53c2bad898` | — |
| `_compute_slot_mappings_kernel` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `597c8b991d` | — |
| `_topk_forward` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `5c4d4d519d` | — |
| `_fused_add_rmsnorm_kernel` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `61903965e1` | — |
| `pack_bitmatrix` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `62b879e054` | — |
| `_rms_norm_kernel` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `63fa64b777` | — |
| `_swiglu_bf16xbf16_16x128` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `6860ae4cbc` | — |
| `kernel_unified_attention_2d` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `699e09c798` | — |
| `kernel_unified_attention_2d` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `6a27646a37` | — |
| `kernel_unified_attention_3d` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `6d038c64ca` | — |
| `_prepare_pos_seq_lens_kernel` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `6eb8b8627f` | — |
| `_rope_kernel_cached_thd_2c_gqa_onehead_fwd` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `6f97774106` | — |
| `kernel_unified_attention_2d` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `7433030bb8` | — |
| `reduce_segments` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `7bcee0d9ae` | — |
| `pack_bitmatrix` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `7da1e46e4d` | — |
| `_matmul_ogs_NNT_bf16xbf16xmxfp4_32x256x128x1` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `8b64fa68f3` | — |
| `_rope_kernel_cached_thd_2c_gqa_fwd` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `8f39bbd1bd` | — |
| `_matmul_ogs_NNT_bf16xbf16xmxfp4_64x256x128x1_swiglu` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `99783dba6b` | — |
| `_masked_compaction` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `a3e32aef6c` | — |
| `kernel_unified_attention_2d` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `a5497ec8d0` | — |
| `_matmul_ogs_NNT_bf16xbf16xmxfp4_32x256x128x1` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `b60dc71885` | — |
| `_rms_norm_kernel` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `b6dcdd21dd` | — |
| `_topk_topp_kernel` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `b7b68ee37b` | — |
| `_fused_add_rmsnorm_kernel` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `b7cbcd5bb8` | — |
| `_get_num_sampled_and_rejected_kernel` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `cb7a82decf` | — |
| `_rms_norm_kernel` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `cca3ceda2f` | — |
| `_post_update_kernel` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `d39ae3b1c4` | — |
| `_topk_forward` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `d46a6cefb7` | — |
| `_rope_kernel_cached_thd_2c_gqa_onehead_fwd` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `d5458dc446` | — |
| `_fused_add_rmsnorm_pad` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `dc6e5bea86` | — |
| `pack_bitmatrix` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `de085311a4` | — |
| `_fused_add_rmsnorm_kernel` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `df48165cc5` | — |
| `_rms_norm_kernel` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `e189a11509` | — |
| `_rms_norm_kernel` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `e5d86d5d3e` | — |
| `_stage2_pow2` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `e6acd0c33b` | — |
| `_rms_norm_kernel` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `ec7c47196f` | — |
| `kernel_unified_attention_3d` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `f437d8850a` | — |
| `_prepare_prefill_inputs_kernel` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `fa4bd10847` | — |
| `_gather_block_tables_kernel` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `ffba6285b6` | — |

</details>

<details><summary><b>waitcheck-qwen3</b> — observed 🟢 pass</summary>

Observation: waitcheck pass

Reproduce: `python scripts/sanitizers/model_survey.py run --inventory recipes/sanitizers/models/qwen3.yaml --out results`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `_topk_topp_kernel` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `0673f02020` | — |
| `_compute_slot_mapping_kernel` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `2b4432007a` | — |
| `_fwd_kernel` | 1 | 🟢 pass | 0 | `dispatched.hsaco` | `336ef011aa` | — |

</details>

## FpSan

**FpSan** (reference/candidate digest check) runs a reference and a candidate Triton kernel under FpSan and compares the digests of their outputs; different digests mean the two kernels do not compute the same thing. Observed only, so a mismatch here does not change the nightly. Each case lists a copy-paste command to reproduce the run.

Compared 105 kernel pairs across 2 workloads — 🟢 105 pass

| Workload | Pairs | Observed | Mismatches | Note |
|---|--:|---|--:|---|
| gpt oss | 102 | 🟢 pass | 0 | — |
| qwen3 | 3 | 🟢 pass | 0 | — |

<details><summary><b>fpsan-gpt-oss</b> — observed 🟢 pass</summary>

Observation: fpsan pass

Reproduce: `python3 scripts/sanitizers/model_survey.py run --sanitizers fpsan --inventory recipes/sanitizers/models/gpt-oss.yaml --out results`

| Kernel pair | Observed | Reference digest | Candidate digest | Detail |
|---|---|---|---|---|
| `_fused_add_rmsnorm_kernel/D1` | 🟢 pass | `6e37a322cfc6cce1…` | `6e37a322cfc6cce1…` | — |
| `_fused_add_rmsnorm_kernel/D2` | 🟢 pass | `484446e57c0def4d…` | `484446e57c0def4d…` | — |
| `_fused_add_rmsnorm_kernel/D3` | 🟢 pass | `5abfefc32c7193b6…` | `5abfefc32c7193b6…` | — |
| `_fused_add_rmsnorm_kernel/D4` | 🟢 pass | `256ee9cb29e84124…` | `256ee9cb29e84124…` | — |
| `_fused_add_rmsnorm_kernel/P1` | 🟢 pass | `ea63473d20507f81…` | `ea63473d20507f81…` | — |
| `_fused_add_rmsnorm_kernel/P2` | 🟢 pass | `f6b48b854a126c8c…` | `f6b48b854a126c8c…` | — |
| `_fused_add_rmsnorm_kernel/P3` | 🟢 pass | `cace856f0299123a…` | `cace856f0299123a…` | — |
| `_fused_add_rmsnorm_kernel/P4` | 🟢 pass | `b2f72cffe35cad4c…` | `b2f72cffe35cad4c…` | — |
| `_fused_add_rmsnorm_pad/D1` | 🟢 pass | `a787f850de78f8a3…` | `a787f850de78f8a3…` | — |
| `_fused_add_rmsnorm_pad/D2` | 🟢 pass | `95df39f4d58cbd37…` | `95df39f4d58cbd37…` | — |
| `_fused_add_rmsnorm_pad/D3` | 🟢 pass | `6014a3d0468da4bc…` | `6014a3d0468da4bc…` | — |
| `_fused_add_rmsnorm_pad/D4` | 🟢 pass | `2fb1efcb8ab6f7bd…` | `2fb1efcb8ab6f7bd…` | — |
| `_fused_add_rmsnorm_pad/P1` | 🟢 pass | `d38cce118057b3ae…` | `d38cce118057b3ae…` | — |
| `_fused_add_rmsnorm_pad/P2` | 🟢 pass | `c2fe692abe5a500b…` | `c2fe692abe5a500b…` | — |
| `_fused_add_rmsnorm_pad/P3` | 🟢 pass | `96162bf879bcfee2…` | `96162bf879bcfee2…` | — |
| `_fused_add_rmsnorm_pad/P4` | 🟢 pass | `db1f23622bc6078f…` | `db1f23622bc6078f…` | — |
| `_matmul_ogs/D1` | 🟢 pass | `5152594de1f4e53d…` | `5152594de1f4e53d…` | — |
| `_matmul_ogs/D2` | 🟢 pass | `91b5096017a39a79…` | `91b5096017a39a79…` | — |
| `_matmul_ogs/D3` | 🟢 pass | `b1e341807d343019…` | `b1e341807d343019…` | — |
| `_matmul_ogs/D4` | 🟢 pass | `62c91d1188d407ac…` | `62c91d1188d407ac…` | — |
| `_matmul_ogs/P1` | 🟢 pass | `da31c40c6f42a74c…` | `da31c40c6f42a74c…` | — |
| `_matmul_ogs/P2` | 🟢 pass | `770ebaeb7f83970b…` | `770ebaeb7f83970b…` | — |
| `_matmul_ogs/P3` | 🟢 pass | `3f8de647521441c6…` | `3f8de647521441c6…` | — |
| `_matmul_ogs/P4` | 🟢 pass | `609c0a9d6f6f334d…` | `609c0a9d6f6f334d…` | — |
| `_matmul_ogs@w2/D1` | 🟢 pass | `44c065a74eab8ca9…` | `44c065a74eab8ca9…` | — |
| `_matmul_ogs@w2/D2` | 🟢 pass | `e9c04082588ba8ee…` | `e9c04082588ba8ee…` | — |
| `_matmul_ogs@w2/D3` | 🟢 pass | `72b7585192ea410b…` | `72b7585192ea410b…` | — |
| `_matmul_ogs@w2/D4` | 🟢 pass | `67dd7d1570208734…` | `67dd7d1570208734…` | — |
| `_matmul_ogs@w2/P1` | 🟢 pass | `fb56e9d7c16007ab…` | `fb56e9d7c16007ab…` | — |
| `_matmul_ogs@w2/P2` | 🟢 pass | `54cdc9e01e5f3aa6…` | `54cdc9e01e5f3aa6…` | — |
| `_matmul_ogs@w2/P3` | 🟢 pass | `649be589cbe95faa…` | `649be589cbe95faa…` | — |
| `_matmul_ogs@w2/P4` | 🟢 pass | `8021aca8d2747431…` | `8021aca8d2747431…` | — |
| `_rms_norm_kernel/D1` | 🟢 pass | `e87ddeeef77354b0…` | `e87ddeeef77354b0…` | — |
| `_rms_norm_kernel/D2` | 🟢 pass | `96d61b7bde70e154…` | `96d61b7bde70e154…` | — |
| `_rms_norm_kernel/D3` | 🟢 pass | `5f4638701b557afe…` | `5f4638701b557afe…` | — |
| `_rms_norm_kernel/D4` | 🟢 pass | `47a40d16778be8d5…` | `47a40d16778be8d5…` | — |
| `_rms_norm_kernel/P1` | 🟢 pass | `073f18d1d51f7e27…` | `073f18d1d51f7e27…` | — |
| `_rms_norm_kernel/P2` | 🟢 pass | `736b391d210c40bc…` | `736b391d210c40bc…` | — |
| `_rms_norm_kernel/P3` | 🟢 pass | `8906f4ac708203f9…` | `8906f4ac708203f9…` | — |
| `_rms_norm_kernel/P4` | 🟢 pass | `9f73ee1035bff333…` | `9f73ee1035bff333…` | — |
| `_rope_kernel_cached_thd_2c_gqa_fwd/D1` | 🟢 pass | `32b682cd94c40b7d…` | `32b682cd94c40b7d…` | — |
| `_rope_kernel_cached_thd_2c_gqa_fwd/D2` | 🟢 pass | `f30ac2598a7018aa…` | `f30ac2598a7018aa…` | — |
| `_rope_kernel_cached_thd_2c_gqa_fwd/D3` | 🟢 pass | `6a3cd679f0705d8f…` | `6a3cd679f0705d8f…` | — |
| `_rope_kernel_cached_thd_2c_gqa_fwd/D4` | 🟢 pass | `82756a77173cdfbe…` | `82756a77173cdfbe…` | — |
| `_rope_kernel_cached_thd_2c_gqa_fwd/P1` | 🟢 pass | `adf251cb45af58e6…` | `adf251cb45af58e6…` | — |
| `_rope_kernel_cached_thd_2c_gqa_fwd/P2` | 🟢 pass | `32d43670b0a0f104…` | `32d43670b0a0f104…` | — |
| `_rope_kernel_cached_thd_2c_gqa_fwd/P3` | 🟢 pass | `01ccf8d4dab5779f…` | `01ccf8d4dab5779f…` | — |
| `_rope_kernel_cached_thd_2c_gqa_fwd/P4` | 🟢 pass | `c640b749f4288f02…` | `c640b749f4288f02…` | — |
| `_rope_kernel_cached_thd_2c_gqa_onehead_fwd/D1` | 🟢 pass | `5b51de9b45ace5bd…` | `5b51de9b45ace5bd…` | — |
| `_rope_kernel_cached_thd_2c_gqa_onehead_fwd/D2` | 🟢 pass | `ea63f21b9c92ceca…` | `ea63f21b9c92ceca…` | — |
| `_rope_kernel_cached_thd_2c_gqa_onehead_fwd/D3` | 🟢 pass | `c259e259979796ef…` | `c259e259979796ef…` | — |
| `_rope_kernel_cached_thd_2c_gqa_onehead_fwd/D4` | 🟢 pass | `ca50fbc8bdc1df81…` | `ca50fbc8bdc1df81…` | — |
| `_rope_kernel_cached_thd_2c_gqa_onehead_fwd/P1` | 🟢 pass | `c3d1849bc5857585…` | `c3d1849bc5857585…` | — |
| `_rope_kernel_cached_thd_2c_gqa_onehead_fwd/P2` | 🟢 pass | `76ea103292b9cf15…` | `76ea103292b9cf15…` | — |
| `_rope_kernel_cached_thd_2c_gqa_onehead_fwd/P3` | 🟢 pass | `64903d518690f25c…` | `64903d518690f25c…` | — |
| `_rope_kernel_cached_thd_2c_gqa_onehead_fwd/P4` | 🟢 pass | `ebb01513ea7009e4…` | `ebb01513ea7009e4…` | — |
| `_swiglu/D1` | 🟢 pass | `8c5a2f68f254bd4a…` | `8c5a2f68f254bd4a…` | — |
| `_swiglu/D2` | 🟢 pass | `223c356e393a179b…` | `223c356e393a179b…` | — |
| `_swiglu/D3` | 🟢 pass | `af07d9ece3e30c76…` | `af07d9ece3e30c76…` | — |
| `_swiglu/D4` | 🟢 pass | `fa395db60e22d78a…` | `fa395db60e22d78a…` | — |
| `_swiglu/P1` | 🟢 pass | `225fdade79c927c9…` | `225fdade79c927c9…` | — |
| `_swiglu/P2` | 🟢 pass | `4390b81e125c6677…` | `4390b81e125c6677…` | — |
| `_swiglu/P3` | 🟢 pass | `ce849354885a0484…` | `ce849354885a0484…` | — |
| `_swiglu/P4` | 🟢 pass | `0642644a2c5b62a1…` | `0642644a2c5b62a1…` | — |
| `_topk_forward/D1` | 🟢 pass | `f45486e386dc5683…` | `f45486e386dc5683…` | — |
| `_topk_forward/D2` | 🟢 pass | `eadf2627e74b7402…` | `eadf2627e74b7402…` | — |
| `_topk_forward/D3` | 🟢 pass | `e9977e13eba0680e…` | `e9977e13eba0680e…` | — |
| `_topk_forward/D4` | 🟢 pass | `23b5e433ea6f5fc8…` | `23b5e433ea6f5fc8…` | — |
| `_topk_forward/P1` | 🟢 pass | `f6b864c1b89cf8b7…` | `f6b864c1b89cf8b7…` | — |
| `_topk_forward/P2` | 🟢 pass | `e4a29b8159f2677e…` | `e4a29b8159f2677e…` | — |
| `_topk_forward/P3` | 🟢 pass | `252d48ee2bc97aa3…` | `252d48ee2bc97aa3…` | — |
| `_topk_forward/P4` | 🟢 pass | `7739451c83e080f3…` | `7739451c83e080f3…` | — |
| `_topk_topp_kernel/D1` | 🟢 pass | `68d3b34339f5cfbb…` | `68d3b34339f5cfbb…` | — |
| `_topk_topp_kernel/D2` | 🟢 pass | `7e9413c66234b385…` | `7e9413c66234b385…` | — |
| `_topk_topp_kernel/D3` | 🟢 pass | `174ad6c9c8604aef…` | `174ad6c9c8604aef…` | — |
| `_topk_topp_kernel/D4` | 🟢 pass | `1a7f72e91b5eb2ad…` | `1a7f72e91b5eb2ad…` | — |
| `_topk_topp_kernel/P1` | 🟢 pass | `68d3b34339f5cfbb…` | `68d3b34339f5cfbb…` | — |
| `_topk_topp_kernel/P2` | 🟢 pass | `68d3b34339f5cfbb…` | `68d3b34339f5cfbb…` | — |
| `_topk_topp_kernel/P3` | 🟢 pass | `68d3b34339f5cfbb…` | `68d3b34339f5cfbb…` | — |
| `_topk_topp_kernel/P4` | 🟢 pass | `7e9413c66234b385…` | `7e9413c66234b385…` | — |
| `kernel_unified_attention_2d/D1_full` | 🟢 pass | `1b183b97fcfd9657…` | `1b183b97fcfd9657…` | — |
| `kernel_unified_attention_2d/D1_swa` | 🟢 pass | `e7fe4a149aa149d6…` | `e7fe4a149aa149d6…` | — |
| `kernel_unified_attention_2d/D2_full` | 🟢 pass | `170e86d5d5b756e9…` | `170e86d5d5b756e9…` | — |
| `kernel_unified_attention_2d/D2_swa` | 🟢 pass | `8e9dc2e68cac67ae…` | `8e9dc2e68cac67ae…` | — |
| `kernel_unified_attention_2d/D3_full` | 🟢 pass | `8564266319eb7ac8…` | `8564266319eb7ac8…` | — |
| `kernel_unified_attention_2d/D3_swa` | 🟢 pass | `f20f742563d2f6d6…` | `f20f742563d2f6d6…` | — |
| `kernel_unified_attention_2d/D4_full` | 🟢 pass | `13ca5ce07e225bff…` | `13ca5ce07e225bff…` | — |
| `kernel_unified_attention_2d/D4_swa` | 🟢 pass | `d3cf7f32a55b722c…` | `d3cf7f32a55b722c…` | — |
| `kernel_unified_attention_2d/P1_full` | 🟢 pass | `5023ad372afd0dcb…` | `5023ad372afd0dcb…` | — |
| `kernel_unified_attention_2d/P1_swa` | 🟢 pass | `5023ad372afd0dcb…` | `5023ad372afd0dcb…` | — |
| `kernel_unified_attention_2d/P2_full` | 🟢 pass | `5ff7542e563ad82a…` | `5ff7542e563ad82a…` | — |
| `kernel_unified_attention_2d/P2_swa` | 🟢 pass | `bad332baf8619815…` | `bad332baf8619815…` | — |
| `kernel_unified_attention_2d/P3_full` | 🟢 pass | `320bad23dd01eb3c…` | `320bad23dd01eb3c…` | — |
| `kernel_unified_attention_2d/P3_swa` | 🟢 pass | `4e23fd9fc586c110…` | `4e23fd9fc586c110…` | — |
| `kernel_unified_attention_2d/P4_full` | 🟢 pass | `14b8d771899c62ea…` | `14b8d771899c62ea…` | — |
| `kernel_unified_attention_2d/P4_swa` | 🟢 pass | `ca687daf4f52089f…` | `ca687daf4f52089f…` | — |
| `kernel_unified_attention_2d/X1_swa` | 🟢 pass | `36a7faacb8498efc…` | `36a7faacb8498efc…` | — |
| `kernel_unified_attention_2d/X2_swa` | 🟢 pass | `c48a1b82028a6e7c…` | `c48a1b82028a6e7c…` | — |
| `kernel_unified_attention_3d/X1_full` | 🟢 pass | `3db05bc0633a42fe…` | `3db05bc0633a42fe…` | — |
| `kernel_unified_attention_3d/X2_full` | 🟢 pass | `0c0cc9e1d04d22ff…` | `0c0cc9e1d04d22ff…` | — |
| `reduce_segments/X1_full` | 🟢 pass | `f0ddc325ece16bdb…` | `f0ddc325ece16bdb…` | — |
| `reduce_segments/X2_full` | 🟢 pass | `f67f4fb3f8614b74…` | `f67f4fb3f8614b74…` | — |

</details>

<details><summary><b>fpsan-qwen3</b> — observed 🟢 pass</summary>

Observation: fpsan pass

Reproduce: `python3 scripts/sanitizers/model_survey.py run --sanitizers fpsan --inventory recipes/sanitizers/models/qwen3.yaml --out results`

| Kernel pair | Observed | Reference digest | Candidate digest | Detail |
|---|---|---|---|---|
| `_fwd_kernel/C0` | 🟢 pass | `dba12a952bdea343…` | `dba12a952bdea343…` | — |
| `_fwd_kernel/C1` | 🟢 pass | `3929fc52b083d4e0…` | `3929fc52b083d4e0…` | — |
| `_topk_topp_kernel/C0` | 🟢 pass | `a7f866a7262c7c7e…` | `a7f866a7262c7c7e…` | — |

</details>

## History / trend

| Run | Commit | daily-waitcheck-gemm | daily-consan-clean | daily-consan-racy | daily-fpsan-match | daily-fpsan-mismatch | Gate |
|---|---|---|---|---|---|---|---|
| 2026-11-07T140920-42371579575 | `ac11f84616a8` | Observed: 🟡 warn<br>**🟢 matched** | Observed: 🟢 pass<br>**🟢 matched** | Observed: 🔴 fail<br>**🟢 matched** | Observed: 🟢 pass<br>**🟢 matched** | Observed: 🔴 fail<br>**🟢 matched** | 🟢 healthy |
| 2026-11-06T141021-42216230928 | `ac11f84616a8` | Observed: 🟡 warn<br>**🟢 matched** | Observed: 🟢 pass<br>**🟢 matched** | Observed: 🔴 fail<br>**🟢 matched** | Observed: 🟢 pass<br>**🟢 matched** | Observed: 🔴 fail<br>**🟢 matched** | 🟢 healthy |
| 2026-11-05T141031-42025576155 | `6ad9b1aefedd` | Observed: 🟡 warn<br>**🟢 matched** | Observed: 🟢 pass<br>**🟢 matched** | Observed: 🔴 fail<br>**🟢 matched** | Observed: 🟢 pass<br>**🟢 matched** | Observed: 🔴 fail<br>**🟢 matched** | 🟢 healthy |
| 2026-11-04T141045-41827691971 | `6ad9b1aefedd` | Observed: 🟡 warn<br>**🟢 matched** | Observed: 🟢 pass<br>**🟢 matched** | Observed: 🔴 fail<br>**🟢 matched** | Observed: 🟢 pass<br>**🟢 matched** | Observed: 🔴 fail<br>**🟢 matched** | 🟢 healthy |
| 2026-11-04T104441-41802201446 | `6ad9b1aefedd` | Observed: 🟡 warn<br>**🟢 matched** | Observed: 🟢 pass<br>**🟢 matched** | Observed: 🔴 fail<br>**🟢 matched** | Observed: 🟢 pass<br>**🟢 matched** | Observed: 🔴 fail<br>**🟢 matched** | 🟢 healthy |
| 2026-11-03T140644-41634136516 | `12e75e694488` | Observed: 🟡 warn<br>**🟢 matched** | Observed: 🟢 pass<br>**🟢 matched** | Observed: 🔴 fail<br>**🟢 matched** | Observed: ⚪ —<br>**⚪ not run** | Observed: ⚪ —<br>**⚪ not run** | 🟢 healthy |
| 2026-11-02T140936-41443404361 | `d22f6116235c` | Observed: 🟡 warn<br>**🟢 matched** | Observed: 🟢 pass<br>**🟢 matched** | Observed: 🔴 fail<br>**🟢 matched** | Observed: ⚪ —<br>**⚪ not run** | Observed: ⚪ —<br>**⚪ not run** | 🟢 healthy |
| 2026-11-01T140444-41305497680 | `d22f6116235c` | Observed: 🟡 warn<br>**🟢 matched** | Observed: 🟢 pass<br>**🟢 matched** | Observed: 🔴 fail<br>**🟢 matched** | Observed: ⚪ —<br>**⚪ not run** | Observed: ⚪ —<br>**⚪ not run** | 🟢 healthy |
| 2026-10-31T140437-41200155808 | `d22f6116235c` | Observed: 🟡 warn<br>**🟢 matched** | Observed: 🟢 pass<br>**🟢 matched** | Observed: 🔴 fail<br>**🟢 matched** | Observed: ⚪ —<br>**⚪ not run** | Observed: ⚪ —<br>**⚪ not run** | 🟢 healthy |
| 2026-10-30T141145-41054343941 | `b78967101935` | Observed: 🟡 warn<br>**🟢 matched** | Observed: 🟢 pass<br>**🟢 matched** | Observed: 🔴 fail<br>**🟢 matched** | Observed: ⚪ —<br>**⚪ not run** | Observed: ⚪ —<br>**⚪ not run** | 🟢 healthy |
| 2026-10-29T141105-40872193672 | `b78967101935` | Observed: 🟡 warn<br>**🟢 matched** | Observed: 🟢 pass<br>**🟢 matched** | Observed: 🔴 fail<br>**🟢 matched** | Observed: ⚪ —<br>**⚪ not run** | Observed: ⚪ —<br>**⚪ not run** | 🟢 healthy |
| 2026-10-28T140907-40686657660 | `4fb536db06a8` | Observed: 🟡 warn<br>**🟢 matched** | Observed: 🟢 pass<br>**🟢 matched** | Observed: 🔴 fail<br>**🟢 matched** | Observed: ⚪ —<br>**⚪ not run** | Observed: ⚪ —<br>**⚪ not run** | 🟢 healthy |
| 2026-10-27T155149-40503613953 | `4fb536db06a8` | Observed: 🟡 warn<br>**🟢 matched** | Observed: 🟢 pass<br>**🟢 matched** | Observed: 🔴 fail<br>**🟢 matched** | Observed: ⚪ —<br>**⚪ not run** | Observed: ⚪ —<br>**⚪ not run** | 🟢 healthy |
| 2026-10-27T135759-40500801889 | `71099ef28e19` | Observed: 🟡 warn<br>**🟢 matched** | Observed: 🟢 pass<br>**🟢 matched** | Observed: 🔴 fail<br>**🟢 matched** | Observed: ⚪ —<br>**⚪ not run** | Observed: ⚪ —<br>**⚪ not run** | 🟢 healthy |
| 2026-10-26T135357-40333939608 | `53636046fa47` | Observed: 🟡 warn<br>**🟢 matched** | Observed: 🟢 pass<br>**🟢 matched** | Observed: 🔴 fail<br>**🟢 matched** | Observed: ⚪ —<br>**⚪ not run** | Observed: ⚪ —<br>**⚪ not run** | 🟢 healthy |
| 2026-10-25T122557-40192731938 | `a347a262ea51` | Observed: 🟡 warn<br>**🟢 matched** | Observed: 🟢 pass<br>**🟢 matched** | Observed: 🔴 fail<br>**🟢 matched** | Observed: ⚪ —<br>**⚪ not run** | Observed: ⚪ —<br>**⚪ not run** | 🟢 healthy |
| 2026-10-24T122537-40093556970 | `a347a262ea51` | Observed: 🟡 warn<br>**🟢 matched** | Observed: 🟢 pass<br>**🟢 matched** | Observed: 🔴 fail<br>**🟢 matched** | Observed: ⚪ —<br>**⚪ not run** | Observed: ⚪ —<br>**⚪ not run** | 🟢 healthy |
| 2026-10-23T122811-39952936160 | `7d567f0262b1` | Observed: 🟡 warn<br>**🟢 matched** | Observed: 🟢 pass<br>**🟢 matched** | Observed: 🔴 fail<br>**🟢 matched** | Observed: ⚪ —<br>**⚪ not run** | Observed: ⚪ —<br>**⚪ not run** | 🟢 healthy |
| 2026-10-22T175155-39822417864 | `7d567f0262b1` | Observed: 🟡 warn<br>**🟢 matched** | Observed: 🟢 pass<br>**🟢 matched** | Observed: 🔴 fail<br>**🟢 matched** | Observed: ⚪ —<br>**⚪ not run** | Observed: ⚪ —<br>**⚪ not run** | 🟢 healthy |
| 2026-10-22T141431-39778180398 | `3c6a345ae84f` | Observed: 🟡 warn<br>**🟢 matched** | Observed: 🟢 pass<br>**🟢 matched** | Observed: 🔴 fail<br>**🟢 matched** | Observed: ⚪ —<br>**⚪ not run** | Observed: ⚪ —<br>**⚪ not run** | 🟢 healthy |
| 2026-10-21T141345-39599326762 | `3c6a345ae84f` | Observed: 🟡 warn<br>**🟢 matched** | Observed: 🟢 pass<br>**🟢 matched** | Observed: 🔴 fail<br>**🟢 matched** | Observed: ⚪ —<br>**⚪ not run** | Observed: ⚪ —<br>**⚪ not run** | 🟢 healthy |
| 2026-10-20T141440-39422086566 | `3c6a345ae84f` | Observed: 🟡 warn<br>**🟢 matched** | Observed: 🟢 pass<br>**🟢 matched** | Observed: 🔴 fail<br>**🟢 matched** | Observed: ⚪ —<br>**⚪ not run** | Observed: ⚪ —<br>**⚪ not run** | 🟢 healthy |
| 2026-10-19T141550-39252050781 | `3c6a345ae84f` | Observed: 🟡 warn<br>**🟢 matched** | Observed: 🟢 pass<br>**🟢 matched** | Observed: 🔴 fail<br>**🟢 matched** | Observed: ⚪ —<br>**⚪ not run** | Observed: ⚪ —<br>**⚪ not run** | 🟢 healthy |
| 2026-10-18T141043-39126948533 | `3c6a345ae84f` | Observed: 🟡 warn<br>**🟢 matched** | Observed: 🟢 pass<br>**🟢 matched** | Observed: 🔴 fail<br>**🟢 matched** | Observed: ⚪ —<br>**⚪ not run** | Observed: ⚪ —<br>**⚪ not run** | 🟢 healthy |
| 2026-10-17T140956-39032141956 | `3c6a345ae84f` | Observed: 🟡 warn<br>**🟢 matched** | Observed: 🟢 pass<br>**🟢 matched** | Observed: 🔴 fail<br>**🟢 matched** | Observed: ⚪ —<br>**⚪ not run** | Observed: ⚪ —<br>**⚪ not run** | 🟢 healthy |
| 2026-10-16T160650-38922506989 | `3c6a345ae84f` | Observed: 🟡 warn<br>**🟢 matched** | Observed: 🟢 pass<br>**🟢 matched** | Observed: 🔴 fail<br>**🟢 matched** | Observed: ⚪ —<br>**⚪ not run** | Observed: ⚪ —<br>**⚪ not run** | 🟢 healthy |
| 2026-10-16T123432-38900327093 | `bf117600ca07` | Observed: 🟡 warn<br>**🟢 matched** | Observed: 🔴 error; expected 🟢 pass<br>**🔴 unmatched** | Observed: 🔴 error; expected 🔴 fail<br>**🔴 unmatched** | Observed: ⚪ —<br>**⚪ not run** | Observed: ⚪ —<br>**⚪ not run** | 🔴 failed |
| 2026-10-15T123427-38734461239 | `bf117600ca07` | Observed: 🟡 warn<br>**🟢 matched** | Observed: 🔴 error; expected 🟢 pass<br>**🔴 unmatched** | Observed: 🔴 error; expected 🔴 fail<br>**🔴 unmatched** | Observed: ⚪ —<br>**⚪ not run** | Observed: ⚪ —<br>**⚪ not run** | 🔴 failed |
| 2026-10-14T195118-38563696959 | `bf117600ca07` | Observed: 🟡 warn<br>**🟢 matched** | Observed: 🔴 error; expected 🟢 pass<br>**🔴 unmatched** | Observed: 🔴 error; expected 🔴 fail<br>**🔴 unmatched** | Observed: ⚪ —<br>**⚪ not run** | Observed: ⚪ —<br>**⚪ not run** | 🔴 failed |
| 2026-10-14T194155-38397494175 | `bf117600ca07` | Observed: ⚪ —<br>**🔴 report missing** | Observed: ⚪ —<br>**🔴 report missing** | Observed: ⚪ —<br>**🔴 report missing** | Observed: ⚪ —<br>**⚪ not run** | Observed: ⚪ —<br>**⚪ not run** | 🔴 failed |
