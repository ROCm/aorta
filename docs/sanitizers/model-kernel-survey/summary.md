# Sanitizers Nightly · gfx950

Run `model-kernel-survey` · commit `local` · 2026-10-14 12:56:42 UTC

❌ **INCOMPLETE** — 3/3 sanitizer report(s) are missing

Observed `WARN` or `FAIL` verdicts may be expected positive-control outcomes. Baseline status is the regression-health signal.

| Recipe | Backend | Baseline status | Observed | Expected | Execution | Findings | Coverage |
|---|---|---|---|---|---|--:|---|
| daily-waitcheck-gemm | waitcheck (static) | ❌ **Report missing** | `—` | `warn` | ❌ **missing** | 0 | — |
| daily-consan-clean | consan (dynamic) | ❌ **Report missing** | `—` | `pass` | ❌ **missing** | 0 | — |
| daily-consan-racy | consan (dynamic) | ❌ **Report missing** | `—` | `fail` | ❌ **missing** | 0 | — |

Two views below: **Expected behavior (guardrails)** (baseline-checked, the gate) and **Workload survey (observed-only)** (non-gating).

## Expected behavior (guardrails) · Kernel details

<details><summary><b>daily-waitcheck-gemm</b> — ❌ **Report missing**</summary>

Observed sanitizer verdict: `—`

Observation: report missing

</details>

<details><summary><b>daily-consan-clean</b> — ❌ **Report missing**</summary>

Observed sanitizer verdict: `—`

Observation: report missing

</details>

<details><summary><b>daily-consan-racy</b> — ❌ **Report missing**</summary>

Observed sanitizer verdict: `—`

Observation: report missing

</details>

## Workload survey (observed-only)

How real GPU kernels behave under AMD's sanitizers — **waitcheck** (static `s_waitcnt` wait-count scan) and **ConSan** (dynamic data-race check); where both produced a report the kernel is shown under each, and a scan that was skipped or whose report is missing still appears, marked report missing with no verdict. **No expected-behavior comparison on this tab**; an `error` / `fail` / `warn` here is an observation of how the kernel behaved, not a regression. Each case lists a copy-paste command to reproduce the run.

Surveyed 4 workloads (80 kernels) across 8 sanitizer runs — 4 pass · 1 warn · 1 fail · 2 error

| Workload | Kernels | waitcheck | ConSan | Findings | Note |
|---|--:|---|---|--:|---|
| deepseek v4 | 32 | `pass` | `error` | 0 | 8 of 11 kernels did not complete (7x combined_hook_exit_86, 1x combined_hook_exit_-6) |
| glm 5.3 | 2 | `pass` | `pass` | 0 | — |
| gpt oss | 43 | `warn` | `fail` | 6 | sampled_conflict |
| qwen3 | 3 | `pass` | `error` | 0 | 1 of 1 kernels did not complete (1x combined_hook_exit_86) |

<details><summary><b>consan-deepseek-v4</b> — observed `error`</summary>

Observation: consan error; reason 8 of 11 kernels did not complete (7x combined_hook_exit_86, 1x combined_hook_exit_-6); _combine_topk_swa_indices_kernel: combined_hook_exit_86, _compress_gather_split_sparse_attn: combined_hook_exit_86, _fill_short_context_topk_indices: combined_hook_exit_86, _finalize_norm_rope_quant_store_sparse_attn: combined_hook_exit_-6, _inverse_rope_gptj_kernel: combined_hook_exit_86, _pack_dense_prefix_to_ragged_kernel: combined_hook_exit_86, _pack_global_topk_ragged_kernel: combined_hook_exit_86, _sparse_attn_decode_reduce_kernel: combined_hook_exit_86

Reason: `_combine_topk_swa_indices_kernel: combined_hook_exit_86; _compress_gather_split_sparse_attn: combined_hook_exit_86; _fill_short_context_topk_indices: combined_hook_exit_86; _finalize_norm_rope_quant_store_sparse_attn: combined_hook_exit_-6…`

Reproduce: `aorta sweep run --recipe recipes/sanitizers/daily-consan-deepseek-v4.yaml`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `_combine_topk_swa_indices_kernel` | 1 | `error` | 0 | `dispatched.hsaco` | `e4ad357286` | combined_hook_exit_86 |
| `_compress_gather_split_sparse_attn` | 1 | `error` | 0 | `dispatched.hsaco` | `39bdba3f63` | combined_hook_exit_86 |
| `_compute_topk_lens_kernel` | 1 | `pass` | 0 | `dispatched.hsaco` | `71faafe5e2` | — |
| `_fill_short_context_topk_indices` | 1 | `error` | 0 | `dispatched.hsaco` | `89e91399e5` | combined_hook_exit_86 |
| `_finalize_norm_rope_quant_store_sparse_attn` | 1 | `error` | 0 | `dispatched.hsaco` | `c915363658` | combined_hook_exit_-6 |
| `_inverse_rope_gptj_kernel` | 1 | `error` | 0 | `dispatched.hsaco` | `45cdc0fa70` | combined_hook_exit_86 |
| `_pack_dense_prefix_to_ragged_kernel` | 1 | `error` | 0 | `dispatched.hsaco` | `a057ab2905` | combined_hook_exit_86 |
| `_pack_global_topk_ragged_kernel` | 1 | `error` | 0 | `dispatched.hsaco` | `28106f4285` | combined_hook_exit_86 |
| `_sparse_attn_decode_reduce_kernel` | 1 | `error` | 0 | `dispatched.hsaco` | `35d592d7ef` | combined_hook_exit_86 |
| `_sparse_attn_prefill_ragged_kernel` | 1 | `pass` | 0 | `dispatched.hsaco` | `57dea6f61f` | — |
| `_token_request_mapping_kernel` | 1 | `pass` | 0 | `dispatched.hsaco` | `7b759ad406` | — |

</details>

<details><summary><b>consan-glm-5.3</b> — observed `pass`</summary>

Observation: consan pass

Reproduce: `aorta sweep run --recipe recipes/sanitizers/daily-consan-glm-5.3.yaml`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `rotary_kernel` | 1 | `pass` | 0 | `dispatched.hsaco` | `666743abf4` | — |

</details>

<details><summary><b>consan-gpt-oss</b> — observed `fail`</summary>

Observation: consan fail; reason 30 of 38 kernels did not complete (21x combined_hook_exit_86, 9x consan_output_parse_error: access site count mismatch); _combine_sampled_and_draft_tokens_kernel: combined_hook_exit_86, _compute_slot_mappings_kernel: combined_hook_exit_86, _gather_block_tables_kernel: combined_hook_exit_86, _get_num_sampled_and_rejected_kernel: combined_hook_exit_86, _masked_compaction: combined_hook_exit_86, _matmul_ogs: consan_output_parse_error: reader 425028496 access site count mismatch, _matmul_ogs@w2: consan_output_parse_error: reader 425105568 access site count mismatch, _post_update_kernel: combined_hook_exit_86, _prepare_pos_seq_lens_kernel: combined_hook_exit_86, _prepare_prefill_inputs_kernel: combined_hook_exit_86, _rmsnorm_bwd_dg_reduce_triton: consan_output_parse_error: reader 423397536 access site count mismatch, _rope_kernel_cached_thd_2c_gqa_bwd: combined_hook_exit_86, _rope_kernel_cached_thd_2c_gqa_fwd: combined_hook_exit_86, _rope_kernel_cached_thd_2c_gqa_onehead_bwd: combined_hook_exit_86, _rope_kernel_cached_thd_2c_gqa_onehead_fwd: combined_hook_exit_86, _rope_kernel_sbhd_bwd: combined_hook_exit_86, _rope_kernel_sbhd_cached_bwd: combined_hook_exit_86, _rope_kernel_sbhd_cached_fwd: combined_hook_exit_86, _rope_kernel_sbhd_fwd: combined_hook_exit_86, _rope_kernel_thd_bwd: combined_hook_exit_86, _rope_kernel_thd_cached_2c_bwd: combined_hook_exit_86, _rope_kernel_thd_cached_2c_fwd: combined_hook_exit_86, _rope_kernel_thd_fwd: combined_hook_exit_86, _swiglu: consan_output_parse_error: reader 426307376 access site count mismatch, _topk_topp_kernel: consan_output_parse_error: reader 423009760 access site count mismatch, kernel_unified_attention_2d: consan_output_parse_error: reader 422986320 access site count mismatch, kernel_unified_attention_3d: consan_output_parse_error: reader 425587856 access site count mismatch, pack_bitmatrix: consan_output_parse_error: reader 423893744 access site count mismatch, reduce_segments: consan_output_parse_error: reader 422120448 access site count mismatch, reshape_and_cache_kernel_flash: combined_hook_exit_86; 3 finding(s) (sampled_conflict)

Finding: `[rocjitsu-dbi-hooks] ConSan conflict reader=423133376 first_index=112 second_index=118 first_kind=2 second_kind=2 first_owner=1 second_owner=3 epoch=8 generati…`

Reproduce: `aorta sweep run --recipe recipes/sanitizers/daily-consan-gpt-oss.yaml`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `_combine_sampled_and_draft_tokens_kernel` | 1 | `error` | 0 | `dispatched.hsaco` | `0ce43b2c5b` | combined_hook_exit_86 |
| `_compute_slot_mappings_kernel` | 1 | `error` | 0 | `dispatched.hsaco` | `95dd35be7f` | combined_hook_exit_86 |
| `_fused_add_rmsnorm_kernel` | 1 | `pass` | 0 | `dispatched.hsaco` | `6aa48fd479` | — |
| `_fused_add_rmsnorm_pad` | 1 | `pass` | 0 | `dispatched.hsaco` | `cf3206092f` | — |
| `_gather_block_tables_kernel` | 1 | `error` | 0 | `dispatched.hsaco` | `8cd6cf1378` | combined_hook_exit_86 |
| `_get_num_sampled_and_rejected_kernel` | 1 | `error` | 0 | `dispatched.hsaco` | `3a838a2498` | combined_hook_exit_86 |
| `_masked_compaction` | 1 | `error` | 0 | `dispatched.hsaco` | `6f836ca1d0` | combined_hook_exit_86 |
| `_matmul_ogs` | 1 | `error` | 0 | `dispatched.hsaco` | `dd24bae8f6` | consan_output_parse_error: reader 425028496 access site count mismatch |
| `_matmul_ogs@w2` | 1 | `error` | 0 | `dispatched.hsaco` | `ecbd1616e8` | consan_output_parse_error: reader 425105568 access site count mismatch |
| `_post_update_kernel` | 1 | `error` | 0 | `dispatched.hsaco` | `d704821bc8` | combined_hook_exit_86 |
| `_prepare_pos_seq_lens_kernel` | 1 | `error` | 0 | `dispatched.hsaco` | `f6f47aa910` | combined_hook_exit_86 |
| `_prepare_prefill_inputs_kernel` | 1 | `error` | 0 | `dispatched.hsaco` | `ff27c2b755` | combined_hook_exit_86 |
| `_quant_fused_add_rmsnorm_kernel` | 1 | `pass` | 0 | `dispatched.hsaco` | `526859094e` | — |
| `_quant_rms_norm_kernel` | 1 | `pass` | 0 | `dispatched.hsaco` | `21355c5f5f` | — |
| `_rms_norm_kernel` | 1 | `pass` | 0 | `dispatched.hsaco` | `6dfc789bbf` | — |
| `_rmsnorm_bwd_dg_reduce_triton` | 1 | `error` | 0 | `dispatched.hsaco` | `602edbef8d` | consan_output_parse_error: reader 423397536 access site count mismatch |
| `_rmsnorm_bwd_triton` | 1 | `pass` | 0 | `dispatched.hsaco` | `1354093922` | — |
| `_rope_kernel_cached_thd_2c_gqa_bwd` | 1 | `error` | 0 | `dispatched.hsaco` | `10021ea2ef` | combined_hook_exit_86 |
| `_rope_kernel_cached_thd_2c_gqa_fwd` | 1 | `error` | 0 | `dispatched.hsaco` | `8436cfed0d` | combined_hook_exit_86 |
| `_rope_kernel_cached_thd_2c_gqa_onehead_bwd` | 1 | `error` | 0 | `dispatched.hsaco` | `cbafb2a013` | combined_hook_exit_86 |
| `_rope_kernel_cached_thd_2c_gqa_onehead_fwd` | 1 | `error` | 0 | `dispatched.hsaco` | `f21088f8fa` | combined_hook_exit_86 |
| `_rope_kernel_sbhd_bwd` | 1 | `error` | 0 | `dispatched.hsaco` | `5520ae50ed` | combined_hook_exit_86 |
| `_rope_kernel_sbhd_cached_bwd` | 1 | `error` | 0 | `dispatched.hsaco` | `89b18133c5` | combined_hook_exit_86 |
| `_rope_kernel_sbhd_cached_fwd` | 1 | `error` | 0 | `dispatched.hsaco` | `a42c772989` | combined_hook_exit_86 |
| `_rope_kernel_sbhd_fwd` | 1 | `error` | 0 | `dispatched.hsaco` | `3e1f620f26` | combined_hook_exit_86 |
| `_rope_kernel_thd_bwd` | 1 | `error` | 0 | `dispatched.hsaco` | `039fddf778` | combined_hook_exit_86 |
| `_rope_kernel_thd_cached_2c_bwd` | 1 | `error` | 0 | `dispatched.hsaco` | `2fcae84748` | combined_hook_exit_86 |
| `_rope_kernel_thd_cached_2c_fwd` | 1 | `error` | 0 | `dispatched.hsaco` | `6e47d429df` | combined_hook_exit_86 |
| `_rope_kernel_thd_fwd` | 1 | `error` | 0 | `dispatched.hsaco` | `f7af5dc7ca` | combined_hook_exit_86 |
| `_stage2_pow2` | 1 | `pass` | 0 | `dispatched.hsaco` | `e44487229e` | — |
| `_swiglu` | 1 | `error` | 0 | `dispatched.hsaco` | `ce24e7a74f` | consan_output_parse_error: reader 426307376 access site count mismatch |
| `_topk_forward` | 1 | `fail` | 3 | `dispatched.hsaco` | `6840d83daa` | — |
| `_topk_topp_kernel` | 1 | `error` | 0 | `dispatched.hsaco` | `ad9c0d27b1` | consan_output_parse_error: reader 423009760 access site count mismatch |
| `kernel_unified_attention_2d` | 1 | `error` | 0 | `dispatched.hsaco` | `0cdaf63bba` | consan_output_parse_error: reader 422986320 access site count mismatch |
| `kernel_unified_attention_3d` | 1 | `error` | 0 | `dispatched.hsaco` | `101cc9099f` | consan_output_parse_error: reader 425587856 access site count mismatch |
| `pack_bitmatrix` | 1 | `error` | 0 | `dispatched.hsaco` | `c28e22fce5` | consan_output_parse_error: reader 423893744 access site count mismatch |
| `reduce_segments` | 1 | `error` | 0 | `dispatched.hsaco` | `248183ee2f` | consan_output_parse_error: reader 422120448 access site count mismatch |
| `reshape_and_cache_kernel_flash` | 1 | `error` | 0 | `dispatched.hsaco` | `2b89739cec` | combined_hook_exit_86 |

</details>

<details><summary><b>consan-qwen3</b> — observed `error`</summary>

Observation: consan error; reason 1 of 1 kernels did not complete (1x combined_hook_exit_86); _compute_slot_mapping_kernel: combined_hook_exit_86

Reason: `1 of 1 kernels did not complete (1x combined_hook_exit_86) — _compute_slot_mapping_kernel: combined_hook_exit_86`

Reproduce: `aorta sweep run --recipe recipes/sanitizers/daily-consan-qwen3.yaml`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `_compute_slot_mapping_kernel` | 1 | `error` | 0 | `dispatched.hsaco` | `d1d7e2e970` | combined_hook_exit_86 |

</details>

<details><summary><b>waitcheck-deepseek-v4</b> — observed `pass`</summary>

Observation: waitcheck pass

Reproduce: `aorta sweep run --recipe recipes/sanitizers/daily-waitcheck-deepseek-v4.yaml`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `_compress_gather_split_sparse_attn` | 1 | `pass` | 0 | `_compress_gather_split_sparse_attn__C0.hsaco` | `01b947eb50` | — |
| `ComputePrefillMetadataKernel` | 1 | `pass` | 0 | `kernel_ComputePrefillMetadataKernel__C4.hsaco` | `023589839e` | — |
| `_combine_topk_swa_indices_kernel` | 1 | `pass` | 0 | `_combine_topk_swa_indices_kernel__C5.hsaco` | `05b2823398` | — |
| `FusedQKVRMSNormKernel` | 1 | `pass` | 0 | `kernel_FusedQKVRMSNormKernel__C0.hsaco` | `0940d0e818` | — |
| `_topp_sb_mask_kernel` | 1 | `pass` | 0 | `_topp_sb_mask_kernel__C5.hsaco` | `0d95a74bb1` | — |
| `_topp_sb_mask_kernel` | 1 | `pass` | 0 | `_topp_sb_mask_kernel__C2.hsaco` | `142a4f4649` | — |
| `_pack_global_topk_ragged_kernel` | 1 | `pass` | 0 | `_pack_global_topk_ragged_kernel__C0.hsaco` | `1c619f1df0` | — |
| `_topp_sb_step_kernel` | 1 | `pass` | 0 | `_topp_sb_step_kernel__C6.hsaco` | `1ee78d3fdb` | — |
| `_fill_short_context_topk_indices` | 1 | `pass` | 0 | `_fill_short_context_topk_indices__C0.hsaco` | `2295afb8c8` | — |
| `_topp_sb_stats_kernel` | 1 | `pass` | 0 | `_topp_sb_stats_kernel__C6.hsaco` | `29be5ab12b` | — |
| `SavePartialStatesKernel` | 1 | `pass` | 0 | `kernel_SavePartialStatesKernel__C1.hsaco` | `2c02e018f4` | — |
| `ComputePrefillMetadataKernel` | 1 | `pass` | 0 | `kernel_ComputePrefillMetadataKernel__C0.hsaco` | `330afebb35` | — |
| `ComputeSlotMappingKernel` | 1 | `pass` | 0 | `kernel_ComputeSlotMappingKernel__C5.hsaco` | `3731d9220c` | — |
| `_compute_swa_indices_and_lens_kernel` | 1 | `pass` | 0 | `_compute_swa_indices_and_lens_kernel__C0.hsaco` | `3866bde62a` | — |
| `_fused_kv_compress_norm_rope_insert_sparse_attn` | 1 | `pass` | 0 | `_fused_kv_compress_norm_rope_insert_sparse_attn__C2.hsaco` | `3dea04ee49` | — |
| `_topp_sb_mask_kernel` | 1 | `pass` | 0 | `_topp_sb_mask_kernel__C4.hsaco` | `3f7c4b4466` | — |
| `ComputeSlotMappingKernel` | 1 | `pass` | 0 | `kernel_ComputeSlotMappingKernel__C4.hsaco` | `4affb18e29` | — |
| `_topk_topp_kernel` | 1 | `pass` | 0 | `_topk_topp_kernel__C1.hsaco` | `4cd381d3ca` | — |
| `_combine_topk_swa_indices_kernel` | 1 | `pass` | 0 | `_combine_topk_swa_indices_kernel__C3.hsaco` | `4d9bab4140` | — |
| `_topp_sb_stats_kernel` | 1 | `pass` | 0 | `_topp_sb_stats_kernel__C0.hsaco` | `4e450eb0f8` | — |
| `DequantizeAndGatherKCacheKernel` | 1 | `pass` | 0 | `kernel_DequantizeAndGatherKCacheKernel__C1.hsaco` | `4edbd8229d` | — |
| `_topk_topp_kernel` | 1 | `pass` | 0 | `_topk_topp_kernel__C7.hsaco` | `4ef16edc9e` | — |
| `ComputePrefillMetadataKernel` | 1 | `pass` | 0 | `kernel_ComputePrefillMetadataKernel__C2.hsaco` | `52c7d523aa` | — |
| `ComputePrefillMetadataKernel` | 1 | `pass` | 0 | `kernel_ComputePrefillMetadataKernel__C6.hsaco` | `5411e59716` | — |
| `_sparse_attn_decode_reduce_kernel` | 1 | `pass` | 0 | `_sparse_attn_decode_reduce_kernel__C0.hsaco` | `5e2f3c523c` | — |
| `_topp_sb_step_kernel` | 1 | `pass` | 0 | `_topp_sb_step_kernel__C1.hsaco` | `61b4d24259` | — |
| `_topp_sb_stats_kernel` | 1 | `pass` | 0 | `_topp_sb_stats_kernel__C3.hsaco` | `64b84187e7` | — |
| `_topp_sb_mask_kernel` | 1 | `pass` | 0 | `_topp_sb_mask_kernel__C3.hsaco` | `6662f8ed8b` | — |
| `_sparse_attn_prefill_ragged_kernel` | 1 | `pass` | 0 | `_sparse_attn_prefill_ragged_kernel__C0.hsaco` | `6819a84ca7` | — |
| `_topp_sb_stats_kernel` | 1 | `pass` | 0 | `_topp_sb_stats_kernel__C4.hsaco` | `68543094f7` | — |
| `CompressedSlotMappingKernel` | 1 | `pass` | 0 | `kernel_CompressedSlotMappingKernel__C0.hsaco` | `6e6c90b0d6` | — |
| `_topp_sb_stats_kernel` | 1 | `pass` | 0 | `_topp_sb_stats_kernel__C5.hsaco` | `70c73fe576` | — |
| `ComputeSlotMappingKernel` | 1 | `pass` | 0 | `kernel_ComputeSlotMappingKernel__C0.hsaco` | `71b81133c0` | — |
| `PrepareUniformDecodeKernel` | 1 | `pass` | 0 | `kernel_PrepareUniformDecodeKernel__C0.hsaco` | `747c73bb71` | — |
| `_token_request_mapping_kernel` | 1 | `pass` | 0 | `_token_request_mapping_kernel__C0.hsaco` | `77197bf68f` | — |
| `_fused_kv_compress_norm_rope_insert_indexer_attn` | 1 | `pass` | 0 | `_fused_kv_compress_norm_rope_insert_indexer_attn__C6.hsaco` | `772792361a` | — |
| `_topp_sb_stats_kernel` | 1 | `pass` | 0 | `_topp_sb_stats_kernel__C1.hsaco` | `7a119a8bd3` | — |
| `ComputePrefillMetadataKernel` | 1 | `pass` | 0 | `kernel_ComputePrefillMetadataKernel__C1.hsaco` | `7eae407156` | — |
| `CombineTopkSwaIndicesKernel` | 1 | `pass` | 0 | `kernel_CombineTopkSwaIndicesKernel__C2.hsaco` | `81848517bb` | — |
| `PackSeqTritonKernel` | 1 | `pass` | 0 | `kernel_PackSeqTritonKernel__C0.hsaco` | `81b889d7c0` | — |
| `_finalize_norm_rope_quant_store_sparse_attn` | 1 | `pass` | 0 | `_finalize_norm_rope_quant_store_sparse_attn__C0.hsaco` | `8937da5629` | — |
| `FusedIndexerQRopeQuantTritonKernel` | 1 | `pass` | 0 | `kernel_FusedIndexerQRopeQuantTritonKernel__C0.hsaco` | `89cb70bea4` | — |
| `ComputePrefillMetadataKernel` | 1 | `pass` | 0 | `kernel_ComputePrefillMetadataKernel__C7.hsaco` | `8b2cff46f2` | — |
| `UnpackSeqTritonKernel` | 1 | `pass` | 0 | `kernel_UnpackSeqTritonKernel__C0.hsaco` | `96ca07c323` | — |
| `_topk_topp_kernel` | 1 | `pass` | 0 | `_topk_topp_kernel__C0.hsaco` | `9dc72f9d4e` | — |
| `_zero_kv_blocks_kernel` | 1 | `pass` | 0 | `_zero_kv_blocks_kernel__C0.hsaco` | `a85ce7fb66` | — |
| `BuildPrefillChunkMetadataKernel` | 1 | `pass` | 0 | `kernel_BuildPrefillChunkMetadataKernel__C0.hsaco` | `aa5b051cee` | — |
| `ComputePrefillMetadataKernel` | 1 | `pass` | 0 | `kernel_ComputePrefillMetadataKernel__C3.hsaco` | `ae769653de` | — |
| `BuildPrefillChunkMetadataKernel` | 1 | `pass` | 0 | `kernel_BuildPrefillChunkMetadataKernel__C1.hsaco` | `b0d7bc5694` | — |
| `ComputePrefillMetadataKernel` | 1 | `pass` | 0 | `kernel_ComputePrefillMetadataKernel__C5.hsaco` | `b2ae28d1e0` | — |
| `_topp_sb_mask_kernel` | 1 | `pass` | 0 | `_topp_sb_mask_kernel__C0.hsaco` | `b4a698c534` | — |
| `ComputeSlotMappingKernel` | 1 | `pass` | 0 | `kernel_ComputeSlotMappingKernel__C3.hsaco` | `b7c9ea7951` | — |
| `_topp_sb_mask_kernel` | 1 | `pass` | 0 | `_topp_sb_mask_kernel__C7.hsaco` | `ba2cbce2f9` | — |
| `_fused_kv_compress_norm_rope_insert_indexer_attn` | 1 | `pass` | 0 | `_fused_kv_compress_norm_rope_insert_indexer_attn__C0.hsaco` | `bf19622aab` | — |
| `CompressedSlotMappingKernel` | 1 | `pass` | 0 | `kernel_CompressedSlotMappingKernel__C4.hsaco` | `c1151bedf3` | — |
| `_topp_sb_stats_kernel` | 1 | `pass` | 0 | `_topp_sb_stats_kernel__C2.hsaco` | `c71cd8e9cf` | — |
| `CombineTopkSwaIndicesKernel` | 1 | `pass` | 0 | `kernel_CombineTopkSwaIndicesKernel__C1.hsaco` | `c9d44e92f5` | — |
| `_combine_topk_swa_indices_kernel` | 1 | `pass` | 0 | `_combine_topk_swa_indices_kernel__C0.hsaco` | `d45a7ee219` | — |
| `SavePartialStatesKernel` | 1 | `pass` | 0 | `kernel_SavePartialStatesKernel__C0.hsaco` | `d550ec71a6` | — |
| `BuildC128ATopkMetadataKernel` | 1 | `pass` | 0 | `kernel_BuildC128ATopkMetadataKernel__C0.hsaco` | `d7176337bb` | — |
| `_inverse_rope_gptj_kernel` | 1 | `pass` | 0 | `_inverse_rope_gptj_kernel__C0.hsaco` | `db84f8104b` | — |
| `_topp_sb_mask_kernel` | 1 | `pass` | 0 | `_topp_sb_mask_kernel__C6.hsaco` | `e10008eed7` | — |
| `_compute_topk_lens_kernel` | 1 | `pass` | 0 | `_compute_topk_lens_kernel__C0.hsaco` | `e233231876` | — |
| `_topp_sb_stats_kernel` | 1 | `pass` | 0 | `_topp_sb_stats_kernel__C7.hsaco` | `e73eb9a3f5` | — |
| `BuildPrefillChunkMetadataKernel` | 1 | `pass` | 0 | `kernel_BuildPrefillChunkMetadataKernel__C2.hsaco` | `f05c045758` | — |
| `SavePartialStatesKernel` | 1 | `pass` | 0 | `kernel_SavePartialStatesKernel__C4.hsaco` | `f25f5ad19a` | — |
| `_pack_dense_prefix_to_ragged_kernel` | 1 | `pass` | 0 | `_pack_dense_prefix_to_ragged_kernel__C0.hsaco` | `f2e27c93be` | — |
| `_fused_kv_compress_norm_rope_insert_sparse_attn` | 1 | `pass` | 0 | `_fused_kv_compress_norm_rope_insert_sparse_attn__C0.hsaco` | `f4ea02304e` | — |
| `CombineTopkSwaIndicesKernel` | 1 | `pass` | 0 | `kernel_CombineTopkSwaIndicesKernel__C0.hsaco` | `f61121fb9e` | — |
| `_topp_sb_step_kernel` | 1 | `pass` | 0 | `_topp_sb_step_kernel__C7.hsaco` | `fbbbedfd90` | — |
| `ComputeSlotMappingKernel` | 1 | `pass` | 0 | `kernel_ComputeSlotMappingKernel__C1.hsaco` | `fcc9ac7395` | — |
| `_topp_sb_mask_kernel` | 1 | `pass` | 0 | `_topp_sb_mask_kernel__C1.hsaco` | `ff0a95bdeb` | — |
| `_topp_sb_step_kernel` | 1 | `pass` | 0 | `_topp_sb_step_kernel__C0.hsaco` | `ff4a928ef9` | — |

</details>

<details><summary><b>waitcheck-glm-5.3</b> — observed `pass`</summary>

Observation: waitcheck pass

Reproduce: `aorta sweep run --recipe recipes/sanitizers/daily-waitcheck-glm-5.3.yaml`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `rotary_kernel` | 1 | `pass` | 0 | `rotary_kernel__C0.hsaco` | `1faae11072` | — |
| `_fwht_quant_kernel` | 1 | `pass` | 0 | `_fwht_quant_kernel__C0.hsaco` | `f042893e3e` | — |

</details>

<details><summary><b>waitcheck-gpt-oss</b> — observed `warn`</summary>

Observation: waitcheck warn; 3 finding(s) (wait_hazard)

Finding: `kernel_unified_attention_3d__X1_full.hsaco:gfx950[0]:.text+0x1648: missing s_waitcnt vmcnt(62) before use of v65`

Reproduce: `aorta sweep run --recipe recipes/sanitizers/daily-waitcheck-gpt-oss.yaml`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `_rmsnorm_bwd_triton` | 1 | `pass` | 0 | `_rmsnorm_bwd_triton__D3.hsaco` | `01099733d3` | — |
| `_quant_rms_norm_kernel` | 1 | `pass` | 0 | `_quant_rms_norm_kernel__D2.hsaco` | `047ca3ef54` | — |
| `_rms_norm_kernel` | 1 | `pass` | 0 | `_rms_norm_kernel__D1.hsaco` | `0487364ba1` | — |
| `_rms_norm_kernel` | 1 | `pass` | 0 | `_rms_norm_kernel__D2.hsaco` | `06623fadc4` | — |
| `_matmul_ogs_TTT_bf16xbf16xmxfp4_32x256x128x1` | 1 | `pass` | 0 | `_matmul_ogs_w2__D1.hsaco` | `0d59b24c00` | — |
| `pack_bitmatrix` | 1 | `pass` | 0 | `pack_bitmatrix__D1.hsaco` | `10335c95fe` | — |
| `_compute_slot_mappings_kernel` | 1 | `pass` | 0 | `_compute_slot_mappings_kernel__D1.hsaco` | `126f1f5a13` | — |
| `_matmul_ogs_TTT_bf16xbf16xmxfp4_64x256x128x1_swiglu` | 1 | `pass` | 0 | `_matmul_ogs__P3.hsaco` | `13bcdc5a78` | — |
| `_rope_kernel_cached_thd_2c_gqa_onehead_bwd` | 1 | `pass` | 0 | `_rope_kernel_cached_thd_2c_gqa_onehead_bwd__D1.hsaco` | `15e49e0c4f` | — |
| `_quant_rms_norm_kernel` | 1 | `pass` | 0 | `_quant_rms_norm_kernel__D1.hsaco` | `1c2b22663e` | — |
| `_rms_norm_kernel` | 1 | `pass` | 0 | `_rms_norm_kernel_validation__P2.hsaco` | `1d6a3e43fc` | — |
| `_topk_forward` | 1 | `pass` | 0 | `_topk_forward__D1.hsaco` | `230ed63e83` | — |
| `kernel_unified_attention_2d` | 1 | `pass` | 0 | `kernel_unified_attention_2d__P1_full.hsaco` | `23afda1caa` | — |
| `_fused_add_rmsnorm_kernel` | 1 | `pass` | 0 | `_fused_add_rmsnorm_kernel__D1.hsaco` | `292bbb4985` | — |
| `_post_update_kernel` | 1 | `pass` | 0 | `_post_update_kernel__D1.hsaco` | `2e4d5e12b5` | — |
| `_topk_topp_kernel` | 1 | `pass` | 0 | `_topk_topp_kernel__D1.hsaco` | `32dc50175f` | — |
| `_rms_norm_kernel` | 1 | `pass` | 0 | `_rms_norm_kernel__P2.hsaco` | `35f2996961` | — |
| `_rope_kernel_sbhd_bwd` | 1 | `pass` | 0 | `_rope_kernel_sbhd_bwd__D1.hsaco` | `36259ec778` | — |
| `_rope_kernel_sbhd_cached_bwd` | 1 | `pass` | 0 | `_rope_kernel_sbhd_cached_bwd__D1.hsaco` | `38d9c1f4da` | — |
| `_quant_fused_add_rmsnorm_kernel` | 1 | `pass` | 0 | `_quant_fused_add_rmsnorm_kernel__D3.hsaco` | `3968404a47` | — |
| `_fused_add_rmsnorm_pad` | 1 | `pass` | 0 | `_fused_add_rmsnorm_pad__D1.hsaco` | `3ae7f83a35` | — |
| `_rope_kernel_cached_thd_2c_gqa_bwd` | 1 | `pass` | 0 | `_rope_kernel_cached_thd_2c_gqa_bwd__D1.hsaco` | `3f47e89f13` | — |
| `reduce_segments` | 1 | `pass` | 0 | `reduce_segments__X2_full.hsaco` | `4840f96d49` | — |
| `_prepare_pos_seq_lens_kernel` | 1 | `pass` | 0 | `_prepare_pos_seq_lens_kernel__D1.hsaco` | `4ffcc54bd4` | — |
| `_rope_kernel_cached_thd_2c_gqa_onehead_fwd` | 1 | `pass` | 0 | `_rope_kernel_cached_thd_2c_gqa_onehead_fwd__D1.hsaco` | `5001b8303e` | — |
| `_rope_kernel_thd_cached_2c_fwd` | 1 | `pass` | 0 | `_rope_kernel_thd_cached_2c_fwd__D1.hsaco` | `50056eec3d` | — |
| `kernel_unified_attention_2d` | 1 | `pass` | 0 | `kernel_unified_attention_2d__P1_swa.hsaco` | `50eb412196` | — |
| `_rmsnorm_bwd_triton` | 1 | `pass` | 0 | `_rmsnorm_bwd_triton__D1.hsaco` | `546d4f5373` | — |
| `kernel_unified_attention_3d` | 1 | `warn` | 1 | `kernel_unified_attention_3d__X1_full.hsaco` | `5580a2ed91` | — |
| `_rmsnorm_bwd_triton` | 1 | `pass` | 0 | `_rmsnorm_bwd_triton__D4.hsaco` | `581ad05967` | — |
| `_swiglu_bf16xbf16_16x128` | 1 | `pass` | 0 | `_swiglu__D1.hsaco` | `5bcfc5d0ed` | — |
| `_rmsnorm_bwd_triton` | 1 | `pass` | 0 | `_rmsnorm_bwd_triton__P3.hsaco` | `5fffc22077` | — |
| `_quant_fused_add_rmsnorm_kernel` | 1 | `pass` | 0 | `_quant_fused_add_rmsnorm_kernel__P2.hsaco` | `6177c7848d` | — |
| `_rope_kernel_thd_bwd` | 1 | `pass` | 0 | `_rope_kernel_thd_bwd__D1.hsaco` | `6a326c568e` | — |
| `_rope_kernel_sbhd_fwd` | 1 | `pass` | 0 | `_rope_kernel_sbhd_fwd__D1.hsaco` | `6bcb4a02b2` | — |
| `_fused_add_rmsnorm_kernel` | 1 | `pass` | 0 | `_fused_add_rmsnorm_kernel__P2.hsaco` | `6c60a17f02` | — |
| `_rms_norm_kernel` | 1 | `pass` | 0 | `_rms_norm_kernel__D4.hsaco` | `6d338eb001` | — |
| `_fused_add_rmsnorm_kernel` | 1 | `pass` | 0 | `_fused_add_rmsnorm_kernel__D4.hsaco` | `706641822f` | — |
| `_quant_rms_norm_kernel` | 1 | `pass` | 0 | `_quant_rms_norm_kernel__D3.hsaco` | `7476d67e52` | — |
| `_get_num_sampled_and_rejected_kernel` | 1 | `pass` | 0 | `_get_num_sampled_and_rejected_kernel__D1.hsaco` | `764eba65ae` | — |
| `_matmul_ogs_TTT_bf16xbf16xmxfp4_32x256x128x1` | 1 | `pass` | 0 | `_matmul_ogs_w2__D2.hsaco` | `76d194435a` | — |
| `_quant_rms_norm_kernel` | 1 | `pass` | 0 | `_quant_rms_norm_kernel__P2.hsaco` | `78006f52d4` | — |
| `kernel_unified_attention_2d` | 1 | `pass` | 0 | `kernel_unified_attention_2d__D1_full.hsaco` | `7a29ce1309` | — |
| `_fused_add_rmsnorm_kernel` | 1 | `pass` | 0 | `_fused_add_rmsnorm_kernel__P1.hsaco` | `7bf1a00117` | — |
| `_combine_sampled_and_draft_tokens_kernel` | 1 | `pass` | 0 | `_combine_sampled_and_draft_tokens_kernel__D1.hsaco` | `7d4b1132b2` | — |
| `_rope_kernel_thd_fwd` | 1 | `pass` | 0 | `_rope_kernel_thd_fwd__D1.hsaco` | `84a257b3be` | — |
| `_fused_add_rmsnorm_kernel` | 1 | `pass` | 0 | `_fused_add_rmsnorm_kernel__D3.hsaco` | `858ec8262a` | — |
| `_quant_fused_add_rmsnorm_kernel` | 1 | `pass` | 0 | `_quant_fused_add_rmsnorm_kernel__P3.hsaco` | `88fc6285dc` | — |
| `_stage2_pow2` | 1 | `pass` | 0 | `_stage2_pow2__D1.hsaco` | `8a397730c8` | — |
| `reduce_segments` | 1 | `pass` | 0 | `reduce_segments__X1_full.hsaco` | `8b44411b89` | — |
| `_rope_kernel_sbhd_cached_fwd` | 1 | `pass` | 0 | `_rope_kernel_sbhd_cached_fwd__D1.hsaco` | `8b60155377` | — |
| `_quant_fused_add_rmsnorm_kernel` | 1 | `pass` | 0 | `_quant_fused_add_rmsnorm_kernel__D4.hsaco` | `8e641def2a` | — |
| `_rmsnorm_bwd_triton` | 1 | `pass` | 0 | `_rmsnorm_bwd_triton__P2.hsaco` | `8f6246a35d` | — |
| `_quant_rms_norm_kernel` | 1 | `pass` | 0 | `_quant_rms_norm_kernel__P3.hsaco` | `92b556b5a6` | — |
| `kernel_unified_attention_3d` | 1 | `warn` | 1 | `kernel_unified_attention_3d__X2_full.hsaco` | `9339121a12` | — |
| `_rms_norm_kernel` | 1 | `pass` | 0 | `_rms_norm_kernel__P1.hsaco` | `9444c2acd4` | — |
| `_quant_fused_add_rmsnorm_kernel` | 1 | `pass` | 0 | `_quant_fused_add_rmsnorm_kernel__D1.hsaco` | `a07700ba95` | — |
| `_gather_block_tables_kernel` | 1 | `pass` | 0 | `_gather_block_tables_kernel__D1.hsaco` | `ab8c3324d1` | — |
| `_quant_fused_add_rmsnorm_kernel` | 1 | `pass` | 0 | `_quant_fused_add_rmsnorm_kernel__P1.hsaco` | `b0c1b84ce9` | — |
| `kernel_unified_attention_2d` | 1 | `pass` | 0 | `kernel_unified_attention_2d__P3_swa.hsaco` | `b35e4a3e75` | — |
| `_rmsnorm_bwd_triton` | 1 | `pass` | 0 | `_rmsnorm_bwd_triton__D2.hsaco` | `b4319de3c7` | — |
| `_quant_fused_add_rmsnorm_kernel` | 1 | `pass` | 0 | `_quant_fused_add_rmsnorm_kernel__D2.hsaco` | `b5354f54fd` | — |
| `reshape_and_cache_kernel_flash` | 1 | `pass` | 0 | `reshape_and_cache_kernel_flash__D1.hsaco` | `bd0653212a` | — |
| `kernel_unified_attention_2d` | 1 | `pass` | 0 | `kernel_unified_attention_2d__P3_full.hsaco` | `be5bd3ce5f` | — |
| `_rmsnorm_bwd_dg_reduce_triton` | 1 | `pass` | 0 | `_rmsnorm_bwd_dg_reduce_triton__D1.hsaco` | `c0f189d496` | — |
| `_rope_kernel_cached_thd_2c_gqa_onehead_bwd` | 1 | `pass` | 0 | `_rope_kernel_cached_thd_2c_gqa_onehead_bwd__D3.hsaco` | `c28b2f07df` | — |
| `_rms_norm_kernel` | 1 | `pass` | 0 | `_rms_norm_kernel__P3.hsaco` | `c4d9dd7268` | — |
| `_rope_kernel_cached_thd_2c_gqa_fwd` | 1 | `pass` | 0 | `_rope_kernel_cached_thd_2c_gqa_fwd__D1.hsaco` | `c9af9646eb` | — |
| `_quant_rms_norm_kernel` | 1 | `pass` | 0 | `_quant_rms_norm_kernel__P1.hsaco` | `d11ffbc4c4` | — |
| `_rms_norm_kernel` | 1 | `pass` | 0 | `_rms_norm_kernel__D3.hsaco` | `d55d343134` | — |
| `_masked_compaction` | 1 | `pass` | 0 | `_masked_compaction__D1.hsaco` | `d7d42053f5` | — |
| `_quant_rms_norm_kernel` | 1 | `pass` | 0 | `_quant_rms_norm_kernel__D4.hsaco` | `e03acc8908` | — |
| `_fused_add_rmsnorm_kernel` | 1 | `pass` | 0 | `_fused_add_rmsnorm_kernel__D2.hsaco` | `e898ba3cc7` | — |
| `_matmul_ogs_TTT_bf16xbf16xmxfp4_32x256x128x1_swiglu` | 1 | `pass` | 0 | `_matmul_ogs__D1.hsaco` | `eb809b063e` | — |
| `_rope_kernel_thd_cached_2c_bwd` | 1 | `pass` | 0 | `_rope_kernel_thd_cached_2c_bwd__D1.hsaco` | `f2a48c1fbf` | — |
| `_matmul_ogs_TTT_bf16xbf16xmxfp4_64x256x128x1` | 1 | `pass` | 0 | `_matmul_ogs_w2__P3.hsaco` | `f45edf28bd` | — |
| `_fused_add_rmsnorm_kernel` | 1 | `pass` | 0 | `_fused_add_rmsnorm_kernel__P3.hsaco` | `f99e96e3ed` | — |
| `kernel_unified_attention_2d` | 1 | `warn` | 1 | `kernel_unified_attention_2d__D1_swa.hsaco` | `fa45648c30` | — |
| `_matmul_ogs_TTT_bf16xbf16xmxfp4_32x256x128x1_swiglu` | 1 | `pass` | 0 | `_matmul_ogs__D2.hsaco` | `fa817a8129` | — |
| `_rmsnorm_bwd_triton` | 1 | `pass` | 0 | `_rmsnorm_bwd_triton__P1.hsaco` | `fc9f8ef204` | — |
| `_prepare_prefill_inputs_kernel` | 1 | `pass` | 0 | `_prepare_prefill_inputs_kernel__D1.hsaco` | `fce5a0c481` | — |
| `_rope_kernel_cached_thd_2c_gqa_onehead_fwd` | 1 | `pass` | 0 | `_rope_kernel_cached_thd_2c_gqa_onehead_fwd__D3.hsaco` | `ff350f718a` | — |

</details>

<details><summary><b>waitcheck-qwen3</b> — observed `pass`</summary>

Observation: waitcheck pass

Reproduce: `aorta sweep run --recipe recipes/sanitizers/daily-waitcheck-qwen3.yaml`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `_fwd_kernel` | 1 | `pass` | 0 | `_fwd_kernel__C0.hsaco` | `652bda291b` | — |
| `_topk_topp_kernel` | 1 | `pass` | 0 | `_topk_topp_kernel__C0.hsaco` | `8cceb417c0` | — |
| `_compute_slot_mapping_kernel` | 1 | `pass` | 0 | `_compute_slot_mapping_kernel__C0.hsaco` | `e1cabd839a` | — |

</details>

## History / trend

| Run | Commit | daily-waitcheck-gemm | daily-consan-clean | daily-consan-racy | Gate |
|---|---|---|---|---|---|
| model-kernel-survey | `local` | ❌ **Report missing**<br>Observed: `—` | ❌ **Report missing**<br>Observed: `—` | ❌ **Report missing**<br>Observed: `—` | Incomplete |
