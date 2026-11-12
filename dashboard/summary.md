# Sanitizers Nightly · gfx950

> ⚠️ **Stale** — latest sanitizer nightly run `43216269914` (2026-11-12 12:21:51 UTC) did not complete successfully (failure); the data below may be stale. [view failed run](https://github.com/ROCm/aorta/actions/runs/43216269914)

Run `2026-11-12T122151-43216269914` · commit `74a407513e6a` · 2026-11-12 12:21:51 UTC

🔴 **Failed** — 5/5 missing

Observed `warn` or `fail` verdicts may be expected positive-control outcomes. A matched row is healthy even when the verdict is fail. Baseline status is matched or unmatched.

| Recipe | Backend | Observed | Expected | Execution | Findings | Coverage | Baseline |
|---|---|---|---|---|--:|---|---|
| daily-waitcheck-gemm | waitcheck (static) | ⚪ — | 🟡 warn | 🔴 missing | 0 | — | **🔴 report missing** |
| daily-consan-clean | consan (dynamic) | ⚪ — | 🟢 pass | 🔴 missing | 0 | — | **🔴 report missing** |
| daily-consan-racy | consan (dynamic) | ⚪ — | 🔴 fail | 🔴 missing | 0 | — | **🔴 report missing** |
| daily-fpsan-match | fpsan (digest) | ⚪ — | 🟢 pass | 🔴 missing | 0 | — | **🔴 report missing** |
| daily-fpsan-mismatch | fpsan (digest) | ⚪ — | 🔴 fail | 🔴 missing | 0 | — | **🔴 report missing** |

Three views below: **Nightly gate** (baseline-checked, the gate), **Workload survey** and **FpSan** (both observed only, non-gating).

## Nightly gate · Kernel details

<details><summary><b>daily-waitcheck-gemm</b> — ⚪ — — **🔴 report missing**</summary>

Observed ⚪ —

Observation: report missing

</details>

<details><summary><b>daily-consan-clean</b> — ⚪ — — **🔴 report missing**</summary>

Observed ⚪ —

Observation: report missing

</details>

<details><summary><b>daily-consan-racy</b> — ⚪ — — **🔴 report missing**</summary>

Observed ⚪ —

Observation: report missing

</details>

<details><summary><b>daily-fpsan-match</b> — ⚪ — — **🔴 report missing**</summary>

Observed ⚪ —

Observation: report missing

</details>

<details><summary><b>daily-fpsan-mismatch</b> — ⚪ — — **🔴 report missing**</summary>

Observed ⚪ —

Observation: report missing

</details>

## Workload survey

How real GPU kernels behave under AMD's sanitizers — **waitcheck** (static `s_waitcnt` wait-count scan) and **ConSan** (dynamic data-race check); where both produced a report the kernel is shown under each, and a scan that was skipped or whose report is missing still appears, marked report missing with no verdict. **No expected-behavior comparison on this tab**; an `error` / `fail` / `warn` here is an observation of how the kernel behaved, not a gate result. Each case lists a copy-paste command to reproduce the run.

No workload-survey kernels in this run.

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
| 2026-11-12T122151-43216269914 | `74a407513e6a` | Observed: ⚪ —<br>**🔴 report missing** | Observed: ⚪ —<br>**🔴 report missing** | Observed: ⚪ —<br>**🔴 report missing** | Observed: ⚪ —<br>**🔴 report missing** | Observed: ⚪ —<br>**🔴 report missing** | 🔴 failed |
| 2026-11-11T122106-43018719581 | `ac11f84616a8` | Observed: ⚪ —<br>**🔴 report missing** | Observed: ⚪ —<br>**🔴 report missing** | Observed: ⚪ —<br>**🔴 report missing** | Observed: ⚪ —<br>**🔴 report missing** | Observed: ⚪ —<br>**🔴 report missing** | 🔴 failed |
| 2026-11-10T122053-42814082346 | `ac11f84616a8` | Observed: ⚪ —<br>**🔴 report missing** | Observed: ⚪ —<br>**🔴 report missing** | Observed: ⚪ —<br>**🔴 report missing** | Observed: ⚪ —<br>**🔴 report missing** | Observed: ⚪ —<br>**🔴 report missing** | 🔴 failed |
| 2026-11-09T122210-42615627016 | `ac11f84616a8` | Observed: ⚪ —<br>**🔴 report missing** | Observed: ⚪ —<br>**🔴 report missing** | Observed: ⚪ —<br>**🔴 report missing** | Observed: ⚪ —<br>**🔴 report missing** | Observed: ⚪ —<br>**🔴 report missing** | 🔴 failed |
| 2026-11-08T121935-42479241147 | `ac11f84616a8` | Observed: ⚪ —<br>**🔴 report missing** | Observed: ⚪ —<br>**🔴 report missing** | Observed: ⚪ —<br>**🔴 report missing** | Observed: ⚪ —<br>**🔴 report missing** | Observed: ⚪ —<br>**🔴 report missing** | 🔴 failed |
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
