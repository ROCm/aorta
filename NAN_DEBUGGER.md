# NaN Debugger

## What It Does

When optimizer raises AssertionError for NaN/Inf:
1. Signals all ranks via TCPStore
2. All ranks export PyTorch profiler traces
3. Failing rank investigates which gradients/parameters contain NaN/Inf
4. Saves diagnostic report

## Output Files

```
{output_dir}/nan_diagnostics/
└── optimizer_failure_step{N}_rank{R}.json

{output_dir}/torch_profiler/
├── rank0/nan_failure_step{N}.json          (failing rank)
├── rank1/nan_coordinated_step{N}.json      (other ranks)
└── ...all ranks export traces
```

## Report Content

```json
{
  "step": 1,
  "rank": 14,
  "optimizer_error": "Encountered gradient containing NaN/Inf...",
  "gradients_with_nan_inf": 1,
  "affected_gradients": [
    {"name": "embedding.weight", "num_nan": 480, "num_inf": 0}
  ],
  "diagnosis": {
    "optimizer": "shampoo",
    "optimizer_eps": 1e-12,
    "embedding_gradients_affected": true
  }
}
```

## Key Features

- Zero overhead (only runs when optimizer fails)
- No false positives (validates actual NaN/Inf counts > 0)
- All ranks save traces for distributed debugging
- View traces in chrome://tracing

## Notes

- Only ranks with actual NaN/Inf (count > 0) create diagnostic files
- Other ranks may get optimizer errors due to distributed operations but won't create files
- All ranks export profiler traces for full distributed debugging
- Old preemptive gradient checking is disabled (caused false positives)

