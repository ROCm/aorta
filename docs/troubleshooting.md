# Troubleshooting

Common issues and solutions when running AORTA benchmarks.

## Common Issues

### ImportError: No module named 'yaml'

**Solution:** Install PyYAML or supply a JSON config instead.

```bash
pip install pyyaml
```

### rocm-smi not found

**Solution:** Install ROCm utilities or omit `--enable-rocm-metrics`.

Ensure ROCm tools are in your `$PATH`:

```bash
export PATH=$PATH:/opt/rocm/bin
```

### CUDA driver errors

**Solution:** Verify `CUDA_DEVICE_MAX_CONNECTIONS=1` (set in launcher) to encourage overlap-friendly scheduling.

This is typically set automatically by the launch scripts, but you can verify:

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
```

### Slow dataloading

**Solution:** Increase `dataloader.num_workers` or reduce dataset volume.

```bash
torchrun --nproc_per_node 4 train.py \
  --config config/default.yaml \
  --override dataloader.num_workers=8
```

### Invalid device ordinal

**Cause:** Launchers' `NPROC` exceeds available GPUs.

**Solution:** The toolkit remaps surplus local ranks modulo the visible devices, but persistent failures usually indicate mismatched visibility.

Check your device visibility:

```bash
# CUDA
echo $CUDA_VISIBLE_DEVICES

# ROCm
echo $HIP_VISIBLE_DEVICES
```

Ensure the launcher's `--nproc_per_node` matches your visible GPU count.

## Extending the Toolkit

- Adjust model depth/width in `config/default.yaml` to stress-test memory and communication pressure
- Swap `MixedPrecision` modes via `training.mixed_precision` (`none`, `fp16`, or `bf16`)
- Leverage the JSONL logs to integrate with external profilers or dashboards (e.g., Prometheus, Weights & Biases)
- Implement custom communication hooks by editing `StreamProfiler.intercept_distributed_ops`

## Getting Help

If you encounter issues not covered here:

1. Check that all prerequisites are installed (see [Getting Started](getting-started.md))
2. Verify your configuration is valid (see [Configuration Guide](configuration.md))
3. Review profiling outputs for error messages (see [Profiling Guide](profiling.md))
4. Open an issue on the GitHub repository with:
   - Your configuration file
   - Error messages and stack traces
   - Hardware/software environment details
