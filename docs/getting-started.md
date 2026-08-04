# Getting Started

Start with the core CLI, then add only the dependencies required by the command
or workload you plan to run.

## 1. Install AORTA

Follow the canonical [installation instructions](../README.md#installation) for
a published or editable install. Verify that the active environment provides
the command:

```bash
aorta --help
```

The core install does not require a GPU, ROCm, Docker, or PyTorch. Those are
runtime requirements for particular workloads and features.

## 2. Choose where to invoke the CLI

`aorta` runs in the environment where it is installed. Relative recipe,
command, and output paths are resolved from your current directory.

- Run `aorta env probe` in the environment you want to describe. That can be a
  host environment or a container.
- A workload that runs in the CLI process needs its runtime dependencies in
  that same environment.
- A Docker-launching workload plugin normally expects the CLI on a host with a
  working Docker CLI and daemon. The plugin owns the container launch; follow
  its guide for host mounts, image access, and dependencies inside the image.

AORTA does not apply one host-or-container rule to every command. The core
dispatcher does not execute `docker run`; Docker-aware workload plugins may do
so and own the launch.

### Repository-relative examples

Commands that name `recipes/...`, `config/...`, or scripts in this repository
assume the repository root as the current directory:

```bash
cd /path/to/aorta
aorta sweep run \
  --recipe recipes/llm-determinism/example-llm-determinism.yaml \
  --dry-run
```

From another directory, pass a path that is valid from that location.

## 3. Add workload requirements

Read the selected workload's guide before installing its runtime:

- Install a matching PyTorch build only for workloads or features that require
  it. AORTA does not bundle PyTorch.
- Install optional AORTA extras such as `hw-queue` only when you use that
  feature; see [Optional dependencies](../README.md#optional-dependencies).
- Provide ROCm tools, GPUs, launchers such as `torchrun`, and Docker only where
  the workload instructions require them.

For the repository's Docker Compose development images for in-tree training
workloads, see [`docker/README.md`](../docker/README.md). That setup is optional,
not the general AORTA installation path.

## 4. Check the installation

The environment probe is a useful first command:

```bash
aorta env probe --summary
```

Unavailable optional components are reported as such. For a recipe, validate
its syntax and workload name before starting GPU work:

```bash
aorta sweep run --recipe /path/to/recipe.yaml --dry-run
```

## Next Steps

- [Recipes](../recipes/README.md) - Understand recipe fields and execution
- [Running the Benchmark](running-benchmark.md) - Launch an in-tree training run
- [Configuration Guide](configuration.md) - Customize trial and workload settings
- [Environment Probe](env-probe.md) - Capture and compare environments
