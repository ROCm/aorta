# Docker Setup for Aorta

This directory contains Docker configurations for building and running Aorta training workloads.

## Overview

We provide a unified Docker Compose configuration that supports multiple Dockerfile variants through environment variables. Each user can maintain their own `.env` file (git-ignored) with personalized settings.

## Quick Start

### Method 1: Interactive Setup (Recommended)

Run the setup script to create your `.env` file interactively:

```bash
bash setup-env.sh
docker compose -f docker-compose.build.yaml up -d
```

The script will guide you through:
- Selecting a Dockerfile
- Naming your container
- Configuring volume mounts
- Setting environment variables

### Method 2: Manual Configuration

Copy the example and edit manually:

```bash
cp .env.example .env
# Edit .env with your settings
docker compose -f docker-compose.build.yaml up -d
```

## Available Dockerfiles

| Dockerfile | Description | Use Case |
|------------|-------------|----------|
| `Dockerfile.rocm70_9-1` | Standard ROCm 7.0.9.1 | General development and testing |
| `Dockerfile.rocm70_9-1-shampoo` | ROCm 7.0.9.1 + Shampoo optimizer | Shampoo optimizer experiments |
| `Dockerfile.rocm70_2-ubuntu-pytorch` | ROCm 7.0.2 Ubuntu PyTorch | Legacy ROCm 7.0.2 support |
| `Dockerfile.rocm70_2-ubuntu-nan` | ROCm 7.0.2 + NaN debugging | Debugging NaN issues |

## Configuration Variables

### Required

- **`DOCKERFILE`**: Which Dockerfile to build from
- **`CONTAINER_NAME`**: Unique name for your container (avoid conflicts with other users)

### Volume Mounts

- **`AORTA_WORKSPACE`**: Path to aorta workspace (default: `..`)
- **`RCCL_PATH`**: Path to custom RCCL build (default: `/tmp/rccl_placeholder`)

### Optional

- **`AMDGPU_DRIVER_VARIANT`**: Driver variant for environment_info.json
- **`EXTRA_MOUNT_SRC_*`** / **`EXTRA_MOUNT_DST_*`**: Additional volume mounts

## Example Configurations

### Example 1: Standard Development

```bash
# .env
DOCKERFILE=Dockerfile.rocm70_9-1
CONTAINER_NAME=myuser-dev-20260205
AORTA_WORKSPACE=..
RCCL_PATH=/tmp/rccl_placeholder
```

Run: `docker compose -f docker-compose.build.yaml up -d`

### Example 2: Shampoo with Custom RCCL

```bash
# .env
DOCKERFILE=Dockerfile.rocm70_9-1-shampoo
CONTAINER_NAME=shampoo-experiment-1
AORTA_WORKSPACE=/apps/username/aorta_work/aorta_1
RCCL_PATH=/apps/username/rccl
```

### Example 3: NaN Debugging

```bash
# .env
DOCKERFILE=Dockerfile.rocm70_2-ubuntu-nan
CONTAINER_NAME=debug-nan-issue
AORTA_WORKSPACE=..
RCCL_PATH=/tmp/rccl_placeholder
AMDGPU_DRIVER_VARIANT=patched
```

## File Structure

```
docker/
├── docker-compose.build.yaml     # Unified compose file (use this!)
├── docker-compose.yaml           # Image-based compose (alternative)
├── .env.example                  # Template for your .env
├── .env                          # Your personal config (git-ignored)
├── setup-env.sh                  # Interactive setup script
├── Dockerfile.rocm70_9-1         # Standard ROCm build
├── Dockerfile.rocm70_9-1-shampoo # Shampoo variant
├── Dockerfile.rocm70_2-ubuntu-*  # Legacy ROCm 7.0.2 builds
└── rccl_test/                    # Separate RCCL testing setup
```

## Common Commands

### Start Container

```bash
docker compose -f docker-compose.build.yaml up -d
```

### Stop Container

```bash
docker compose -f docker-compose.build.yaml down
```

### View Logs

```bash
docker compose -f docker-compose.build.yaml logs -f
```

### Connect to Container

```bash
docker exec -it <your-container-name> bash
```

### Rebuild After Dockerfile Changes

```bash
docker compose -f docker-compose.build.yaml build
docker compose -f docker-compose.build.yaml up -d
```

### View Resolved Configuration

See what environment variables are being used:

```bash
docker compose -f docker-compose.build.yaml config
```

## Tips

1. **Unique Container Names**: Use descriptive, unique names to avoid conflicts with other users on shared systems
   - Good: `username-shampoo-2026-02-05`
   - Bad: `training` (too generic)

2. **Git Ignore**: Your `.env` file is git-ignored, so your personal configuration won't be committed

3. **Environment Override**: You can override any variable at runtime:
   ```bash
   CONTAINER_NAME=test-run docker compose -f docker-compose.build.yaml up
   ```

4. **VSCode Integration**: Use VSCode's "Attach to Running Container" feature for an IDE experience

5. **Multiple Variants**: You can run multiple containers with different Dockerfiles simultaneously by using different container names

## Troubleshooting

### "container name already in use"

Another user or previous run is using that name. Choose a different `CONTAINER_NAME`.

### "No such file or directory" for volumes

Check that paths in your `.env` exist and are accessible:
```bash
ls -la $AORTA_WORKSPACE
ls -la $RCCL_PATH
```

### Changes to .env not taking effect

Stop and restart the container:
```bash
docker compose -f docker-compose.build.yaml down
docker compose -f docker-compose.build.yaml up -d
```

### Need to add more volume mounts

Edit your `.env` and add:
```bash
EXTRA_MOUNT_SRC_1=/path/on/host
EXTRA_MOUNT_DST_1=/path/in/container
```

Then update `docker-compose.build.yaml` to reference them in the volumes section.

## Migration from Old Compose Files

If you were using:
- `docker-compose.rocm70_9-1.yaml` → Use `docker-compose.build.yaml` with `DOCKERFILE=Dockerfile.rocm70_9-1`
- `docker-compose.rocm70_9-1-shampoo.yaml` → Use `docker-compose.build.yaml` with `DOCKERFILE=Dockerfile.rocm70_9-1-shampoo`

These old files are deprecated and will be removed in a future update.

## Related Documentation

- [Getting Started Guide](../docs/getting-started.md)
- [Running Benchmarks](../docs/running-benchmark.md)
- [Profiling Guide](../docs/profiling.md)
