# AORTA Codebase Assistant

Welcome! I'm an AI assistant for the **AORTA** GPU performance benchmarking and debugging toolkit.

## What I Can Help With

- **Navigate the codebase** -- list files, read source code, explore directory structure
- **Search for code** -- find functions, classes, or patterns using semantic search
- **Answer questions** -- explain how components work, trace data flows, describe APIs
- **Run commands** -- *off by default.* When the operator sets `enable_shell_tool = true`, commands from an allowlist (`pytest`, `grep`, `ls`, `python`, ...) run starting in the AORTA repo. This is not a sandbox: an allowed interpreter can reach anything the account running the server can.

## Example Questions

| Question | What It Does |
|----------|-------------|
| *"What does the run() function in core.py do?"* | Retrieves and explains relevant code |
| *"List all Python files in the tests directory"* | Explores the directory structure |
| *"Search for functions that handle configuration parsing"* | Semantic code search |
| *"Run pytest tests/ and show me the results"* | Runs an allowlisted command -- only if the operator enabled command execution |
| *"How is the main training pipeline structured?"* | Multi-step code exploration |

## About AORTA

AORTA is a GPU performance benchmarking and debugging toolkit for PyTorch on AMD ROCm. Key components:

- **FSDP2 Overlap Analysis** -- debug compute-communication overlap in distributed training
- **Hardware Queue Evaluation** -- stress-test GPU queue scheduling with concurrent streams
- **Performance Reports** -- compare ROCm versions across configurations

## Quick Reference

```bash
# FSDP2 overlap benchmark
bash scripts/launch_rocm.sh config/default.yaml

# Hardware queue evaluation
python -m aorta.hw_queue_eval list
python -m aorta.hw_queue_eval run hetero_kernels --streams 8
python -m aorta.hw_queue_eval sweep hetero_kernels --streams 1,2,4,8,16
```

---

*Type your question below to get started.*
