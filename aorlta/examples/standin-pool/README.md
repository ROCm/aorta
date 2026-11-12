# A synthetic stand-in pool

A format example for the stand-in prompt source, not a dataset. The three kernels in `kernels.hip` were written for
this example and have no planted bug. `kernels.json` maps each kernel id to its entry, and `listings/<kid>.json` is
the kernel's gfx950 disassembly as `[offset, "instruction"]` pairs, made as the comment at the top of `kernels.hip`
says. [docs/ADAPTING.md](../../docs/ADAPTING.md) describes the format.

`example_transpose` is held out, so the training pool is the other two. The pool is too small for a run, which needs
at least `rollout.P` training kernels and picks its landing probes from 40 kernels of 60 to 400 instructions.
`tests/rollout/test_example_pool.py` loads it through the stand-in prompt source.
