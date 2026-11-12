# Monarch debugger, in three steps

Three short sessions with the same debugger. Step 1 runs eight CPU workers on
one machine. Step 2 runs one process per GPU on one node. Step 3 adds a second
node. Each program plants one bug right after a `breakpoint()`, and you find
it from a second terminal.

Steps 1 to 3 were run on two MI350X nodes under Slurm. The Spur path has not
been run live.

## How the debugger works

When code inside an actor endpoint calls `breakpoint()`, the worker does not
open pdb on its own terminal. It reports to a debug server that runs with the
controller, the process you started. That server listens on `localhost:27000`,
or on `MONARCH_DEBUG_SERVER_PORT` if you set it. It starts when the first rank
reaches `breakpoint()`, not when the controller starts. If `monarch debug`
reports `Connection refused`, the ranks have not got there yet. Wait for
terminal 1 to print its instructions, give it a few seconds, and run
`monarch debug` again.

`monarch debug` is a thin client: it runs `ncat`, `nc`, or `netcat` against
that port, so one of them must be installed. Because the server only listens on
localhost, both terminals must be on the machine that runs the controller.

At the `monarch_dbg>` prompt, leave with `quit`. `exit` is not a command there;
it prints `Error parsing input` and the prompt comes back.

## Setup

You need Monarch installed, as described in
[Install and run Monarch](../../docs/monarch-install.md), and one of `ncat`,
`nc`, or `netcat`. Steps 2 and 3 also need a Slurm allocation, or a Spur one,
with eight GPUs per node. There is no aorta command for this example; the
scripts drive Monarch directly.

Both terminals start the same way: activate the environment that has Monarch,
then go to the root of your aorta checkout. Every command below runs from
there.

```bash
source "$MONARCH_ROOT/monarch-env.sh"   # ROCm; on CUDA, activate your environment
cd /path/to/aorta
```

The launchers run `$MONARCH_PYTHON` if it is set, and `python` from your
`PATH` otherwise. `MONARCH_PYTHON` only picks the interpreter. On ROCm, still
source `monarch-env.sh` in terminal 1: the GPU workers inherit its
`LD_LIBRARY_PATH`, and without it they can load a different ROCm installed on
the node and fail to import `torch` with a `version 'hip_...' not found` error.
If Monarch cannot be imported, both scripts print an error and exit with
status 2. `--help` works without Monarch.

## Step 1: eight CPU workers on one machine

No Slurm and no GPU. The controller starts eight worker processes on this
host, in an actor mesh named `worker`. Each worker sets `partial = rank + 1`
and stops at `breakpoint()`. The next line, line 73, sets `partial` to 0 on
every rank where `rank % 4 == 2`. That is the bug. With eight workers it hits
ranks 2 and 6, and both are wrong on purpose. The honest total is 36. With the
bug it is 26, which is 36 minus 3 for rank 2 and 7 for rank 6.

Terminal 1:

```bash
python examples/monarch-debug/cpu_mesh.py
```

It prints how to connect, then waits. Terminal 2:

```bash
monarch debug
```

At the `monarch_dbg>` prompt, run `list` until all eight ranks appear. Each one
is stopped in `cpu_mesh.partial_sum` with pdb's arrow on line 73, the bug line.
pdb's arrow marks the line it will run next, not one it has already run. So
until you type `n`, every rank still holds the honest value `rank + 1`, and
`p partial` on rank 2 prints 3. If you print `partial` and detach without `n`,
the bug looks absent.

Attach to rank 2 and print `partial` before and after one `n`:

```text
monarch_dbg> attach worker 2
Attached to debug session for worker 2 (<hostname>)
> .../monarch-debug/cpu_mesh.py(73)partial_sum()
-> partial = 0 if is_bug_rank(rank) else partial
(Pdb) p rank
2
(Pdb) p partial
3
(Pdb) n
> .../monarch-debug/cpu_mesh.py(75)partial_sum()
-> "rank": rank,
(Pdb) p partial
0
(Pdb) detach
Detaching from debug session for worker 2 (<hostname>)
Detached from debug session for worker 2 (<hostname>)
```

One `n` ran line 73, and `partial` went from 3 to 0. pdb now points at line
75, inside the `return` dictionary that follows.

To get back from `(Pdb)` to `monarch_dbg>`, type `detach` on its own. The
debug server intercepts that word before it reaches pdb, so the rank stays
stopped where it is. `c` at `(Pdb)` also gets you back, but it lets that rank
run on, and you return only when its endpoint finishes.

`cast` sends one pdb command to several ranks at once. It prints nothing at
the `monarch_dbg>` prompt. Each rank's reply is held until you next attach to
that rank, and it appears right after the `Attached to debug session` line.
Send `p partial` to both bug ranks, then attach to rank 6 and step it:

```text
monarch_dbg> cast worker ranks(2,6) p partial
monarch_dbg> attach worker 6
Attached to debug session for worker 6 (<hostname>)
7
(Pdb) n
> .../monarch-debug/cpu_mesh.py(75)partial_sum()
-> "rank": rank,
(Pdb) p partial
0
(Pdb) detach
Detaching from debug session for worker 6 (<hostname>)
Detached from debug session for worker 6 (<hostname>)
```

The 7 is the held reply to `cast`. Rank 6 had not run line 73 yet, so it still
held `rank + 1`. After `n` it holds 0, like rank 2. Rank 2's reply to `cast`,
0, waits until you attach to rank 2 again.

For contrast, you can step a healthy rank. Every rank other than 2 and 6 keeps
`rank + 1` after `n`:

```text
monarch_dbg> attach worker 5
Attached to debug session for worker 5 (<hostname>)
> .../monarch-debug/cpu_mesh.py(73)partial_sum()
-> partial = 0 if is_bug_rank(rank) else partial
(Pdb) n
> .../monarch-debug/cpu_mesh.py(75)partial_sum()
-> "rank": rank,
(Pdb) p partial
6
(Pdb) detach
Detaching from debug session for worker 5 (<hostname>)
Detached from debug session for worker 5 (<hostname>)
```

Finally, release every rank:

```text
monarch_dbg> continue
```

`continue` clears all breakpoints and continues every rank that is stopped.
Terminal 1 prints one row per rank and ends with:

```text
total 26  honest total 36  expected with the bug 26
ranks with partial 0: 2, 6
```

Ranks 2 and 6 end at 0 whether or not you stepped them. `n` only runs line 73
early, and `continue` runs it on every rank you did not step. Stepping one rank
here is safe, because the workers never wait for each other.

After `continue`, every rank finishes, the controller in terminal 1 exits, and
its debug server goes away with it. The `monarch debug` client in terminal 2
has nothing left to talk to. Leave it with `quit` or Ctrl-C. For another run,
start `cpu_mesh.py` again in terminal 1, then run `monarch debug` again in
terminal 2. This is different from typing `quit` while the ranks are still
stopped: that closes only the client, the ranks stay where they are, and a new
`monarch debug` picks up the same session.

## What this catches

The debugger stops every rank at `breakpoint()` and lets you read and step
each rank's Python state from one prompt. That finds bugs where one rank, or
one node, holds something different from the rest. Each example below is
buggy code inside an actor endpoint, the commands that expose it, and what you
see. The examples on sharding, shard sizes, seeds, a skipped collective, and
an exception were run on four CPU workers, with a plain line in place of each
collective. The other three are illustrative and were not run.

### A wrong value on one rank

Steps 1 to 3 are runnable versions of this. A realistic form, with 16 ranks
as in step 3:

```python
grad = compute_grad(batch)
breakpoint()
if rank == world - 1:  # bug: meant for the padding rows only
    grad.zero_()
dist.all_reduce(grad)
```

```text
monarch_dbg> attach worker 15
(Pdb) p grad.abs().sum()
(Pdb) n
(Pdb) n
(Pdb) p grad.abs().sum()
(Pdb) detach
```

The first `n` runs the test and stops on `grad.zero_()`, which already shows
that this rank takes the branch. The second runs it, and the sum drops to 0.
Do not step again: the next line is the all-reduce, which waits for every
rank.

### Sharding that drops the tail

```python
data = list(range(10))
n = len(data) // world
shard = data[rank * n:(rank + 1) * n]
breakpoint()
```

```text
monarch_dbg> cast worker ranks(0:4) p (len(data), len(shard), shard)
monarch_dbg> attach worker 3
Attached to debug session for worker 3 (<hostname>)
(10, 2, [6, 7])
(Pdb) detach
```

With four ranks, every shard has two items and the last one ends at 7, so
items 8 and 9 are on no rank. As in step 1, `cast` prints nothing at the
prompt; each rank's reply appears when you next attach to it. Rank 0's reply
is `(10, 2, [0, 1])`.

### Shards of different sizes before a collective

```python
n = -(-len(data) // world)  # ceiling division
x = torch.tensor(data[rank * n:(rank + 1) * n])
breakpoint()
dist.all_gather_into_tensor(out, x)
```

```text
monarch_dbg> cast worker ranks(0:4) p x.shape
monarch_dbg> attach worker 0
Attached to debug session for worker 0 (<hostname>)
torch.Size([3])
(Pdb) detach
monarch_dbg> attach worker 3
Attached to debug session for worker 3 (<hostname>)
torch.Size([1])
(Pdb) detach
```

The last rank holds one item and the others hold three. NCCL does not compare
sizes across ranks, so the gather can hang or fail with an unclear error. At
the breakpoint, the mismatch is one `p` away.

### Ranks that start from different weights

```python
torch.manual_seed(rank)  # bug: every rank should use the same seed
layer = torch.nn.Linear(4, 4)
breakpoint()
```

```text
monarch_dbg> cast worker ranks(0,1) p layer.weight.sum().item()
monarch_dbg> attach worker 0
Attached to debug session for worker 0 (<hostname>)
-1.3448340892791748
(Pdb) detach
monarch_dbg> attach worker 1
Attached to debug session for worker 1 (<hostname>)
0.23327398300170898
(Pdb) detach
```

Data-parallel ranks must start from the same weights, and these do not.
Unless something broadcasts the weights from one rank, nothing fails; the
ranks just train different models.

### A rank that skips a collective

```python
if rank % 2 == 0:  # bug: the odd ranks never join
    breakpoint()
    dist.all_reduce(x)
```

```text
monarch_dbg> list
```

`list` prints one row per stopped rank, with its actor name, rank,
coordinates, hostname, function, and line number. Here it shows ranks 0 and 2
and not 1 or 3. The missing ranks are the ones that took another path. If you
`continue`, ranks 0 and 2 wait in the all-reduce for ranks that never come.

### Two ranks on one GPU

```python
gpu = rank % 4  # bug: written for nodes with four GPUs
torch.cuda.set_device(gpu)
breakpoint()
dist.init_process_group(backend="nccl", ...)
```

```text
monarch_dbg> cast worker ranks(0,4) p torch.cuda.current_device()
monarch_dbg> attach worker 0
(Pdb) detach
monarch_dbg> attach worker 4
(Pdb) detach
```

On a node with eight GPUs, both replies are 0: ranks 0 and 4 share GPU 0, and
GPUs 4 to 7 sit idle. NCCL then fails or hangs when it sets up the group, and
its error does not point at this line. The cruder
`torch.cuda.set_device(rank)` fails by itself on ranks 8 to 15 with an invalid
device ordinal error, so a breakpoint after it is never reached. The next
example shows how to stop on an error like that.

### An exception on one rank

```python
count = counts[rank]
breakpoint()
scale = 1.0 / count  # count is 0 on one rank
```

```text
monarch_dbg> continue
monarch_dbg> list
monarch_dbg> attach worker 1
Attached to debug session for worker 1 (<hostname>)
> .../<file>.py(<line>)<function>()
-> scale = 1.0 / count
(Pdb) p rank, count
(1, 0)
(Pdb) detach
monarch_dbg> continue
```

After the first `continue`, rank 1 raises `ZeroDivisionError`. Monarch opens
pdb on that rank at the line that raised, and `list` shows it there. The last
`continue` ends that session, and the error reaches the controller as an
`ActorError`. This only happens when the same endpoint call already stopped at
`breakpoint()` on that rank: Monarch resets its debug state at the start of
every call, and no setting turns it on otherwise. An endpoint that raises
without passing a breakpoint first just fails, and if no rank has reached
`breakpoint()`, the debug server has not even started.

### Data that one node does not have

```python
files = sorted(glob.glob("/tmp/data/*.bin"))  # bug: /tmp is local to each node
breakpoint()
```

```text
monarch_dbg> cast worker ranks(0,8) p len(files)
monarch_dbg> attach worker 0
(Pdb) detach
monarch_dbg> attach worker 8
(Pdb) detach
```

Ranks 0 and 8 are on different nodes in step 3, and `attach` and `list` both
print each rank's hostname. Different counts mean the nodes see different
files. The same works for anything a node sets up on its own, such as
`p torch.__version__`. The environment is less likely to differ, because
`srun` passes terminal 1's environment to every node.

## Before the GPU steps: get onto an allocated node

Steps 2 and 3 attach to a Slurm allocation that already holds the nodes. The
scripts never call `salloc` or `sbatch`. Run both terminals in a shell on one
of the allocated nodes, the same node for both.

### Use the allocation you already hold

One two-node allocation is enough for both steps, and you do not need a new
one. Find its job id and node list:

```bash
squeue -u $USER
```

Open a shell on one of those nodes. Either `ssh` to it, or ask Slurm for a
shell there:

```bash
srun --jobid <id> --overlap -N1 -w <node> -x <other node> --pty bash
hostname
```

Pass `-x` for every other node in the allocation, and leave it out when the
allocation has one node. On Slurm 25.05, `-w` alone did not pin a step: a
step with `-N1 -w <node>` ran on the other node about half the time. `-x` fixed
that. Check `hostname` anyway.

A shell from `srun --pty` has `SLURM_JOB_ID` set. A shell you opened with
`ssh` does not, so set the job id yourself. Terminal 1, after the setup:

```bash
export MONARCH_SLURM_JOB=<id>
bash examples/monarch-debug/run-slurm.sh 1     # step 2
bash examples/monarch-debug/run-slurm.sh 2     # step 3
```

`MONARCH_SLURM_JOB` takes precedence over `SLURM_JOB_ID`. `run-slurm.sh 1`
inside a two-node allocation uses the node you are on. The script reads the
allocation's nodes with `squeue` and `scontrol show hostnames`, then passes
`-w` with that node and `-x` with the other one to both of its steps. It stops
with a message if you ask for more nodes than the allocation has.

Terminal 2 is a second shell on the same node, opened the same way, with the
setup. It does not need the job id.

### Or hold a new allocation

If you have no allocation, hold one. One `salloc -N2` covers both steps, and
`-N1` is enough for step 2. Your site may need its own partition or account
flags:

```bash
salloc -N2 --gpus-per-node=8
```

A one-shot `srun hostname` prints the hosts and exits. It does not hold the
nodes, and it leaves no `SLURM_JOB_ID` in your shell. The shell that `salloc`
opens has `SLURM_JOB_ID`, but it may still be on the login node; `hostname`
tells you. If it is, open a shell on one of the allocated nodes with the
`srun --pty` command above, and run the steps there.

### Not from the login node

The controller tells the workers its address. It looks that address up from
its own hostname. A login node can have an `/etc/hosts` entry that maps its
own hostname to `127.0.1.1`, a loopback address that the compute nodes cannot
reach. The run then hangs, so the script stops with an error instead. Run
from a shell on an allocated node. If you must run from somewhere else, set
`AORTA_CONTROLLER_IP` (or `MONARCH_CONTROLLER_IP`) to an IPv4 address of that
machine that the nodes can reach, for example the one that `ip -br -4 addr`
shows for the cluster network. Explicit values are used as given.

### Network settings for two nodes

Step 3 connects the two nodes through RCCL or NCCL, and many clusters need
their own network settings for that, such as `NCCL_SOCKET_IFNAME`,
`NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`, and `GLOO_SOCKET_IFNAME`. Without them,
the ranks stall in `init_process_group` after you type `continue`, and
terminal 1 never prints its results.

Put the values from a known-good RCCL or NCCL job on your cluster in a file
of `export` lines, and point `DEBUG_DEMO_NET_ENV` at it:

```bash
export DEBUG_DEMO_NET_ENV=/path/to/cluster-net.sh
```

Both launchers source that file when the variable is set, and the workers
inherit the settings through `srun`. It has no default, so nothing is
sourced unless you set it. You can also export the same variables in
terminal 1 yourself. Step 2 stays on one node and usually does not need them.

### Check that the GPUs are visible

The example needs all eight GPUs visible on every node it uses. An allocation
made without a GPU request may still see all of them, or may not, so check
before you start. If a node shows fewer, the rank that wants a missing GPU
stops with an error that names the host and the count. From a shell with the
Monarch environment active:

```bash
srun --jobid <id> --overlap -N1 -w <node> -x <other node> rocm-smi
srun --jobid <id> --overlap -N1 -w <node> -x <other node> \
  python -c 'import torch; print(torch.cuda.device_count())'
```

On NVIDIA GPUs, use `nvidia-smi` in place of `rocm-smi`. Use
`--gpus-per-node 4` (see step 2) on a node with fewer GPUs.

## Step 2: eight GPUs on one node

Same debugger, now on GPUs. The transcripts below are from the MI350X runs,
with hostnames left out. Each rank puts `rank + 1` in a one-element CUDA
tensor and stops at `breakpoint()`. The next line zeros gpu 3, which is rank
3. Then every rank joins an NCCL process group and all-reduces the tensor. The
honest sum is 36. With the bug it is 32.

Terminal 1, in a shell on an allocated node (see above), after the setup:

```bash
export MONARCH_SLURM_JOB=<id>     # skip if SLURM_JOB_ID is already set
bash examples/monarch-debug/run-slurm.sh 1
```

It prints the controller address, the host it found, the NCCL master, and the
worker log paths, then a line that starts `8 ranks will stop at breakpoint()`.
Open terminal 2 on the same node, do the setup, and run `monarch debug`. Then:

```text
monarch_dbg> list
monarch_dbg> attach worker 3
Attached to debug session for worker 3 (<hostname>)
> .../monarch-debug/gpu_reduce.py(126)reduce()
-> partial = torch.zeros_like(partial) if is_bug_rank(host, gpu, hosts) else partial
(Pdb) p rank
3
(Pdb) p partial
tensor([4.], device='cuda:3')
(Pdb) n
> .../monarch-debug/gpu_reduce.py(127)reduce()
-> before = partial.item()
(Pdb) p partial
tensor([0.], device='cuda:3')
(Pdb) detach
Detaching from debug session for worker 3 (<hostname>)
Detached from debug session for worker 3 (<hostname>)
monarch_dbg> continue
```

`list` shows ranks 0 to 7 stopped in `gpu_reduce.reduce`, with the arrow on
line 126, the bug line. As in step 1, pdb has not run that line yet. Rank 3 is
the only bug rank here. Before `n`, its `partial` is a one-element CUDA tensor
holding `rank + 1`, which is 4. After `n`, it holds 0. After `continue`,
terminal 1 prints one row per rank and ends with `reduced sum 32  honest sum
36` and `ranks with partial 0: 3`.

One rule applies to this step and the next. One `n` over line 126 is safe. It
touches only this rank's tensor, and pdb stops on line 127,
`before = partial.item()`. Do not step further. Line 130,
`dist.init_process_group(...)`, and the `all_reduce` after it wait for every
rank. The other ranks are still stopped at `breakpoint()`, so the rank you
stepped waits forever. Inspect the bad rank, `detach`, then `continue` the
whole mesh.

Extra arguments to `run-slurm.sh` go to `gpu_reduce.py`, for example
`--gpus-per-node 4` on a node with fewer GPUs. The bug is always gpu 3 of the
last host.

## Step 3: sixteen GPUs on two nodes

Same program, two hosts. The mesh has two dimensions, `hosts` and `gpus`, and
the flat rank is `hosts * 8 + gpus`, from 0 to 15. The bug is on gpu 3 of the
last host, which is rank 11. The honest sum is 136. With the bug it is 124.

This needs a two-node allocation, the same one step 2 can use, and possibly
your cluster's network settings (see
[Network settings for two nodes](#network-settings-for-two-nodes)). Terminal
1, in a shell on one of its nodes, after the setup:

```bash
export MONARCH_SLURM_JOB=<id>     # skip if SLURM_JOB_ID is already set
export DEBUG_DEMO_NET_ENV=/path/to/cluster-net.sh   # if your cluster needs it
bash examples/monarch-debug/run-slurm.sh 2
```

The host order follows the allocation, not the node you are on, so the last
host may be either node. `list` shows which one holds rank 11. Terminal 2, on
the same node as terminal 1, after the setup:

```text
monarch debug
monarch_dbg> list
monarch_dbg> attach worker 11
(Pdb) p rank
(Pdb) p partial
(Pdb) n
(Pdb) p partial
(Pdb) detach
monarch_dbg> cast worker ranks(gpus=3) p partial
monarch_dbg> attach worker 3
(Pdb) detach
monarch_dbg> continue
```

`list` now shows both hostnames, and each rank's coordinates, such as
`{'hosts': 1, 'gpus': 3}` for rank 11. `ranks(...)` accepts these names:
`ranks(gpus=3)` picks gpu 3 on every host, which is ranks 3 and 11, and
`ranks(hosts=1,gpus=3)` picks only rank 11.

Rank 11 is the only bug rank. On rank 11, `p partial` before `n` prints
`tensor([12.], device='cuda:3')`, and after `n` it prints
`tensor([0.], device='cuda:3')`. As in step 1, `cast` prints nothing at the
prompt. Attaching to rank 3 shows its held reply right after the
`Attached to debug session` line: `tensor([4.], device='cuda:3')`. That is gpu
3 on the other host, so the zero belongs to rank 11 alone. Rank 11's reply
waits until you attach to rank 11 again.

The rule from step 2 still applies: one `n` over line 126, then `detach`, never
a step into `init_process_group` or `all_reduce`. After `continue`, terminal 1
ends with `reduced sum 124  honest sum 136` and `ranks with partial 0: 11`.

## Steps 2 and 3 on Spur

This path has not been run live. It follows the Spur source, but no one has
run this example under Spur yet. [Install and run Spur](../../docs/spur-install.md)
covers installing and starting Spur.

`spur salloc` opens a shell with `SPUR_JOB_ID` set, on the machine where you
ran it. `SPUR_CONTROLLER_ADDR` must point at the Spur controller; see
[Point `spur` at the controller](../../docs/spur-install.md#point-spur-at-the-controller).
Terminal 1, on one of the Spur nodes; do the setup in the shell that
`spur salloc` opens, then run the steps there:

```bash
spur salloc -N2 --gpus-per-node=8 --exclusive -t 02:00:00
bash examples/monarch-debug/run-spur.sh 1     # step 2
bash examples/monarch-debug/run-spur.sh 2     # step 3
```

If you already hold a Spur allocation, `spur squeue --me` lists it, and
`export SPUR_JOB_ID=<id>` should let `spur srun` run steps in it from another
shell. That comes from the Spur source and has not been tried. With 1 inside a
two-node Spur allocation, Spur uses the first node of the allocation, which
may not be the one you are on. Terminal 2 still goes on the machine where
terminal 1 runs.

Spur cannot cancel one step of a job. Each worker therefore watches for a file
and stops when it appears. The file goes in `DEBUG_DEMO_SHARED_DIR`, or
`AORTA_SHARED_DIR` if that is the one set. `run-spur.sh` defaults it to a
`shared` directory next to the script and creates it; git ignores that
directory. It must be on a filesystem that every node can write, so `/tmp`
does not work.

`run-spur.sh` uses the first of these that is an executable file: `$SPUR_BIN`,
`$SPUR_BIN/spur` (so `SPUR_BIN` may name the directory), and
`$SPUR_HOME/bin/spur`. Otherwise it runs `spur` from your `PATH`. Extra
arguments go to `gpu_reduce.py`, as with `run-slurm.sh`, and
`DEBUG_DEMO_NET_ENV` works the same way.

Three things behave differently from Slurm:

- `spur srun` hands back a step's output only when the step ends, so the
  worker log files stay empty while the run is going.
- The worker port 22222 stays busy for about 22 seconds after a run. Wait
  before the next one, or set another `MONARCH_PORT`.
- On NFS, a node can take a while to see the stop file, so workers may linger
  for a few seconds after the controller exits.

## Troubleshooting

- A script prints that Monarch cannot be imported and exits with status 2:
  the active Python does not have Monarch, or importing it failed; the
  message ends with the import error. Activate the environment from
  [Install and run Monarch](../../docs/monarch-install.md), or set
  `MONARCH_PYTHON` to that environment's `python` before running a launcher.
- Step 2 or 3 fails right after start with an `ActorError` whose traceback
  ends in `import torch` and `version 'hip_...' not found`: the workers loaded
  another ROCm on the node. Source `monarch-env.sh` in terminal 1 before the
  launcher; setting `MONARCH_PYTHON` alone does not set `LD_LIBRARY_PATH`.
- `monarch debug` reports `Connection refused`: the debug server starts only
  when the first rank reaches `breakpoint()`. Wait for terminal 1's
  instructions line and try again. Check also that terminal 2 is on the
  controller's machine.
- Terminal 1 logs `never connected` and the attach fails with
  `MESH_ATTACH_CONFIG_TIMEOUT`: the workers were not where the controller
  looked for them. Two causes seen in testing: a worker step that ran on
  another node, which the `-x` pinning now prevents, and a controller on a
  login node that gave the workers `127.0.1.1`, which the script now refuses.
- `monarch debug` logs `Could not find a suitable netcat binary`: install
  `ncat`, `nc`, or `netcat`. Without one, this Python client does the same job:

  ```bash
  python -c '
  import os, socket, sys, threading
  s = socket.create_connection(("localhost", int(os.environ.get("MONARCH_DEBUG_SERVER_PORT", "27000"))))
  def pump():
      for chunk in iter(lambda: s.recv(4096), b""):
          sys.stdout.write(chunk.decode(errors="replace"))
          sys.stdout.flush()
  threading.Thread(target=pump, daemon=True).start()
  for line in sys.stdin:
      s.sendall(line.encode())
  '
  ```

- `list` prints an empty table, and `ps` shows the workers in state `T`
  (stopped): the terminal stopped them before they reached the debugger. The
  workers share terminal 1 but run in the background. At `breakpoint()`, pdb
  loads `readline`, and in a Python whose `readline` module is built on
  libedit, that changes the terminal settings on stdin. The kernel stops any
  background process that does that (SIGTTOU), and after about 40 seconds the
  controller gives up with `The process likely crashed`. `cpu_mesh.py` avoids
  this by pointing its stdin at `/dev/null` before it starts the workers. For
  your own program that starts workers with `this_host()` from a terminal, do
  the same, or run it with `< /dev/null`. The GPU workers are not affected,
  because `srun` does not give its tasks the terminal.
- Port 27000 is already in use, for example by someone else's debug session:
  export the same `MONARCH_DEBUG_SERVER_PORT` in both terminals before starting.
- `p partial` on a bug rank prints the honest value, such as 3 on rank 2: pdb
  stops before the line its arrow points at, so the bug line has not run yet.
  Type `n` once, then print again.
- `cast` printed nothing: each rank's reply is held until you next attach to
  that rank.
- Only one client can be connected at a time. A new `monarch debug` replaces
  the old one. `quit` at `monarch_dbg>` closes the client and leaves every rank
  stopped, so you can reconnect. After `continue`, the controller exits and
  takes the debug server with it. A command typed after that gets
  `Ncat: Broken pipe`, which is harmless. Exit the old client with `quit` or
  Ctrl-C, restart the program in terminal 1, then run `monarch debug` again.
- Ranks that reach `breakpoint()` after you type `continue` stay stopped. Run
  `list`, and `continue` again.
- The GPU workers write to `/tmp/$USER/debug-demo/gpu-workers.out` and
  `gpu-workers.err` on the controller's machine. Each run overwrites them.
  Monarch's own log path is printed on the first line of terminal 1.
- `MONARCH_PORT` (default 22222) is the worker port, and `DEMO_MASTER_PORT`
  (default 29531) is the NCCL rendezvous port on the first host. Change either
  if a previous run still holds it.
- After `continue` in step 3, terminal 1 never prints its results: the ranks
  are stuck in `init_process_group`, where RCCL or NCCL connects the two
  nodes. This usually means the cluster needs network settings that the run
  did not get. Point `DEBUG_DEMO_NET_ENV` at a file with the values from a
  known-good RCCL or NCCL job on your cluster; see
  [Network settings for two nodes](#network-settings-for-two-nodes).
  `py-spy dump --pid <worker pid>`, run from the Monarch environment, shows
  where a rank is waiting.
- Ctrl-C in terminal 1, or `kill`, stops the worker step on every node within
  a few seconds, even when the ranks are stuck in NCCL. After `kill -9`, the
  step keeps running: find it with `sacct -j <id>` (its name starts with
  `debug-demo-`) and cancel only it with `scancel <id>.<step>`.
- `gpu-workers.err` ends with `CANCELLED ... DUE to SIGNAL Killed` after every
  run. That is the script stopping its worker step on the way out.
- Terminal 1 prints `Task exception was never retrieved` and
  `Unexpected end of input` after its results: the `monarch debug` client
  disconnected before the run ended. It is harmless.

## What it depends on

The GPU steps read these settings:

- The Python environment with Monarch and PyTorch: the active one, or
  `MONARCH_PYTHON`.
- The allocation: `MONARCH_SLURM_JOB` or `SLURM_JOB_ID` for Slurm,
  `SPUR_JOB_ID` for Spur.
- `DEBUG_DEMO_NET_ENV`, an optional file of the cluster's RCCL or NCCL
  network settings. No default.
- `AORTA_CONTROLLER_IP` or `MONARCH_CONTROLLER_IP` (default: this machine's
  IPv4 address, looked up from its hostname).
- `DEBUG_DEMO_SHARED_DIR`, which falls back to `AORTA_SHARED_DIR`. Spur only.
- `MONARCH_PORT` (22222), `DEMO_MASTER_PORT` (29531), and
  `MONARCH_DEBUG_SERVER_PORT` (27000).
- Slurm's `srun`, `squeue`, and `scontrol`, or the `spur` binary, found
  through `SPUR_BIN`, `SPUR_HOME`, or `PATH`.
