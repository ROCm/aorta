# Install and run Spur

[Spur](https://github.com/ROCm/spur) is a job scheduler with Slurm-style commands (`spur srun`, `spur sbatch`, `spur salloc`, `spur squeue`). Aorta's `tiny_transformer` workload uses it in two ways:

- `launcher: spur` starts the 16 training processes with `spur srun`.
- `launcher: monarch` with `scheduler: spur` starts Monarch's workers with `spur srun`.

This page covers installing the Spur binaries, starting Spur, checking that it works, and stopping it. The full Spur documentation is at <https://instinct.docs.amd.com/projects/spur/en/latest/>.

## What you need

- Linux on x86_64, with glibc 2.28 or newer (Ubuntu 20.04+, Debian 10+, RHEL 8+).
- The machines you will run on. Spur needs one controller (`spurctld`) and one agent (`spurd`) on every machine that runs work.
- If Spur is already your cluster's scheduler, you only need the `spur` command and the controller address. Skip to [Point `spur` at the controller](#point-spur-at-the-controller).

## 1. Install the binaries

Download a release into a directory of its own. A dedicated directory keeps Spur from replacing Slurm's commands on a machine that also has Slurm:

```bash
export SPUR_HOME=$HOME/spur          # any directory you own
curl -fsSL https://raw.githubusercontent.com/ROCm/spur/main/install.sh \
  | INSTALL_DIR="$SPUR_HOME/bin" bash
```

The installer picks the latest release. To pin one, append `-s -- v0.15.0` after `bash`. It downloads `spur-<version>-linux-amd64.tar.gz` from the [releases page](https://github.com/ROCm/spur/releases) and checks its SHA-256 when a checksum is published.

Check what was installed:

```bash
ls "$SPUR_HOME/bin"
"$SPUR_HOME/bin/spur" --version
```

You need `spur`, `spurctld`, `spurd`, and `spurstepd`.

**If this machine also has Slurm,** do not let Spur replace Slurm's `srun` or `sbatch`:

- Do not add `$SPUR_HOME/bin` to `PATH` ahead of Slurm. Call Spur as `$SPUR_HOME/bin/spur srun ...`, and set `spur_bin: <SPUR_HOME>/bin/spur` in the aorta recipe.
- If `$SPUR_HOME/bin` contains `srun`, `sbatch`, `squeue`, `scancel`, `sinfo`, `sacct`, or `scontrol` links, delete them: `rm -f "$SPUR_HOME"/bin/{srun,sbatch,squeue,scancel,sinfo,sacct,scontrol}`.

Install the binaries on a filesystem that every machine can read, or install them on each machine at the same path.

## 2. Start Spur

Choose the case that matches your machines.

### Case A: machines without another scheduler

Spur's own quickstart runs a controller and an agent on one machine:

```bash
mkdir -p /tmp/spur-state
"$SPUR_HOME/bin/spurctld" -D --state-dir /tmp/spur-state            # terminal 1
"$SPUR_HOME/bin/spurd" -D --controller http://localhost:6817         # terminal 2
```

For several machines, follow Spur's [native host deployment guide](https://instinct.docs.amd.com/projects/spur/en/latest/deployment/native-host.html). It covers the configuration file, system services, and networking.

### Case B: inside a Slurm allocation you already have

This runs Spur as your own user, for the length of a Slurm allocation, without root. `spurctld` and each `spurd` run as Slurm job steps, so they stop when the allocation ends.

**1. Get the allocation and choose a controller machine.** Any one of the allocated machines can be the controller.

```bash
salloc -N 2 --exclusive -t 08:00:00
scontrol show hostnames "$SLURM_JOB_NODELIST"       # the two machine names
```

Note each machine's IPv4 address on the network the machines use to reach each other (`ip -br -4 addr`). Use IPv4 addresses below, not hostnames, if hostnames resolve to IPv6 link-local addresses first.

**2. Write the configuration file.** Save this as `$SPUR_HOME/spur.conf` and replace the values in angle brackets:

```toml
cluster_name = "spur-in-slurm"

[controller]
listen_addr = "0.0.0.0:6817"
rest_addr = "127.0.0.1:6820"
raft_listen_addr = "127.0.0.1:6821"
hosts = ["<controller IPv4>"]
state_dir = "<SPUR_HOME>/state/ctld"

[[partitions]]
name = "gpu"
default = true
state = "UP"
nodes = "<machine 1>,<machine 2>"

[network]
wg_enabled = false
agent_port = 16818

[auth]
plugin = "none"
mode = "disabled"

[accounting]
database_url = ""

[rlimits]
memlock = "inherit"

[cgroup]
enabled = false
required = false
```

What these settings do:

- `agent_port = 16818`: Slurm's `slurmd` already listens on 6818 on Slurm machines. The agents need a different port.
- `[cgroup] enabled = false`: cgroup enforcement needs a root `spurd`. Slurm already confines the allocation.
- `[auth] mode = "disabled"`: anyone who can reach ports 6817 or 16818 can submit or cancel jobs. Use this only on machines you trust.
- `memlock = "inherit"`: the agents keep the memory-lock limit Slurm gave the allocation.

**3. Start the controller on the controller machine.** Clear the Slurm step variables first, because Spur sets its own `SLURM_*` variables inside its jobs:

```bash
CLEAN='unset $(compgen -e | grep "^SLURM_\|^SPUR_");'
mkdir -p "$SPUR_HOME/state/ctld" "$SPUR_HOME/logs"

setsid srun --jobid "$SLURM_JOB_ID" --overlap -N1 -n1 -w <controller machine> \
  bash -c "$CLEAN exec $SPUR_HOME/bin/spurctld -D -f $SPUR_HOME/spur.conf \
    --state-dir $SPUR_HOME/state/ctld --listen 0.0.0.0:6817" \
  > "$SPUR_HOME/logs/spurctld.log" 2>&1 < /dev/null &
```

**4. Start one agent on each machine.** Run this once per machine, with that machine's name and IPv4 address:

```bash
NODE=<machine name>; IP=<machine IPv4>; CTRL=<controller IPv4>
mkdir -p "$SPUR_HOME/state/spurd-$NODE"

setsid srun --jobid "$SLURM_JOB_ID" --overlap -N1 -n1 -w "$NODE" \
  bash -c "$CLEAN cd $SPUR_HOME/state && exec $SPUR_HOME/bin/spurd -D -f $SPUR_HOME/spur.conf \
    --controller http://$CTRL:6817 --listen $IP:16818 -N $NODE --address $IP \
    --state-dir spurd-$NODE" \
  > "$SPUR_HOME/logs/spurd-$NODE.log" 2>&1 < /dev/null &
```

`--state-dir` is relative to `$SPUR_HOME/state` on purpose. Each job step binds a Unix socket under that directory, and a socket path longer than 108 bytes fails. A long absolute path can exceed that.

## Point `spur` at the controller

Every `spur` command, and every aorta run that uses Spur, needs the controller address:

```bash
export SPUR_CONTROLLER_ADDR=http://<controller IPv4>:6817
```

## 3. Check that it works

```bash
"$SPUR_HOME/bin/spur" sinfo -N
"$SPUR_HOME/bin/spur" srun -N2 --ntasks-per-node=1 -t 00:02:00 \
  bash -c 'echo "host=$(hostname -s) node=$SPUR_NODEID task=$SPUR_PROCID"'
```

Spur works when `sinfo -N` lists both machines as `idle`, and the `srun` prints two lines with two different hosts.

Then check that Spur gives the job GPUs:

```bash
"$SPUR_HOME/bin/spur" srun -N2 --ntasks-per-node=1 --gpus-per-node=8 --exclusive -t 00:02:00 \
  bash -c 'echo "host=$(hostname -s) gpus=$SPUR_JOB_GPUS"'
```

## Things to know before running aorta on Spur

- **Use `spur salloc` to run a controller inside an allocation, not `spur sbatch`.** `spur salloc` opens a shell on the machine where you run it, with `SPUR_JOB_ID` set. Spur runs a multi-node batch script on every node, not only on the first one, so `spur sbatch` would start one aorta controller per node.
- **Step output arrives when the step ends.** Without `--pty`, `spur srun` prints a step's output after that step finishes.
- **Leave about 30 seconds between GPU jobs.** An agent can still hold the GPUs from the previous job, and the next job then waits for the agent to retry.
- **A dead agent can look `idle` for up to about 2 minutes** before the controller marks it `down`. Setting `heartbeat_timeout_secs` under `[controller]` shortens that window.
- **Old machines can stay in the list.** If you reuse the controller's state directory with a different allocation, `sinfo` can still show machines from the old one. Remove them with `$SPUR_HOME/bin/spur node remove <machine>`, or mark them down with `$SPUR_HOME/bin/spur scontrol update NodeName=<machine> State=DOWN Reason="not in allocation"`.

## 4. Stop Spur

Cancel any Spur jobs, then stop the agents and the controller. For Case B:

```bash
for id in $("$SPUR_HOME/bin/spur" squeue -h -o %i); do "$SPUR_HOME/bin/spur" scancel "$id"; done
squeue -s -j "$SLURM_JOB_ID" -o '%i %j'        # the Slurm steps running spurctld and spurd
scancel <job id>.<step id>                      # once for each of those steps
```

The Slurm allocation is still yours after that. Ending the allocation also stops Spur.
