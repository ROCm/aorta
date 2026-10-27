# rocjitsu environment: reward v2 (pre-registered)

**Status: pre-registered.** This file and the reward code it describes
(`examples/rl/rocjitsu_env.py`: `POINTS_V2`, `KERNEL_PAID_ONLY_WITH`,
`kernel_paid`, `reprice`, `reward_spec`) were committed together on 2026-10-05,
in the development history this code comes from, before any val or test score
under v2 was computed. The only scores computed before the commit are on the
corpus-v2 **train** split (section 5). The hash below is unchanged since then.

```
reward_spec_sha256("v2") = 74db77d5286fd5707529dbb777b757cc8738f3b197af1875d897c28d20f3bfc5
reward_spec_sha256("v1") = 4503f7f068887471412d7b191fa79b5a5b34cc419211475f955f7856719f8ea1
```

The hash is SHA-256 of `json.dumps(reward_spec(v), sort_keys=True,
separators=(",", ":"))`: every terminal point, the kernel rule, the "near"
distance and both corpus layouts' tool prices. A test pins it.

## 1. The table

| event | v1 | v2 |
|---|---|---|
| `kernel_correct` | +2.0 | **+0.5, paid only with `class_correct`, `location_exact` or `location_near`** |
| `class_correct` | +1.0 | +1.0 |
| `location_exact` | +4.0 | +4.0 |
| `location_near` (within 2 instructions) | +2.0 | +2.0 |
| `kernel_wrong` | −1.0 | −1.0 |
| `clean_correct` | +7.0 | +7.0 |
| `false_alarm` | −2.0 | −2.0 |
| `missed_bug` | −1.0 | −1.0 |
| `no_answer` | −1.0 | −1.0 |
| `malformed` | −1.0 | −1.0 |
| tool prices | layout's | unchanged (same price for the same call) |

Tool prices belong to the corpus layout, not the reward version. v1 layout:
`run_native` 0.5, `run_sim` 0.1 per kernel / 0.6 for `all`, `waitcheck` 0.05,
`consan` 1.0, `disasm` 0.02. v2 layout (`rjenv2.py`): `list_kernels` 0.05,
`run_native` 0.5, `run_sim` 0.1 per kernel, `run_sim all` 0.1 + 0.05 × the
workload's kernel count, `waitcheck` 0.05, `disasm` 0.02, `consan` free.

What one answer earns on a faulty scenario:

| answer | v1 | v2 |
|---|---|---|
| right kernel, right class, exact offset | 7 | 5.5 |
| right kernel, right class, near offset | 5 | 3.5 |
| right kernel, right class, no or far offset | 3 | 1.5 |
| right kernel, wrong class, exact offset | 6 | 4.5 |
| right kernel, wrong class, near offset | 4 | 2.5 |
| **right kernel, wrong class, no or far offset** | **2** | **0** |
| wrong kernel / "none" | −1 | −1 |

## 2. The one rule

> The right kernel is worth +0.5, and it is paid only when the same answer also
> earns `class_correct`, `location_exact` or `location_near`. Naming the right
> kernel and nothing else that checks out earns 0.

The `kernel_correct` event is still recorded when it is unpaid (with 0 points),
so a kernel hit stays countable and `solved` keeps its meaning (kernel, class
and exact offset all fired).

## 3. Why

On the v1 held-out split a constant answer -- one fixed kernel, class and
offset for every scenario -- scored 0.85, about the same as base Qwen3-8B's 0.90,
because the kernel alone earns +2: a guess that lands on the right kernel is
paid half of what an exact offset is worth with no location at all. A reward under
which a constant matches the base model cannot tell learning from guessing.
v2 removes exactly that: the kernel stops paying for itself. Everything a
localisation earns -- the offset credit -- and everything else (the clean
control credit, every penalty, every tool price) is v1's, so v2 differs from v1
in the one place the defect is.

Two consequences, stated rather than discovered later:

- **A perfect answer on a faulty scenario is 5.5, a correct "none" on a clean
  control is still 7.** The clean credit was left unchanged on purpose (it is
  not the defect), so under v2 the oracle's mean depends on the clean fraction
  more than under v1, and the always-"none" policy is relatively stronger on a
  clean-heavy split (corpus-v2 `test_reencode` is 38% clean controls).
- Kernel plus the right class with no location still pays 1.5. That is not
  kernel-only: the class is a second, checkable claim. Section 5 measures what
  guessing both earns.

## 4. Selection

`rocjitsu_env.load_corpus(..., reward=)`, else `$AORTA_RJ_REWARD`, else `v1`,
so every earlier result reproduces unchanged; anything else is refused. The
version is carried by every `Scenario`, the system prompt states the version's
scoring, and `rocjitsu_env.reprice(events, version)` re-prices a recorded
episode's `(name, points)` events (terminal events take the version's points,
tool costs are kept), which is how one rollout is scored under both.

## 5. Pre-registered check, train split only

Criterion: under v2, every guessing policy below scores **at most about 0.5**
mean on corpus-v2 `train` (869 scenarios: 761 faulty, 108 clean). Each answer is
played through the corpus's reference environment (`rjenv2.Episode`) and its
events are priced by `rocjitsu_env.reprice`. The check script is kept with the
corpus build, not in this repository.

Guessing policies (the criterion):

- `always_none` -- the constant "none";
- `const_full` -- the best single fixed answer (class, kernel, offset) or "none",
  chosen in-sample on train;
- `const_kernel_class` -- the best fixed (class, kernel) with no offset;
- `workload_kernel_class` -- per workload (its description opens every prompt),
  the best fixed (class, kernel) or "none", fit in-sample on train;
- `largest_kernel_waitcnt`, `busiest_kernel_waitcnt` -- the workload's largest /
  most-dispatched non-runtime kernel, class `waitcnt` (train's majority), no
  offset: kernel guesses that need no fitting.

Informational, not the criterion: `workload_full` (per-workload best full answer,
which memorises offsets -- offset credit is unchanged by design) and
`kernel_oracle_*` (told the faulty kernel and whether there is a bug, no offset:
the ceiling of kernel identification without localisation).

Result (train, n = 869):

| policy | v1 | v2 |
|---|---|---|
| `always_none` (= best `const_full` and best `const_kernel_class`, under either version) | −0.006 | −0.006 |
| `workload_kernel_class`, fit under v1 | +0.743 | +0.192 |
| `workload_kernel_class`, fit under v2 | +0.726 | **+0.247** |
| `largest_kernel_waitcnt` | +0.174 | −0.410 |
| `busiest_kernel_waitcnt` | +0.166 | −0.406 |
| *`workload_full`, fit under v2 (informational)* | +0.975 | +0.503 |
| *`kernel_oracle_true_class` (informational)* | +3.497 | +2.184 |
| *`kernel_oracle_workload_class` (informational)* | +3.375 | +2.001 |
| *`kernel_oracle_wrong_class` (informational)* | +2.621 | +0.870 |

**The criterion holds: the best guessing policy under v2 is 0.247 (bound 0.5), so
nothing was adjusted.** Under v1 the same per-workload kernel prior earns 0.743
in-sample, which is the shortcut v2 exists to remove. With 40 workloads the
single best constant is "none" under both versions -- a fixed kernel is wrong
on almost every workload -- so on this corpus the shortcut is the per-workload
prior, not one constant. `kernel_oracle_wrong_class` is the rule seen directly:
told the kernel and nothing else, it earns 0 on every faulty scenario under v2
(its 0.870 is the clean controls' +7 × 108/869). The kernel oracles are not
guessing policies: they are given the kernel and whether there is a bug.

## 6. What follows this commit

CPU baselines on `val`, `test` and `test_reencode` under both versions, then
base Qwen3-8B on `test_reencode`, scored under v2 and re-scored under v1 from
the same replies.
