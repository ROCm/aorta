# Copilot instructions -- ROCm/aorta

## This repository is public

`ROCm/aorta` is a public repository. Everything that reaches `main` -- source,
docs, recipes, test fixtures, sample logs, commit messages -- is world-readable
and permanent in git history. Reverting a leak does not unpublish it.

AORTA's private counterpart, `ROCm/aorta-internal`, owns the customer-escalation
workloads, NDA'd reproducer sources, and the mitigation harness. The two sides
meet only through the public `aorta.workloads` entry-point group, and this
repository's plugin system exists precisely so downstream and private workloads
can register from outside. **Customer-specific material is therefore never
required here for the platform to work.** If a diff seems to need it, that is the
signal that the change belongs on the private side.

Treat every pull request as a disclosure review as well as a code review. When
something in a diff looks like it belongs on the private side, raise it
explicitly, name the category below that it falls under, and say whether the fix
is to relocate it to `aorta-internal` or to redact it in place. Prefer raising a
borderline case over staying silent: a false positive costs a review comment, a
false negative is public forever.

## What must not land in this repository

1. **Customer and partner identity.** Names, codenames, abbreviations, ticket
   IDs, or logos of external customers and partners -- in code, comments,
   docstrings, docs, recipe names, cell names, fixture filenames, or test data.
   This includes indirect identification: a workload named after the customer's
   product, or a model/config combination unique enough to identify them.

2. **Customer workload material.** Reproducer sources, model definitions,
   training or inference scripts, hyperparameters, dataset names or paths, and
   model architecture details that arrived from a customer escalation. These live
   in the owning workload directory on the private side.

3. **Real captured environments and logs.** Verbatim environment-variable dumps,
   `aorta env probe` snapshots, training logs, loss curves, step timings, or
   profiler output taken from a real customer or partner run. Fixtures and
   examples in this repository should be synthetic or drawn from AMD's own
   hardware. Watch for these arriving as large `.json` / `.log` / `.txt`
   additions under `tests/`, `examples/`, or `docs/`.

4. **Internal infrastructure identifiers.** Lab, dev, and CI machine hostnames;
   internal IP addresses and subnets; cluster, partition, or queue names;
   internal dashboard, wiki, or document-store URLs; and absolute paths that
   embed a user's home directory or an internal mount. A setup guide or example
   command that only works on one specific machine is the usual way these arrive
   -- rewrite it against a placeholder.

5. **Non-public container images.** Image tags or digests from AMD-internal or
   customer-gated registries. Only images a member of the public can actually
   pull belong in committed Dockerfiles, recipes, and docs.

6. **Internal-only planning material.** Design documents, roadmaps, escalation
   post-mortems, and status reports written for an internal audience. Public
   planning documents are welcome (see the "expected here" list below); the test
   is the intended audience and the content, not the file's location.

## This file is public too

Do not add real customer names, codenames, hostnames, or internal URLs to this
file or to any other instructions file, whether as a deny-list, an example, or a
"do not use this string" note. Doing so publishes the exact value the rule exists
to protect. Describing the *shape* of a sensitive value is the most any file in
this repository can carry. The authoritative list lives on the private side.

## What is expected here, and should not be flagged

These are all normal in this repository. Flagging them produces noise and trains
reviewers to ignore disclosure comments.

- **AMD product and architecture names** -- `MI300X`, `MI350X`, `MI355X`,
  `MI450X`, `gfx942`, `gfx1250`, ROCm and HIP version numbers. Public.
- **Links to public repositories and docs**, including `ROCm/rocm-systems`,
  `mirage`, `rocjitsu`, and `rocm.docs.amd.com`.
- **Cross-references to `aorta-internal` issue numbers.** Pointing at a private
  issue by number is established practice here (see
  `tests/probe/test_recurring_issue_regression.py`). The pointer is fine; what
  must not follow it across is the customer-identifying content behind it.
- **`docs/plans/`.** A legitimate home for planning documents written for a
  public audience, such as feature rubrics tied to public issues.
- **Public mitigation and environment-variable names** -- `tf32_off`,
  `hsa_xnack`, `HSA_NO_SCRATCH_RECLAIM`, and the rest of the built-in registry.
  These are published debugging knobs, not customer information.
- **Contributor names, emails, and GitHub handles** in git metadata, CODEOWNERS,
  and changelogs.
- **`docker/` package URLs on `artifactory-cdn.amd.com`.** These predate these
  instructions and are already on `main`; do not raise them as new findings.
  Newly added credentials, tokens, or auth headers alongside them are in scope.

## How to raise a finding

Comment on the specific line, name the category number, and state the remedy.
For example: "Category 4 -- this example command hardcodes a lab hostname, so it
only works on one machine and publishes its name. Suggest a placeholder such as
`<your-host>` with the requirement described in prose."

If a diff mixes platform work with customer-specific material, say which files
are safe to merge and which need to move, so the author can split the pull
request rather than rework it wholesale.
