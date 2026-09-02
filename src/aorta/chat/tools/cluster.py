"""Tools that let the assistant drive the Cluster Intelligence Agents.

These wrap the Launch -> Watch -> Autopsy pipeline so a user can go from "why did
my kernel race?" to a classified failure bundle without touching a terminal. The
agents live in their own virtualenv, so everything here shells out to that venv's
console scripts rather than importing them into the chatbot process.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

from aorta.chat.config import settings
from aorta.cia.triage import _default_aorta_root, run_triage, write_asm_recipe
from aorta.chat.tools.harness.assembly import AsmHarnessError, prepare_asm
from aorta.chat.tools.harness.kernel import WAVEFRONT, HarnessError, prepare_source

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ARCH = os.environ.get("CIA_GPU_ARCH", "gfx950")
# The assembler lives with ROCm on the compute nodes, not on the login node.
_ROCM_LLVM = os.environ.get("ROCM_LLVM_BIN", "/opt/rocm-7.0.2.2/lib/llvm/bin")

# Recipes the demo knows how to run, so the model picks from a validated set
# instead of inventing a path that does not exist on disk.
KNOWN_RECIPES: dict[str, str] = {
    "racy": "recipes/sanitizers/daily-consan-racy.yaml",
    "fixed": "recipes/sanitizers/demo-consan-fixed.yaml",
}




# ConSan encodes the access type numerically in its conflict records. The mapping
# is confirmed against its own coverage_site lines, which label the same
# instruction offsets as read / write / atomic.
_ACCESS_KIND = {"1": "read", "2": "write", "4": "atomic"}


def _access_kind(raw: object) -> str:
    """Name an access kind, falling back to the raw code rather than dropping it."""
    key = str(raw)
    return _ACCESS_KIND.get(key, f"access(kind={key})")


def _explain_reason(reason: str) -> list[str]:
    """Name a known sanitizer failure mode so the model does not invent one.

    Without this the assistant guesses plausible-sounding causes ("build error")
    for an exit code whose real cause is documented here.
    """
    if not reason.startswith("combined_hook_exit"):
        return []
    return [
        "This is NOT a build failure: the binary compiles and exits 0 when run "
        "without the sanitizer.",
        "Known ConSan limitation on gfx950: instrumenting a kernel that contains "
        "an s_barrier (i.e. __syncthreads()) faults with 'an illegal memory "
        "access was encountered' at hipDeviceSynchronize, so the kernel dies "
        "part-way through and the trace is truncated.",
        "Consequence: a 'conflict=false' / zero-finding result on a barrier-"
        "containing kernel is an artifact of the truncated trace, NOT evidence "
        "that the race is fixed. Do not report the fix as verified.",
    ]


def _fmt_sanitizer(san: dict) -> list[str]:
    """Render the sanitizer verdict, which is the ground truth for a race."""
    lines = [
        f"Sanitizer verdict: {san.get('overall_verdict')} "
        f"({san.get('execution_status')}) on {san.get('target')}",
    ]
    if san.get("kernel"):
        lines.append(f"Kernel under test: {san['kernel']}")
    lines.append(f"Total findings: {san.get('total_findings', 0)}")

    for check in san.get("checks") or []:
        lines.append(
            f"  - {check.get('sanitizer')}: {check.get('verdict')} "
            f"(state={check.get('state')}, findings={check.get('findings', 0)})"
        )
        reason = check.get("reason")
        if reason:
            lines.append(
                f"      reason: {reason} (exit {check.get('returncode')}) "
                f"— the sanitizer did not complete, so this run proves nothing "
                f"about whether the kernel races"
            )
            lines += [f"      {line}" for line in _explain_reason(reason)]
        example = check.get("example") or {}
        if example:
            first_kind = _access_kind(example.get("first_kind"))
            second_kind = _access_kind(example.get("second_kind"))
            lines.append(
                f"      conflict: wave {example.get('first_owner')} {first_kind} "
                f"LDS {example.get('first_lds')} (inst {example.get('first_inst')}) vs "
                f"wave {example.get('second_owner')} {second_kind} "
                f"LDS {example.get('second_lds')} (inst {example.get('second_inst')})"
            )
    return lines


def _fmt_tools_used(result: dict) -> list[str]:
    """Name the agents and sanitizers that actually ran.

    The value of the answer depends on it being attributable, so this reports
    what each component did rather than letting the model narrate a pipeline it
    cannot see.
    """
    san = result.get("sanitizer") or {}
    checks = san.get("checks") or []
    lines = ["Tools used on this run:"]
    if result.get("compiled_from_source"):
        lines.append(
            f"  - hipcc — compiled the submitted kernel for {_ARCH} on the GPU node"
        )
    lines.append(
        f"  - CIA Launch agent — submitted slurm job {result.get('slurm_job_id', '?')} "
        f"to an MI355X node"
    )
    lines.append(
        f"  - CIA Watch agent — monitored the job log "
        f"({'raised an alert' if result.get('watch_alerted') else 'no alert raised'})"
    )
    if checks:
        lines.append("  - AORTA sanitizer sweep — ran the sanitizers below:")
    for check in checks:
        lines.append(
            f"      · {check.get('sanitizer')} — {check.get('verdict')} "
            f"({check.get('findings', 0)} finding(s))"
        )
    if result.get("autopsy"):
        lines.append(
            f"  - CIA Autopsy agent — classified the bundle as "
            f"{result['autopsy'].get('category')}"
        )
    return lines


def _format_result(result: dict, label: str) -> str:
    lines = [
        f"Job {result['job_id']} (slurm {result.get('slurm_job_id', '?')}) — {label}",
        f"Recipe:   {result.get('recipe')}",
        f"State:    {result.get('slurm_state')}",
        f"Bundle:   {result.get('bundle')}",
    ]
    if result.get("kernel"):
        lines.append(f"Kernel:   {result['kernel']}")
    lines.append(f"Watch:    {'alerted' if result.get('watch_alerted') else 'no alert raised'}")

    lines.append("")
    lines += _fmt_tools_used(result)

    san = result.get("sanitizer")
    if isinstance(san, dict):
        lines.append("")
        lines += _fmt_sanitizer(san)
        lines.append(f"Sanitizer report: {result.get('sanitizer_report_path')}")

    autopsy = result.get("autopsy")
    if autopsy:
        lines.append("")
        lines.append("Autopsy verdict:")
        lines.append(f"  category:   {autopsy.get('category')}")
        lines.append(f"  confidence: {autopsy.get('confidence')}")
        rationale = autopsy.get("rationale")
        if rationale:
            lines.append(f"  rationale:  {rationale}")
        for probe in (autopsy.get("next_probes") or [])[:4]:
            lines.append(f"  next probe: {probe}")
        for gap in (autopsy.get("tooling_gaps") or [])[:4]:
            lines.append(f"  gap:        {gap}")
        lines.append(f"  report:     {result.get('report_path')}")
    else:
        lines.append("")
        lines.append(f"No Autopsy report produced. {result.get('warning', '')}")

    return "\n".join(lines)


def _run_triage(extra_args: list[str], label: str) -> str:
    """Run Launch -> Watch -> Autopsy in-process and render the result.

    The agents live in this environment now, so this is a call rather than a
    subprocess. The timeout still stands: the work happens on a worker thread
    the caller abandons on expiry, so a wedged cluster job cannot hang the chat.
    """
    argv = ["--jobs-root", str(settings.jobs_root), "--label", label, *extra_args]
    if settings.cia_demo_node:
        argv += ["--node", settings.cia_demo_node]
    # Exported in the batch job rather than in this process: LD_PRELOAD would
    # otherwise apply to everything the chatbot spawns, and these are only
    # meaningful to the sanitizer.
    for key, value in (
        ("ROCJITSU_BUILD", settings.rocjitsu_build),
        ("LD_PRELOAD", settings.rocjitsu_preload),
        # The root the AORTA CLI resolves recipes against, which is not
        # settings.aorta_root -- that is the codebase the chat tools may read,
        # and it points at the package while recipes sit beside src/.
        ("AORTA_PATH", _default_aorta_root()),
        ("CIA_JOBS_ROOT", str(settings.jobs_root)),
    ):
        if value:
            argv += ["--env", f"{key}={value}"]

    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(run_triage, argv)
    pool.shutdown(wait=False)
    try:
        result = future.result(timeout=settings.triage_timeout)
    except FuturesTimeout:
        return (f"Error: triage exceeded {settings.triage_timeout}s. "
                f"Check {settings.jobs_root} for a partial bundle.")
    except Exception as exc:
        return f"Error: triage failed: {type(exc).__name__}: {exc}"

    if not result.get("ok"):
        if result.get("stage") == "compile":
            diags = "\n".join(f"  {d}" for d in result.get("compile_diagnostics") or [])
            return (
                "The submitted kernel did not compile, so nothing was analysed.\n"
                f"Compiler diagnostics:\n{diags or '  (none captured)'}\n"
                f"Job directory: {result.get('job_dir')}"
            )
        return (f"Triage failed at stage {result.get('stage', '?')}: "
                f"{result.get('error', 'unknown error')}")

    return _format_result(result, label)


_PYTORCH_MARKERS = ("import torch", "nn.Module", "def forward", "torch.nn", "@torch")


def _wrong_tool_hint(source: str) -> str:
    """Point a misrouted paste at the tool that can actually handle it.

    A user showing the model that produced a NaN will paste Python, which ConSan
    cannot compile. Without a redirect the turn dead-ends on an error message and
    the agent asks for a kernel it was never going to get.
    """
    if not any(marker in source for marker in _PYTORCH_MARKERS):
        return ""
    return (
        " This is a PyTorch model, not a HIP kernel, and ConSan needs a __global__ "
        "function to compile. If you are chasing a NaN or a non-finite loss, call "
        "run_nan_demo instead: it runs this model and its kernel on an MI355X, traps "
        "the first non-finite value with a device-side assert, and returns the root "
        "cause. Do not ask the user for kernel source -- call run_nan_demo now."
    )


# Triaging one kernel costs a multi-minute cluster job, and the same kernel gets
# submitted more than once per conversation: the model re-checks its own answer and
# the critic can re-run a turn. Reusing the verdict for an identical kernel keeps a
# demo to one job per kernel instead of one per attempt.
_TRIAGE_CACHE: dict[tuple[str, int, int], str] = {}


@tool
def triage_kernel_source(
    source: str,
    label: str = "",
    block_size: int = 0,
    grid_size: int = 0,
    force: bool = False,
) -> str:
    """Compile a HIP kernel the user supplied and run it under a sanitizer.

    Builds the kernel for gfx950 on a GPU node, generates a launch harness if
    the source is a bare __global__ function rather than a whole program, and
    runs it twice under record/replay. Reports accesses that two waves made to
    the same shared-memory bytes with nothing ordering them, naming each wave,
    the byte range and the instruction offsets, and classifies the result.
    Takes several minutes.

    Args:
        source: The HIP kernel source to analyse, exactly as the user pasted it.
        label: Short human label for the run, e.g. "user reduction kernel".
        block_size: Threads per block. Leave 0 to infer from the kernel's
            __shared__ array size.
        grid_size: Blocks to launch. Leave 0 for a single block.
        force: Re-run on hardware even if this exact kernel was already triaged
            in this conversation. Leave false; the cached verdict is the same run.

    Returns:
        The sanitizer findings, the tools that ran, and the Autopsy verdict.
    """
    try:
        prepared = prepare_source(source, block=block_size, grid=grid_size)
    except HarnessError as exc:
        return f"Cannot analyse this source: {exc}{_wrong_tool_hint(source)}"

    cache_key = (source.strip(), block_size, grid_size)
    if not force and cache_key in _TRIAGE_CACHE:
        return (
            "(Reusing the triage already run for this exact kernel in this "
            "conversation — no second cluster job was submitted.)\n\n"
            + _TRIAGE_CACHE[cache_key]
        )

    staging = settings.jobs_root / "chat-kernels"
    staging.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    src_path = staging / f"{prepared.kernel}-{stamp}.hip"
    src_path.write_text(prepared.program, encoding="utf-8")

    lines = [f"Analysing kernel '{prepared.kernel}' from {src_path}."]
    if prepared.wrapped:
        lines.append(
            f"No main() was pasted, so a launch harness was generated: "
            f"{prepared.kernel}<<<{prepared.grid}, {prepared.block}>>>."
        )
    lines.append("")

    body = _run_triage(
        ["--source", str(src_path), "--kernel-name", prepared.kernel],
        label or f"user kernel {prepared.kernel}",
    )

    if prepared.single_wave:
        lines.append(
            f"WARNING — geometry caveat: the harness launched {prepared.block} "
            f"threads, which is a single {WAVEFRONT}-lane wavefront on gfx950. "
            f"ConSan only reports conflicts BETWEEN waves, so this run cannot "
            f"show a cross-wave race and a 'pass' here does NOT mean the kernel "
            f"is race-free. Re-run with block_size={prepared.block * 2} to put "
            f"the kernel on at least two waves.\n"
        )

    rendered = "\n".join(lines) + body
    # Only a completed run is worth reusing; a transient launch failure should be
    # retried rather than remembered.
    if "Autopsy verdict:" in rendered:
        _TRIAGE_CACHE[cache_key] = rendered
    return rendered


# The NaN demo is a fixed workload rather than user input, so a conversation that
# asks about it twice should not pay for two cluster jobs.
_NAN_CACHE: dict[bool, str] = {}

_NAN_VALUE_RE = re.compile(r"nan-trap value\s+(\w+)=(\S+)")
_NAN_ASSERT_RE = re.compile(
    r'assertion=.*?"([^"]+)".*?file=.*?"([^"]+)".*?line=(\d+)'
)
_NAN_WAVE_RE = re.compile(r"AMDGPU Wave\s+[\d:]+\s+\((\d+),(\d+),(\d+)\)")
_NAN_KERNEL_RE = re.compile(r"#1\s+.*?\bin\s+([A-Za-z_]\w*)\s*\(")









WAITCHECK_RECIPE = "recipes/sanitizers/demo-waitcheck-asm.yaml"

# Assembling a paste costs a compute node for about a second, and the same paste
# arrives more than once per conversation when the model re-checks itself.
_ASM_CACHE: dict[str, str] = {}


@tool
def triage_assembly_source(source: str, label: str = "") -> str:
    """Assemble gfx950 assembly the user supplied and analyse it for wait hazards.

    Wraps the pasted instructions in a minimal kernel if they are a fragment,
    assembles them for gfx950, and runs the static wait checker over the
    resulting code object. Reports instructions whose result is consumed before
    any wait guarantees it has landed, naming the producing and consuming
    instructions, their byte offsets and the register involved, plus waits
    stronger than the dependency requires.

    Args:
        source: The assembly text, exactly as the user pasted it.
        label: Human label for the run.

    Returns:
        The hazards found in the supplied assembly, or a clean result.
    """
    try:
        prepared = prepare_asm(source)
    except AsmHarnessError as exc:
        return f"Cannot analyse this assembly: {exc}"

    cache_key = source.strip()
    if cache_key in _ASM_CACHE:
        return (
            "(Reusing the analysis already run for this exact assembly in this "
            "conversation.)\n\n" + _ASM_CACHE[cache_key]
        )

    staging = settings.jobs_root / "chat-asm"
    staging.mkdir(parents=True, exist_ok=True)
    stem = f"{prepared.kernel}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    asm_path = staging / f"{stem}.s"
    obj_path = staging / f"{stem}.hsaco"
    asm_path.write_text(prepared.program, encoding="utf-8")

    clang = f"{_ROCM_LLVM}/clang"
    build = (
        f"{shlex.quote(clang)} -target amdgcn-amd-amdhsa -mcpu={shlex.quote(_ARCH)} "
        f"{shlex.quote(str(asm_path))} -o {shlex.quote(str(obj_path))}"
    )
    proc = subprocess.run(
        ["srun", "--nodes=1", "-t", "5", "bash", "-c", build],
        capture_output=True, text=True, timeout=settings.waitcheck_timeout,
        stdin=subprocess.DEVNULL,
    )
    if not obj_path.is_file():
        diagnostics = "\n".join(
            line for line in (proc.stderr or "").splitlines()
            if "unused during compilation" not in line
        )
        return (
            "The pasted assembly did not assemble, so nothing was analysed.\n"
            f"Assembler output:\n{diagnostics[-1200:] or '(none captured)'}\n"
            f"Wrapped source kept at: {asm_path}"
        )

    recipe = write_asm_recipe(
        recipe_path=staging / f"{stem}.yaml",
        kernel_name=prepared.kernel,
        code_object=obj_path,
        target=_ARCH,
        ticket=f"CHAT-{prepared.kernel}",
    )
    body = _run_triage(["--recipe", str(recipe)], label or f"{prepared.kernel} asm")

    note = []
    if prepared.wrapped:
        note.append(
            f"  (wrapped as a kernel for assembly: {prepared.sgpr_count} SGPRs, "
            f"{prepared.vgpr_count} VGPRs, sized from the registers your code names)"
        )
    note.append(
        "  This is static analysis of the code object: nothing executed, so a "
        "clean result means the instruction stream is well-ordered, not that "
        "the kernel was tested."
    )
    result = "\n".join([body, "", *note])
    _ASM_CACHE[cache_key] = result
    return result

@tool
def triage_workload(source: str = "", command: str = "", label: str = "") -> str:
    """Run a workload the user supplied and diagnose why it failed.

    Takes either training code to run or a command to launch. Submits it to a
    GPU node, reads the log it produces while it runs, and classifies whatever
    it leaves behind -- the step a value stopped being finite, the error a
    framework raised, the artifacts an aborted process left -- naming the
    category, the confidence and the evidence each conclusion rests on.

    Args:
        source: Python the user pasted, written to a file and run.
        command: A command line to run instead, when the workload is not a
            single file.
        label: Short name for the run, used in the job record.

    Returns:
        The Watch signal and the Autopsy verdict, with the evidence cited.
    """
    if bool(source.strip()) == bool(command.strip()):
        return (
            "Error: pass either source (code to run) or command (a command "
            "line), not both and not neither."
        )

    name = label or "workload"
    if command.strip():
        return _run_triage(["--command", command.strip()], name)

    staged = settings.jobs_root / "staged"
    staged.mkdir(parents=True, exist_ok=True)
    script = staged / f"{name.replace(' ', '_')}.py"
    script.write_text(source, encoding="utf-8")
    # {bundle} is substituted by the driver, so a workload that writes
    # artifacts puts them where Autopsy will look.
    return _run_triage(
        ["--command", f"{shlex.quote(sys.executable)} {shlex.quote(str(script))} {{bundle}}"],
        name,
    )


@tool
def list_cluster_jobs(limit: int = 10) -> str:
    """List recent Cluster Intelligence jobs and whether they have a verdict.

    Use this to find a previous GPU failure to explain, before running anything new.

    Args:
        limit: How many of the most recent jobs to show.

    Returns:
        One line per job with its recipe, status and Autopsy category if present.
    """
    root = settings.jobs_root
    if not root.is_dir():
        return f"Error: jobs root {root} does not exist."

    dirs = sorted(
        (d for d in root.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )[: max(1, min(limit, 50))]

    if not dirs:
        return f"No jobs found under {root}."

    lines = [f"Recent jobs under {root}:"]
    for job_dir in dirs:
        recipe, status = "?", "?"
        job_json = job_dir / "job.json"
        if job_json.is_file():
            try:
                data = json.loads(job_json.read_text())
                recipe = data.get("recipe", "?")
                status = data.get("status", "?")
            except Exception:
                pass

        verdict = "no report"
        report = job_dir / "bundle" / "report.json"
        if report.is_file():
            try:
                data = json.loads(report.read_text())
                verdict = f"{data.get('category', '?')} @ {data.get('confidence', '?')}"
            except Exception:
                verdict = "unreadable report"

        lines.append(f"  {job_dir.name}  recipe={recipe}  status={status}  verdict={verdict}")
        lines.append(f"      bundle: {job_dir / 'bundle'}")
    return "\n".join(lines)


@tool
def read_autopsy_report(job_id: str) -> str:
    """Read the full Autopsy report for a cluster job.

    Args:
        job_id: The CIA job id (e.g. cia-20260819-232554-6ec890), or an absolute
            path to a job directory.

    Returns:
        The report JSON, including category, confidence, rationale and evidence.
    """
    job_dir = Path(job_id) if Path(job_id).is_absolute() else settings.jobs_root / job_id
    report = job_dir / "bundle" / "report.json"
    if not report.is_file():
        return (f"Error: no report at {report}. "
                f"Use list_cluster_jobs to see jobs that have a verdict.")
    text = report.read_text(encoding="utf-8", errors="replace")
    if len(text) > 8000:
        text = text[:8000] + f"\n... (truncated, {len(text)} total chars)"
    return f"Autopsy report {report}:\n{text}"
