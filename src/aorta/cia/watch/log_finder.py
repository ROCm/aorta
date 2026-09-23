from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dspy

from aorta.cia.launch.cluster import ssh_user
from aorta.cia.autopsy.adapters.base import resolve_in_bundle
from aorta.cia.llm import ensure_configured

if TYPE_CHECKING:  # pragma: no cover - the annotation must not cost an import
    from aorta.laya.predictor import LayaPredictor, Noul

# Extensions considered log files by default
DEFAULT_LOG_EXTENSIONS = {".log", ".txt", ".out", ".err"}
DEFAULT_EXCLUDE_PATTERNS = {".ckpt", ".bin", ".pt", ".safetensors", ".pkl", ".npz", ".npy"}
DEFAULT_MAX_FILES = 8
# Directories that never hold training logs but do hold thousands of .txt files.
SKIP_DIR_PARTS = {".git", ".venv", "site-packages", "node_modules", "__pycache__"}

#: What the Laya tier resolves when the config names no checkpoint.
DEFAULT_LAYA_BACKEND = "laya-typed-decisions"

#: How sure the classifier has to be that a listed file is a log before Watch
#: watches it.
#:
#: A placeholder and not a fitted number. The temperature refit that makes a
#: Laya probability mean what it says is per (question type, option count) and
#: has not been run against anything in this repository, so 0.5 is here as the
#: indifference point of a probability rather than as a measurement. That is
#: also why ``laya.enabled`` ships false: the tier is wired, not justified.
DEFAULT_LAYA_MIN_PROBABILITY = 0.5


# ---------------------------------------------------------------------------
# Scheduler-native log path discovery (highest priority, most reliable)
# ---------------------------------------------------------------------------

def query_scheduler_logs(scheduler: str, scheduler_job_id: str, head_node: str = "") -> list[str]:
    """Ask the scheduler for the exact stdout/stderr paths of a running job.

    Returns a list of absolute file paths. Empty list if not available.
    This is the most reliable source — no guessing needed.

    Slurm:  scontrol show job <id>  →  StdOut=, StdErr= fields
    K8s:    kubectl get pod <name> -o json  →  volumeMounts + log paths
    """
    paths: list[str] = []
    if not scheduler_job_id:
        return paths

    def _run(cmd: str) -> str:
        try:
            if head_node:
                user = ssh_user()
                r = subprocess.run(
                    ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                     f"{user}@{head_node}", cmd],
                    capture_output=True, text=True, timeout=15,
                )
            else:
                r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
            return r.stdout + r.stderr
        except Exception:
            return ""

    if scheduler == "slurm":
        out = _run(f"scontrol show job {scheduler_job_id}")
        # Parse StdOut=... StdErr=... (may be on same line or separate)
        for token in out.split():
            for key in ("StdOut=", "StdErr="):
                if token.startswith(key):
                    p = token[len(key):].strip()
                    if p and p != "/dev/null" and p not in paths:
                        paths.append(p)
        # Also grab WorkDir and Command for context
        for token in out.split():
            if token.startswith("WorkDir="):
                wd = token[len("WorkDir="):].strip()
                if wd:
                    # Look for log files in the workdir too
                    paths.append(f"__workdir__:{wd}")

    elif scheduler == "kubernetes":
        # Get pod spec to find volume mounts and log paths
        out = _run(f"kubectl get pod {scheduler_job_id} -o json")
        try:
            pod = json.loads(out)
            containers = pod.get("spec", {}).get("containers", [])
            for c in containers:
                for vm in c.get("volumeMounts", []):
                    mount = vm.get("mountPath", "")
                    # Common training log mount paths
                    if any(k in mount.lower() for k in ("log", "output", "scratch", "work")):
                        paths.append(f"__dir__:{mount}")
            # Also try kubectl logs path pattern
            paths.append(f"__kubectl_logs__:{scheduler_job_id}")
        except Exception:
            pass

    return paths


def _dir_listing(job_dir: Path) -> str:
    """Return a find-style listing of job_dir with sizes and mtimes."""
    try:
        r = subprocess.run(
            ["find", str(job_dir), "-maxdepth", "4", "-type", "f",
             "-printf", "%T@ %s %p\n"],
            capture_output=True, text=True, timeout=10,
        )
        lines = []
        for line in r.stdout.strip().splitlines():
            parts = line.split(" ", 2)
            if len(parts) == 3:
                mtime, size, path = parts
                lines.append(f"{path}  size={size}  mtime={mtime}")
        return "\n".join(lines[:60])
    except Exception as e:
        return f"listing error: {e}"


# ---------------------------------------------------------------------------
# DSPy signature for auto-discovery
# ---------------------------------------------------------------------------

class LogDiscovery(dspy.Signature):
    """
    You are discovering which files in a job directory contain training logs
    worth monitoring for a GPU cluster job. Given a file listing, identify the
    files most likely to contain training progress output such as loss values,
    throughput, step counts, GPU errors, or stack traces.

    Rules:
    - Prefer recently modified files (high mtime).
    - Prefer files with 'log', 'stderr', 'stdout', 'out', 'err' in their name.
    - Skip checkpoints (.ckpt, .bin, .pt, .safetensors), weights, and binaries.
    - Skip files smaller than 100 bytes — likely empty placeholders.
    - Return at most 8 files, ranked by relevance (most relevant first).
    """
    job_dir_listing: str = dspy.InputField(desc="File listing with size and mtime")
    job_context: str = dspy.InputField(desc="Recipe name, framework, node, what the job does")

    relevant_files: list[str] = dspy.OutputField(desc="Ranked list of absolute file paths to monitor")
    reasoning: str = dspy.OutputField(desc="Brief explanation of why each file was selected")


#: The tier's question, with ``{label}`` naming one entry of the listing.
#:
#: **One definition, here, and the corpus builder imports it.** The alternative
#: had already happened: ``aorta.laya.corpus.log_finder`` had independently
#: written its own phrasing, so a temperature fitted on that corpus would have
#: been fitted against a question this tier never asks. That does not fail, it
#: answers slightly worse, for a reason nobody would go looking for -- the same
#: defect Track B found in the proposer and Phase 5 found in chat. The question
#: belongs to the module that owns the decision, which is this one; the seam in
#: :mod:`aorta.laya.predictor` deliberately holds none of them, because it
#: serves three tracks and a string registry is what it would become.
#: ``tests/cia/test_laya_questions.py`` fails if this text appears anywhere else.
#:
#: A template rather than a fixed string, and that is what makes the shape one
#: forward pass rather than N. Questions about one state share its encoding, so
#: the file has to vary in the *question*; putting it in the state instead would
#: give every entry its own state and turn a sixty-file listing into sixty
#: passes. It is also what tells the answers apart: two entries sharing a
#: question text get one answer, and which file it was about would be whichever
#: the ranking happened to pick.
#:
#: Worded from the same four things ``LogDiscovery`` above asks for -- loss,
#: steps, throughput, faults -- because the two tiers have to be answering the
#: same question for a comparison between them to mean anything.
LAYA_USEFUL_QUESTION_TEMPLATE = (
    "In the file listing above, is {label} a training log worth watching "
    "for loss values, step counts, throughput, GPU errors or stack traces?"
)

#: The two sides, glossed rather than left implicit. A job directory is mostly
#: checkpoints and placeholders, which is what the extension scan's exclude list
#: and its hundred-byte floor already encode; a noul with a gloss on one side
#: only asks the model to guess what the other side of the question is.
LAYA_USEFUL_WHEN_TRUE = "Watch would read useful training output out of this file"
LAYA_USEFUL_WHEN_FALSE = (
    "a checkpoint, a binary, an empty placeholder, or an unrelated file"
)


def useful_question(label: str) -> Noul:
    """The noul asked about one listed file. One definition, per the constants above.

    A function so that the label is filled in exactly one way wherever the
    question is asked -- here at inference, and in the corpus builder that
    labels it.
    """
    from aorta.laya.predictor import Noul

    return Noul(
        question=LAYA_USEFUL_QUESTION_TEMPLATE.format(label=label),
        when_true=LAYA_USEFUL_WHEN_TRUE,
        when_false=LAYA_USEFUL_WHEN_FALSE,
    )


def listing_label(job_dir: Path, listed: str) -> str:
    """How one line of a ``_dir_listing`` is named in its question.

    Exported beside :func:`useful_question` and imported by the corpus builder
    for the same reason the question itself is: the label is *part of* the
    question text, so a corpus that derived it differently -- absolute where
    this is relative, say -- would be labelling a different question however
    faithfully it imported the template.

    Relative to the job because sixty absolute paths is a lot of tokens to spend
    saying the same prefix, and unique because ``find`` lists each file once.
    Resolved on both sides, so a symlinked job directory does not produce a
    label that shares no prefix with its own root.

    Falls back to the bare filename for a listing entry that does not sit under
    *job_dir*. Nothing ``find`` emits can do that, so this covers an injected
    listing in a test rather than a real one.
    """
    try:
        return Path(listed).resolve().relative_to(Path(job_dir).resolve()).as_posix()
    except (OSError, ValueError):
        return Path(listed).name


class LogFinder(dspy.Module):
    """Discover which log files to monitor for a job.

    Config hints (all optional):
      paths      — explicit glob patterns (highest priority)
      extensions — file extensions to include (fallback if paths absent)
      exclude    — patterns to never watch
      max_files  — cap on number of files returned
      laya       — enabled / backend / min_probability for the tier 3 classifier
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        predictor: LayaPredictor | None = None,
    ):
        ensure_configured()
        cfg = config or {}
        self.paths: list[str] = cfg.get("paths", [])
        self.extensions: set[str] = set(cfg.get("extensions", DEFAULT_LOG_EXTENSIONS))
        self.exclude: set[str] = set(cfg.get("exclude", DEFAULT_EXCLUDE_PATTERNS))
        self.max_files: int = int(cfg.get("max_files", DEFAULT_MAX_FILES))
        self._discovery = dspy.Predict(LogDiscovery)

        laya_cfg = cfg.get("laya", {}) or {}
        self.laya_enabled: bool = bool(laya_cfg.get("enabled", False))
        self.laya_backend: str = str(laya_cfg.get("backend", DEFAULT_LAYA_BACKEND))
        self.laya_min_probability: float = float(
            laya_cfg.get("min_probability", DEFAULT_LAYA_MIN_PROBABILITY)
        )
        # *predictor* is the injection point, the same one ``LayaAgentPredictor``
        # offers through ``load=``: a test drives the tier with
        # ``FakeLayaPredictor`` and no checkpoint goes anywhere near CI.
        self._predictor = predictor
        # One LogFinder serves the whole poll loop, so a checkpoint that is not
        # staged on this node must be discovered once rather than on every
        # ambiguous directory for the life of the process.
        self._laya_unavailable = False

    def find(
        self,
        job_dir: Path,
        job_context: str = "",
        scheduler: str = "",
        scheduler_job_id: str = "",
        head_node: str = "",
    ) -> list[Path]:
        """Return ordered list of paths to monitor. Caches nothing — caller caches.

        Priority:
          0. Scheduler-native query (scontrol / kubectl) — most reliable, no guessing
          1. Explicit config path hints
          2. Extension-based scan
          3. Laya, or the LLM, over the dir listing

        **Nothing in this method runs for a job that declared a log.** The only
        caller is the poll loop, which prefers ``job.log_path`` outright, and
        ``aorta.cia.triage`` -- the only writer of ``job.json`` in the tree --
        always sets it. Discovery is reached by a record that carries
        ``log_path: ""``, which is a launcher other than triage, so the whole
        of ``find`` is dormant rather than dead. Tier 3 is not more dormant
        than tier 2; they are behind the same gate. It is recorded here because
        two readers in a row have taken "the LLM tier is expensive" as a claim
        about a call this machine is making, and it is not one.
        """
        job_dir = Path(job_dir)

        # 0. Scheduler-native: ask Slurm/K8s for StdOut/StdErr paths directly
        if scheduler and scheduler_job_id:
            sched_paths = query_scheduler_logs(scheduler, scheduler_job_id, head_node)
            resolved: list[Path] = []
            extra_dirs: list[Path] = []
            for p_str in sched_paths:
                if p_str.startswith("__workdir__:") or p_str.startswith("__dir__:"):
                    extra_dirs.append(Path(p_str.split(":", 1)[1]))
                elif p_str.startswith("__kubectl_logs__:"):
                    pass  # handled separately if needed
                else:
                    p = Path(p_str)
                    if p.is_file() and not self._excluded(p):
                        resolved.append(p)
            # StdOut/StdErr from the scheduler are authoritative. Only trawl the
            # work dir when it named no usable file, or a WorkDir that happens to
            # be a source checkout drags in unrelated .txt files.
            if not resolved:
                for d in extra_dirs:
                    resolved.extend(self._scan_by_extension(d))
            if resolved:
                return resolved[: self.max_files]

        # 1. Explicit path hints — expand globs
        if self.paths:
            found: list[Path] = []
            for pattern in self.paths:
                pattern = pattern.replace("{job_id}", job_dir.name)
                for p in sorted(Path("/").glob(pattern.lstrip("/"))):
                    if p.is_file() and not self._excluded(p):
                        found.append(p)
            if found:
                return found[: self.max_files]

        # 2. Extension-based scan — fast, no LLM
        by_ext = self._scan_by_extension(job_dir)
        if by_ext:
            # If we found files with known extensions, return them directly
            # without burning an LLM call on the obvious case
            if len(by_ext) <= 3:
                return by_ext[: self.max_files]

        # 3. Auto-discovery — for ambiguous or large directories
        listing = _dir_listing(job_dir)
        candidates = self._candidates(listing, job_dir)
        if not candidates:
            return by_ext[: self.max_files]

        if self.laya_enabled:
            ranked = self._rank_with_laya(listing, candidates)
            if ranked:
                return ranked[: self.max_files]
        else:
            named = self._ask_the_model(listing, job_context, job_dir, candidates)
            if named:
                return named[: self.max_files]

        return by_ext[: self.max_files]

    def _candidates(self, listing: str, job_dir: Path) -> list[tuple[str, Path]]:
        """The files tier 3 is allowed to return, as ``(label, resolved path)``.

        This is the security fix, and it is deliberately not behind the Laya
        flag. The hazard is not that an LLM ranks badly, it is that it answers
        in free text: shown a listing and asked which files are logs, it can
        name a path that was never in the listing, and ``is_file()`` plus the
        exclude list were all that stood between such an answer and being
        watched. Watch then grants the parent of every watched path as a root
        for its own file tools, so one hallucinated ``/etc/passwd`` would have
        handed it ``/etc``.

        Deriving the answer space from the listing removes that, and it removes
        it for whichever engine is doing the scoring. Landing it only inside a
        flag that defaults to off would have shipped the fix and the hazard in
        one change with the fix switched off.

        Parsed back out of the rendered listing rather than re-walking the
        directory, and for the same reason the corpus builder in
        ``aorta.laya.corpus.log_finder`` parses it the same way: a second walk
        can return a path the listing does not contain, which is the one
        property this is here to remove.

        The label is :func:`listing_label`'s, not one derived here, because it
        is part of the question text and the corpus builder has to arrive at the
        same one.

        Note what is *not* filtered: ``SKIP_DIR_PARTS``, which keeps the
        extension scan out of ``.git`` and ``site-packages``. The listing shows
        those files, so dropping them here would offer an answer space that
        differs from the prompt. Narrowing the listing itself is the right fix
        and is a change to what the model sees, not to what it may say.
        """
        found: list[tuple[str, Path]] = []
        seen: set[Path] = set()
        for line in listing.splitlines():
            raw, separator, _ = line.partition("  size=")
            if not separator:
                continue
            listed = raw.strip()
            path = self._within_job(listed, job_dir)
            if path is None or path in seen:
                continue
            if not path.is_file() or self._excluded(path):
                continue
            seen.add(path)
            found.append((listing_label(job_dir, listed), path))
        return found

    def _ask_the_model(
        self,
        listing: str,
        job_context: str,
        job_dir: Path,
        candidates: list[tuple[str, Path]],
    ) -> list[Path]:
        """Tier 3 as it has always been, with its answer bound to *candidates*.

        Swallowing every exception is the established contract for this whole
        method: discovery failing must cost the extension scan's answer, not
        the poll loop. What changed is the acceptance test below -- membership
        in the listing rather than ``is_file()``.
        """
        allowed = {path for _label, path in candidates}
        result: list[Path] = []
        try:
            pred = self._discovery(
                job_dir_listing=listing,
                job_context=job_context or f"job_dir={job_dir}",
            )
            for p_str in (pred.relevant_files or []):
                path = self._within_job(p_str.strip(), job_dir)
                if path is not None and path in allowed and path not in result:
                    result.append(path)
        except Exception:
            return []
        return result

    def _rank_with_laya(
        self, listing: str, candidates: list[tuple[str, Path]]
    ) -> list[Path]:
        """Score every listed file in one forward pass and rank by p(worth watching).

        **One state, N questions**, which is the shape the seam's contract makes
        cheap: ``ask`` answers every question about one state in a single pass,
        and only a second *state* costs a second pass. The state is the whole
        listing and there is one noul per entry, so a sixty-file directory is
        one pass. The alternative -- one state per file line -- is sixty passes
        for one directory, and it also throws away the only information the
        judgement has. "Is this worth watching" is comparative: the newest file
        here, the one ``.log`` among forty ``.txt``. A line on its own cannot
        answer it.

        Independent nouls rather than one ``choice`` over the entries for the
        reason the plan gives for the chat selector: a choice is a distribution
        that sums to one, so it can rank but cannot say "none of these are
        logs" or "six of these are", and a high-cardinality option set splits
        the model's fixed option-marker budget. A job directory genuinely has
        zero or six answers more often than it has exactly one.

        What is unmeasured, and is Phase 1's to settle: sixty nouls in one pass
        is far more questions than the ten the selector plans to ask, and no
        one has checked what that does to the budget or the latency on this
        hardware. The ``_dir_listing`` cap of sixty lines bounds it; nothing
        here claims the bound is comfortable.
        """
        predictor = self._laya()
        if predictor is None:
            return []
        try:
            from aorta.laya.predictor import NoulAnswer, ask_one

            answers = ask_one(
                predictor, listing, [useful_question(label) for label, _ in candidates]
            )
            scored: list[tuple[float, Path]] = []
            for (_label, path), answer in zip(candidates, answers, strict=True):
                if not isinstance(answer, NoulAnswer):
                    raise TypeError(
                        f"{type(predictor).__name__} answered a noul with "
                        f"{type(answer).__name__}"
                    )
                if answer.at(self.laya_min_probability):
                    scored.append((answer.probability, path))
        except Exception as exc:
            # Said out loud, once, rather than degraded silently. An operator
            # who turned this on and got the extension scan anyway has no other
            # way to find out; a warning is the difference between "the weights
            # are not staged on this node" and "discovery is worse than it was".
            self._laya_unavailable = True
            print(
                f"[watch] the Laya log-finder tier failed and will not be "
                f"retried this run; falling back to the extension scan: "
                f"{type(exc).__name__}: {exc}"
            )
            return []
        # Stable, so files the classifier cannot separate keep the order the
        # listing gave them rather than an order that changes between polls.
        return [path for _probability, path in sorted(scored, key=lambda pair: -pair[0])]

    def _laya(self) -> LayaPredictor | None:
        """The predictor, built once, or None when this tier cannot run.

        Imported inside the method, not at module scope. ``aorta.cia.watch`` is
        on the import path of every Watch poll and of ``aorta.cli`` through the
        chat tools, and the loader behind this name brings torch; Decision 22
        in ``docs/laya-packaging.md`` is the whole argument, and
        ``tests/cli/test_chat_boundaries.py`` is what enforces it.

        A failed load is remembered rather than retried. The alternative is a
        node with no staged weights paying a load attempt per ambiguous job for
        as long as Watch runs.
        """
        if self._laya_unavailable:
            return None
        if self._predictor is None:
            try:
                from aorta.laya.predictor import make_predictor

                if self.laya_backend == "fake":
                    # Refused rather than resolved, matching the watch.laya
                    # section of watch_config.yaml. ``make_predictor`` would
                    # hand back ``FakeLayaPredictor`` quite happily, and it
                    # answers from a hash of the question and the listing --
                    # stable, arbitrary, and indistinguishable from a model
                    # with an opinion. Here that is a hash choosing which files
                    # Watch tails, and Watch grants the parent of each one as a
                    # root for its own file tools.
                    raise LookupError(
                        "the fake predictor ranks by hash; pass predictor= to "
                        "drive this tier in a test"
                    )
                self._predictor = make_predictor(self.laya_backend)
            except Exception as exc:
                self._laya_unavailable = True
                print(
                    f"[watch] could not build the Laya log-finder predictor "
                    f"{self.laya_backend!r}; falling back to the extension "
                    f"scan: {type(exc).__name__}: {exc}"
                )
                return None
        return self._predictor

    def _within_job(self, candidate: str, job_dir: Path) -> Path | None:
        """*candidate* as a path inside *job_dir*, or None when it is not.

        These come from the model: it is shown a directory listing and asked
        which files are logs, and it can answer with anything. Only is_file()
        and the exclude list stood between that answer and being watched, so
        an absolute path it invented was accepted whenever the file happened
        to exist.

        That does not stop at reading the wrong file. Watch grants the parent
        of every watched path as a root for its own file tools, so one
        hallucinated /etc/passwd would have handed it /etc.

        It is no longer the only barrier: ``_candidates`` now builds tier 3's
        answer space out of the listing itself, so a path that is not in the
        directory has nothing to be accepted from. This stays anyway, and stays
        first, because it is also what makes ``_candidates`` safe -- the listing
        is produced by ``find`` walking the job, and a symlink inside the job
        still points wherever it likes.

        The check is the one the adapters use rather than a second copy of it,
        because two containment checks is how one of them ends up without the
        symlink case.
        """
        return resolve_in_bundle(job_dir, candidate)

    def _scan_by_extension(self, job_dir: Path) -> list[Path]:
        candidates: list[tuple[float, Path]] = []
        try:
            for p in job_dir.rglob("*"):
                if not p.is_file():
                    continue
                if SKIP_DIR_PARTS & set(p.parts):
                    continue
                if self._excluded(p):
                    continue
                # Rooted at job_dir, but rglob walks into symlinked
                # directories, so a link inside the job still leads out.
                if self._within_job(str(p), job_dir) is None:
                    continue
                if p.suffix.lower() not in self.extensions:
                    continue
                if p.stat().st_size < 100:
                    continue
                candidates.append((p.stat().st_mtime, p))
        except Exception:
            pass
        # Sort newest-first
        candidates.sort(reverse=True)
        return [p for _, p in candidates]

    def _excluded(self, p: Path) -> bool:
        name = p.name.lower()
        return any(name.endswith(ext) for ext in self.exclude)
