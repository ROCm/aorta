"""The run entry point: `python -m aorlta.orchestrator.run --config RUN.yaml [--resume] [--components fake|real]`.

`--components fake` drives the loop with the thin fakes of `fakes.py`, from the same config file a real run reads
(the CPU smoke test of the job script). `--components real` wires WP2's rollout, WP5's reward service, WP4's trainer
and WP3's sync through `aorlta.orchestrator.wiring`, which refuses (exit 2) a configuration it cannot run.

Exit status: 0 the run completed, 3 it stopped (the reason is in run_result.json), 2 a usage or wiring error,
1 an unexpected error. Only a stand-in run prints a traceback: an exception's text may carry the data it handled (6.10).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

from ..config import ConfigError, load_config
from .ports import describe_error


def fake_components(cfg, *, config_path: Path | None = None, commit: str | None = None):
    from ..spans import SpanRecorder
    from .fakes import (FakeEvalHook, FakeExporter, FakeReward, FakeRollout, FakeRolloutConfig, FakeSync, FakeTrainer,
                        FakeTrainerConfig, VirtualClock, ensure_fake_base, make_pool)
    from .loop import Orchestrator

    ensure_fake_base(cfg)
    clock = VirtualClock()
    spans = SpanRecorder(monotonic=clock.now)
    rollout = FakeRollout(cfg, clock, spans, FakeRolloutConfig(engines=len(cfg.rollout.engines)))
    trainer = FakeTrainer(FakeTrainerConfig(min_scored=cfg.orchestrator.min_scored_per_group))
    sync = FakeSync(trainer, rollout, clock, spans, run_id=cfg.run.run_id)
    hook = None
    if cfg.orchestrator.eval_every is not None:
        hook = FakeEvalHook(cfg.run_dir / "eval", run_id=cfg.run.run_id, every=cfg.orchestrator.eval_every,
                            final_iteration=cfg.orchestrator.iterations, exporter=FakeExporter(sync), base_sha256=None,
                            spans=spans, source=cfg.run.source)
    return Orchestrator(cfg, pool=make_pool(2 * cfg.rollout.P, source=cfg.run.source), rollout=rollout,
                        reward=FakeReward(clock), trainer=trainer, sync=sync, eval_hook=hook, clock=clock,
                        spans=spans, commit=commit, config_path=config_path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="aorlta.orchestrator.run", description="run AORLTA's v1 loop (Section 6.6)")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--resume", action="store_true", help="resume the run directory from its last checkpoint")
    ap.add_argument("--components", choices=("fake", "real"), default="real")
    ap.add_argument("--commit", default=None, help="the code commit, when git is not available (in a container)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        cfg = load_config(args.config)
    except (ConfigError, OSError) as exc:
        print(f"cannot load the config: {exc}", file=sys.stderr)
        return 2
    if args.components == "fake":
        orch = fake_components(cfg, config_path=args.config, commit=args.commit)
    else:
        from . import wiring

        try:
            orch = wiring.build(cfg, config_path=args.config, commit=args.commit)
        except wiring.WiringError as exc:
            print(f"cannot wire the real components: {exc}", file=sys.stderr)
            return 2
    try:
        result = orch.run(resume=args.resume)
    except Exception as exc:  # noqa: BLE001 - run_result.json already records it
        if cfg.run.source == "standin":
            traceback.print_exc()
        print(f"unexpected error: {describe_error(exc, cfg.run.source)}", file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "iterations_completed": result["iterations_completed"],
                      "policy_version": result["policy_version"],
                      "stop": (result.get("stop") or {}).get("reason")}))
    return 0 if result["status"] == "completed" else 3


if __name__ == "__main__":
    sys.exit(main())
