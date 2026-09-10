#!/usr/bin/env python3
"""Seam demonstration: drive TokenSpeed's RL weight-sync control plane.

Not a workload and not wired into CI. This is the Phase 2 probe route from
`docs/tokenspeed-rl-post-training.md`: point it at a running `tokenspeed serve`
and it records what every weight-transfer endpoint accepts and returns, plus the
wall clock of the operations that would sit in an RL loop's inner cycle.

The question it exists to answer is whether an RL iteration costs a weight
update or a cold restart. Bring-up on gfx950 measures 189-322 s; 400 iterations
of that is ~28 hours of loading weights against ~2.5 hours of generation. If
weights can be swapped in place, that cost disappears.

What a run establishes:

  * pause / resume / is_paused latency, against the cold-start cost it replaces
  * that the lifecycle guards (ordering, validation) actually refuse
  * which backend is wired -- `ipc` parses its metadata and then raises
    NotImplementedError on this image, so only `nccl` is real
  * that the server still serves, unrestarted, after the whole lifecycle

What it does NOT establish: the NCCL tensor transfer itself, which needs a
trainer peer joining the process group. The metadata contract for that is
exercised here; the broadcast is not.

Stdlib only, and no f-string-free dialect needed beyond 3.6 -- this is meant to
run from a compute node's system Python, which on CentOS 9 is 3.9.

Usage:

    python examples/rl/probe_weight_transfer.py \\
        --control http://127.0.0.1:8001 \\
        --gateway http://127.0.0.1:8000 \\
        --model Qwen/Qwen3-0.6B \\
        --out ./weight_transfer_probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


class Probe:
    def __init__(self, control: str, gateway: str, model: str) -> None:
        self.control = control.rstrip("/")
        self.gateway = gateway.rstrip("/")
        self.model = model
        self.results: list[dict[str, Any]] = []

    def call(
        self, method: str, url: str, body: Any = None, timeout: int = 120
    ) -> tuple[int | None, Any, float]:
        data = None
        headers = {}
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        started = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", "replace")
                status = resp.status
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", "replace")
            status = exc.code
        except Exception as exc:  # noqa: BLE001 - a probe reports, it does not raise
            return None, f"EXC: {type(exc).__name__}: {exc}", time.time() - started
        elapsed = time.time() - started
        try:
            return status, json.loads(raw), elapsed
        except ValueError:
            return status, raw[:400], elapsed

    def step(
        self,
        label: str,
        method: str,
        path: str,
        body: Any = None,
        base: str | None = None,
        timeout: int = 120,
    ) -> dict[str, Any]:
        status, payload, elapsed = self.call(
            method, (base or self.control) + path, body, timeout
        )
        record = {
            "step": label,
            "method": method,
            "path": path,
            "request": body,
            "status": status,
            "response": payload,
            "elapsed_sec": round(elapsed, 4),
        }
        self.results.append(record)
        shown = payload if isinstance(payload, str) else json.dumps(payload)
        print(f"[{label:<46}] {method:<4} {path:<34} -> {status}  {elapsed:.3f}s")
        print(f"      {shown[:220]}")
        return record

    def generate(self, label: str, prompt: str = "The capital of France is") -> dict:
        """A real completion, to prove the server serves before and after."""
        record = self.step(
            label,
            "POST",
            "/v1/completions",
            {"model": self.model, "prompt": prompt, "max_tokens": 12, "temperature": 0.0},
            base=self.gateway,
            timeout=180,
        )
        text = None
        if isinstance(record["response"], dict):
            try:
                text = record["response"]["choices"][0]["text"]
            except (KeyError, IndexError, TypeError):
                text = None
        record["completion_text"] = text
        print(f"      completion={text!r}")
        return record


def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def run(probe: Probe) -> None:
    banner("A. Baseline: the server is up and generating")
    probe.step("health", "GET", "/health")
    probe.generate("generate-before")
    probe.step("is_paused (initial)", "GET", "/is_paused")
    probe.step("get_world_size", "GET", "/get_world_size")
    probe.step("get_world_size include_dp=false", "GET", "/get_world_size?include_dp=false")

    banner("B. Lifecycle ordering: are the state-machine guards real?")
    probe.step("start_weight_update before init", "POST", "/start_weight_update", {})
    probe.step(
        "update_weights with no active update",
        "POST",
        "/update_weights",
        {"update_info": {"names": ["x"], "dtype_names": ["bfloat16"], "shapes": [[1]]}},
    )
    probe.step("finish_weight_update with none active", "POST", "/finish_weight_update", {})

    banner("C. Request validation")
    probe.step("init missing init_info", "POST", "/init_weight_transfer_engine", {})
    probe.step("pause invalid mode", "POST", "/pause?mode=bogus")

    banner("D. Pause / resume -- the cost that sits in the RL inner loop")
    for mode in ("abort", "wait", "keep"):
        probe.step(f"pause mode={mode}", "POST", f"/pause?mode={mode}")
        probe.step(f"is_paused after pause({mode})", "GET", "/is_paused")
        probe.step(f"resume after {mode}", "POST", "/resume")
        probe.step(f"is_paused after resume({mode})", "GET", "/is_paused")
    probe.generate("generate-after-pause-resume-cycles")

    banner("E. The weight-update lifecycle itself")
    probe.step(
        "init_weight_transfer_engine", "POST",
        "/init_weight_transfer_engine", {"init_info": {}}, timeout=180,
    )
    probe.step("start_weight_update", "POST", "/start_weight_update", {})
    probe.step(
        "update_weights (dense, 1 tensor)", "POST", "/update_weights",
        {
            "update_info": {
                "names": ["model.embed_tokens.weight"],
                "dtype_names": ["bfloat16"],
                "shapes": [[151936, 1024]],
                "ipc_handles": {"model.embed_tokens.weight": "placeholder"},
            }
        },
        timeout=180,
    )
    probe.step(
        "update_weights unknown key", "POST", "/update_weights",
        {
            "update_info": {
                "names": ["a"], "dtype_names": ["bfloat16"], "shapes": [[1]],
                "ipc_handles": {"a": "p"}, "bogus_key": 1,
            }
        },
    )
    probe.step(
        "update_weights sparse update_kind", "POST", "/update_weights",
        {
            "update_info": {
                "names": ["a"], "dtype_names": ["bfloat16"], "shapes": [[1]],
                "ipc_handles": {"a": "p"}, "update_kind": "sparse",
            }
        },
    )
    probe.step(
        "update_weights missing ipc handles", "POST", "/update_weights",
        {"update_info": {"names": ["a"], "dtype_names": ["bfloat16"], "shapes": [[1]]}},
    )
    probe.step("finish_weight_update", "POST", "/finish_weight_update", {})
    probe.step("start again after finish", "POST", "/start_weight_update", {})
    probe.step("double start (should conflict)", "POST", "/start_weight_update", {})
    probe.step("finish (cleanup)", "POST", "/finish_weight_update", {})

    banner("F. The server still serves after all of that")
    probe.generate("generate-after-lifecycle")
    probe.step("is_paused (final)", "GET", "/is_paused")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--control", default="http://127.0.0.1:8001")
    parser.add_argument("--gateway", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--out", default="weight_transfer_probe.json")
    args = parser.parse_args(argv)

    probe = Probe(args.control, args.gateway, args.model)
    run(probe)

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(probe.results, handle, indent=2)
    print()
    print(f"wrote {args.out} ({len(probe.results)} steps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
