#!/usr/bin/env python3
"""Write arith/{train,heldout}/qa.jsonl (3-digit by 3-digit multiplication) and arith4/{train,heldout}/qa.jsonl
(4-digit by 4-digit, harder), each from a fixed seed, with no pair in both halves. `python examples/qa/make_arith.py`
writes the committed files again, byte for byte."""

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent


def write(name: str, digits: int, train: int = 512, heldout: int = 64, seed: int = 20261113) -> None:
    rng = random.Random(seed)
    lo, hi = 10 ** (digits - 1), 10 ** digits - 1
    pairs, seen = [], set()
    while len(pairs) < train + heldout:
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        if (a, b) not in seen and (b, a) not in seen:
            seen.add((a, b))
            pairs.append((a, b))
    name_ = name
    for part_name, part in (("train", pairs[:train]), ("heldout", pairs[train:])):
        rows = [{"id": f"mul-{a}x{b}", "question": f"What is {a} × {b}?", "answer": str(a * b)} for a, b in part]
        (HERE / name_ / part_name).mkdir(parents=True, exist_ok=True)
        (HERE / name_ / part_name / "qa.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def main() -> None:
    write("arith", 3)
    write("arith4", 4)


if __name__ == "__main__":
    main()
