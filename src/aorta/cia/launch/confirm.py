from __future__ import annotations

import sys


class DeployError(Exception):
    pass


def confirm_gate(questions: list[str], context: str = "") -> dict[str, str]:
    """Print questions and collect user answers interactively.

    Raises DeployError if stdin is not a TTY (non-interactive mode).
    """
    if not sys.stdin.isatty():
        raise DeployError(
            f"Deploy needs input but running non-interactively. "
            f"Unresolved questions:\n" + "\n".join(f"  - {q}" for q in questions)
        )

    if context:
        print(f"\n[deploy] {context}")

    answers: dict[str, str] = {}
    for q in questions:
        print(f"\n[deploy] {q}")
        try:
            answers[q] = input("  → ").strip()
        except (EOFError, KeyboardInterrupt):
            raise DeployError("Interrupted during confirmation.")

    return answers


def proceed_gate(summary: str) -> bool:
    """Always-show confirmation before executing any SSH command."""
    if not sys.stdin.isatty():
        # Non-interactive: print summary but proceed (CI/script mode)
        print(f"\n[deploy] Auto-proceeding (non-interactive):\n{summary}")
        return True

    print(f"\n[deploy] Ready to execute:\n{summary}")
    try:
        ans = input("\nProceed? [y/N] ").strip().lower()
        return ans in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False
