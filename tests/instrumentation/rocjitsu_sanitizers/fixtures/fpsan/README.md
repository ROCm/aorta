Stdout of a recipe `fpsan_command`, in the `AORTA_FPSAN_RESULT` contract shape.
These are hand-written, not captured from a GPU run: no customer reference /
candidate command exists yet. The digests are SHA-256 of fixed strings and stand
in for the SHA-256 of two sanitized outputs. Replace them with a verbatim
capture once a live recipe lands.

| file | what it exercises |
|---|---|
| `match.txt` | both instrumented, equal digests: `pass` |
| `mismatch.txt` | both instrumented, different digests: `fail` |
| `candidate_not_instrumented.txt` | equal digests but the candidate was not instrumented: `error` |
| `no_result_line.txt` | no result line: `error` |
| `two_result_lines.txt` | two result lines: `error` |
| `aka_marker.txt` | AgentKernelArena's `AKA_FPSAN_RESULT` marker only: `error` |
| `unparsable.txt` | a truncated JSON payload: `error` |
