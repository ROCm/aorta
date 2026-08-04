---
applyTo: "docs/**,**/README.md"
---

# Disclosure review for prose

Narrative files are where private material has actually reached public `main` in
this repository, because prose carries the incidental detail that code does not:
the machine someone ran it on, the internal document that explains why, the
customer the investigation was really about. Apply the categories in
`.github/copilot-instructions.md`, and additionally check the following.

## Is this document written for a public reader?

A page belongs here only if a ROCm user outside AMD could act on it. Ask who the
audience is:

- **Reproducible by anyone** -- flag nothing. Setup guides, CLI walkthroughs,
  schema references, and design notes for published features are the point of
  `docs/`.
- **Reproducible only by the author** -- flag it. A walkthrough pinned to one
  developer's host, container name, absolute home-directory path, or scratch
  mount is a private artifact even when nothing in it is formally confidential.
  The remedy is usually to parameterise the specifics, not to delete the page.
- **Written for an internal audience** -- flag it. Escalation narratives, status
  updates addressed to an internal team, and design documents for unpublished
  work belong in `aorta-internal`. Public planning documents tied to public
  issues are fine and live in `docs/plans/`.

## Checks specific to prose

- **Example commands and shell blocks.** Every host, path, container name,
  registry, and image tag in a fenced block is published. Prefer a placeholder
  (`<your-host>`, `$HOME`, `<output-dir>`) with the requirement stated in prose
  over a literal value that happened to work on one machine.
- **Pasted output.** Sample terminal output, `env probe` snapshots, and log
  excerpts should be synthetic or from AMD hardware, and trimmed to the lines
  that illustrate the point. A full paste tends to carry hostnames, absolute
  paths, and environment details nobody reviewed.
- **Cross-links.** Links must resolve for an anonymous reader. A link to a
  private repository path, an internal wiki, or a document store is both a broken
  link and a disclosure of the internal structure. Naming a private issue by
  number is fine; linking into private content is not.
- **Deletions that leave dangling references.** When a page is removed because it
  should not have been public, the same pull request has to drop every reference
  to it -- other docs, README links, and module docstrings. A stale link tells a
  reader exactly what was withdrawn and invites them to go looking for it.

## Do not rewrite technical substance

These instructions cover what is safe to publish, not how the documentation
should read. Do not use a disclosure review to argue for restructuring a page,
changing its tone, or expanding its coverage.
