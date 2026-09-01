# The RAG index

Chat answers from retrieval, so the index is what decides whether an answer is
grounded or plausible-sounding. This page covers what is in it, when it must be
rebuilt, and the one failure mode worth being paranoid about.

## Two collections in one file

The index is a single sqlite file — `$XDG_CACHE_HOME/aorta/chat/index.sqlite`
by default — holding two collections with different provenance and different
refresh cadences.

| Collection | Contents | Rebuilt when | Leaves the machine? |
| --- | --- | --- | --- |
| Source | The AORTA tree at `aorta_path`: code, docs, recipes | You upgrade or move the checkout | It is public source, and it is what the published index contains |
| Run artifacts | *Your* sweep output: `matrix.json`, `env.json` | You run a sweep you want to ask about | **Never.** Built locally, never published |

The split is not tidiness. The run-artifact collection is per-user data that can
contain customer hostnames, filesystem layouts and environment variables, so it
must never be built or shipped by CI, and rebuilding the source collection must
not touch it. Retrieved chunks from it are also the reason
[redaction](redaction.md) is on by default.

Alongside the index, chat generates a **repo map** — a function and class index
over the same tree, at `$XDG_CACHE_HOME/aorta/chat/repo_map.md`. The planner
gets a capped slice of it in its prompt (`repo_map_prompt_max_chars`), and the
`search_repo_map` tool queries the whole file.

## The index is valid for exactly one configuration

An index is a set of vectors, and vectors only mean anything to the model that
produced them. Change any of the following and the old index is not stale, it is
wrong:

- the embedding model, and therefore its **dimensions**;
- `chunk_size` / `chunk_overlap`;
- the source tree the vectors describe.

The first is enforced structurally. The store records each collection's vector
dimension in a registry table and keys its tables on the collection name — a
sqlite-vec table's dimension is fixed in its `CREATE` statement — so switching
embedding provider produces a named error rather than a crash from inside the
extension, and the two providers' collections can coexist in one file. The local
and remote providers use different collection names for exactly this reason.

The third is the dangerous one, because **nothing about it errors**. An index
built against a different revision of AORTA does not fail; it answers
confidently out of code you do not have. For a debugging assistant that is the
worst available failure mode, and no amount of care in the prompt detects it.

So: rebuild after upgrading `amd-aorta`, after pointing `aorta_path` somewhere
else, and after changing the chunk settings.

## Managing it

> **Every index carries a manifest.** It records the AORTA version, embedding
> model, dimensions and chunk parameters, which is what makes drift detectable:
> an index whose model or dimensions disagree with the running configuration is
> refused rather than queried, because a silent mismatch returns plausible
> nonsense instead of an error. `aorta chat doctor` reports what it finds.

```bash
aorta chat index build           # build locally from aorta_path
aorta chat index fetch           # download the index matching your version
aorta chat index fetch --from ./index.sqlite   # side-load, for an air-gapped node
aorta chat doctor                # extras, backend reachability, index freshness
```

`fetch` is the normal path: the source collection is identical for every user of
a given AORTA revision, so building it locally is work someone already did. It
resolves by installed version — a released wheel gets that release's asset, a
`.dev` build gets the rolling `main` asset with a warning about the commit delta.

Building locally is the developer path and the air-gapped path. It takes a few
minutes and runs on CPU.

The log tells you what it indexed. Expect a few hundred files; tens of thousands
means `aorta_path` is pointing at a tree that includes build output.

## Air-gapped nodes

`index fetch --from <path>` side-loads an index someone copied in, which solves
half the problem. The other half is the embedding model: the local provider
downloads its weights (~130 MB) from Hugging Face on first use, so a node with
no egress can neither fetch an index nor build one.

Pre-seed the model cache on a connected machine, copy it across, and point at it:

```bash
export HF_HOME=/shared/hf-cache
export HF_HUB_OFFLINE=1
```

The failure without this arrives as a Hugging Face connection error from inside
the embedding library, which reads as a bug rather than as "you need to pre-seed
a cache" — so if that is what you are looking at, this is the section you want.

Alternatively set `embedding_provider = "remote"` if the node can reach an
embeddings API but not Hugging Face; note that this changes the collection name
and requires building the index against that provider.
