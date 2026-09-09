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
| Run artifacts | *Your* sweep output: `matrix.json`, `env.json` | You run a sweep you want to ask about | **Never published.** Built locally, and never shipped as an index asset — but see the note below on remote embeddings |

The split is not tidiness. The run-artifact collection is per-user data that can
contain customer hostnames, filesystem layouts and environment variables, so it
must never be built or shipped by CI, and rebuilding the source collection must
not touch it. Retrieved chunks from it are also the reason
[redaction](redaction.md) is on by default.

> **"Never published" is not the same as "never sent."** With
> `embedding_provider = "remote"`, building this collection sends the rendered
> `matrix.json` and `env.json` text to the embeddings API so it can be turned
> into vectors, and each later query is sent the same way. That path is not
> covered by chat-message redaction, which applies to the LLM request rather
> than the embeddings one. Keep `embedding_provider = "local"` — the default,
> and what every `config init` profile writes — if this data must not leave the
> machine. Turning it off is a manual change with a
> [documented procedure and a documented cost](configuration.md#configuring-a-remote-embedding-provider-by-hand).

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
aorta chat index build           # build the source collection from aorta_path
aorta chat index fetch           # download the index matching your version
aorta chat index fetch --from ./index.sqlite   # side-load, for an air-gapped node
aorta chat index runs            # (re)build the run-artifact collection, locally
aorta chat doctor                # extras, backend reachability, index freshness
```

`index runs` is the only one that touches the second collection, and the only
one you need after a sweep. `build`, `fetch` and `--from` all replace the source
collection and leave it alone.

**Interrupting any of them is safe.** `build`, `fetch` and `--from` write the
new index beside the old one and move it into place in a single step, then write
the manifest, so a `Ctrl-C` leaves the previous index and its manifest exactly as
they were rather than a half-populated file under a manifest describing the
complete one. If a run is interrupted in the window after the move, the manifest
no longer matches the chunk count in the index and every query is refused until
you rebuild or re-fetch — which is the point, because that state cannot produce a
correct answer and would not otherwise announce itself.

`index runs` cannot move a file, because its collection shares the `.sqlite`
with the source one. It gets the same guarantee a different way: it embeds into
a scratch database and swaps the finished collection in one transaction, so an
interruption — or a remote embedding endpoint that stops answering part way
through — leaves the collection you already had, rather than an empty or
half-rebuilt one.

`fetch` is the normal path: the source collection is identical for every user of
a given AORTA revision, so building it locally is work someone already did. It
resolves by installed version — a released wheel gets that release's asset, a
`.dev` build gets the rolling `main` asset with a warning about the commit delta.
It is the normal path for every chat provider, because the embedding provider
is a separate choice and every `config init` profile leaves it local. The asset
is built by CI under default settings, and `fetch_index` validates its manifest
against this install's provider identity before writing anything — the
comparison is the model name and the collection identity, not only the flow. So
`fetch` works exactly as long as your install still embeds with the default
local model: a hand-set `embedding_model` is refused just as a remote provider
is, before anything is installed. That is the reason [choosing a remote
embedder](configuration.md#configuring-a-remote-embedding-provider-by-hand)
means taking over the build.

Because both of those are knowable up front, `aorta chat index fetch` is only
*offered* as a remedy when this install could actually install what it
downloads — a fetch that would be refused is not proposed in the first place.
`aorta chat doctor` and the manifest-validation messages check both conditions
before advising, for an absent index, a refused one and a stale one, and name
`aorta chat index build` instead, which embeds with whatever this install is
configured for.

A third condition withholds *both* index commands rather than choosing
between them: an `embedding_provider` that names a provider AORTA does not
have. `fetch` and `build` each resolve the provider before doing anything
else, so a typo in that one setting makes both fail identically. `doctor`
reports it as a configuration error on the `embedding provider` row and gives
the same remedy on both rows — set `embedding_provider` to a name that exists;
the index commands become available again once it does.

Two things that deliberately do *not* withhold it. `chunk_size` and
`chunk_overlap` drift is a warning, not a refusal, and `fetch_index` installs
through it, so a fetch is still the right advice there. And the messages a
*failing query* prints are not conditioned yet and can still name the fetch;
on a remote provider or a customised model, read them as `build`.

Note that a fetch *replaces* whatever index is at the configured path. Nothing
in the manifest records which corpus an index was built from, so neither
`doctor` nor the validation messages can tell a published index from one you
built over a different `aorta_path` — if you have one of those, `build` is your
refresh command, not `fetch`.

Withholding the fetch is only half of it, because a remote embedder is often
not a decision anyone made — a profile for a remote LLM is where the setting
usually comes from, and no template selects a remote embedder any more.
Nothing rewrites a `chat.toml` that already exists, though, so a profile
written by an older install still carries it until it is edited or
regenerated. So `aorta chat doctor` also warns when the
configured provider is remote *and* there is no index this install can query,
and names the edit: `embedding_provider = "local"`, the
`AORTA_CHAT_EMBEDDING_PROVIDER=local` spelling for one session, or `aorta chat
config init --force --profile <name>` to rewrite the profile from the current
template — `--profile` is required and `chat.toml` does not record which one
wrote it, so that last option needs a name you remember. It stays
quiet for a remote embedder with an index built to match, which is a correct
setup that should not be told to change.

Those profiles are usually also missing `remote_embedding_api_key` — no profile
template prompts for it, and it does not fall back to the key the chat model
uses — so `doctor` will not tell them to build the index locally either. Every
chunk of the corpus goes through the embeddings API, and `RemoteApiProvider`
raises on an empty key before it sends anything, so with no key that build
fails on the first chunk. Switching to local embeddings is the remedy that
needs neither a key nor a rebuild.

That is knowable from settings alone, with no network call, so the remedy lists
act on it: with `embedding_provider = "remote"` and no key, **neither** index
command is offered. The fetch is refused because the published asset is built
with the local embedder, and the build is refused because the provider cannot
be constructed — so the list leads with the switch to local, which is the only
remedy that runs, and then says why the other two are missing rather than going
quiet about them. Set the key and the build comes back.

A fetch downloads vectors, not the embedding model, so straight afterwards
`doctor` reports the model cache as cold. With an index present that this
install could query — one that opens, matches the manifest, and holds a
collection retrieval can actually read: chunks for this provider, the
collection registered, and a vector for each chunk beside it — that is
reported as `[ -- ] ... does not need to be yet`,
because the weights download themselves on the first query. An index that fails
any of those stays a warning: there is then nothing for the softer wording to
protect, and "nothing to do" over an unusable index withholds the remedy.
**Do not "pre-warm" with `aorta chat index build`**: its `--output` defaults to
the index you just fetched and its corpus defaults to `src/aorta` alone, so it
is an attempt to replace a code-and-prose index with a code-only one, dropping
`docs/` and `README.md` out of retrieval. To download the weights on their own,
`doctor` prints the one-line `TextEmbedding` command.

Building locally is the developer path and the air-gapped path. It takes a few
minutes and runs on CPU.

The log tells you what it indexed. Expect a few hundred files; tens of thousands
means `aorta_path` is pointing at a tree that includes build output.

## Air-gapped nodes

`index fetch --from <path>` side-loads an index someone copied in, which solves
half the problem. The other half is the embedding model: the local provider
downloads its weights from Hugging Face on first use, so with no egress
`index fetch` (which resolves a release asset over the network) and `index
build` both fail — the first for want of the asset, the second for want of the
model.

The download is ~65 MB, not the 130 MB of the fp32 weights: `fastembed` serves
this model from a quantised ONNX re-host (`qdrant/bge-small-en-v1.5-onnx-q`),
which is also why its vectors are not interchangeable with a torch-built
index's.

Pre-seed the model cache on a connected machine, copy it across, and point at it:

```bash
export HF_HOME=/shared/hf-cache
export HF_HUB_OFFLINE=1
```

`HF_HOME` is checked first and wins when set. Without it the cache is
`model_cache_path`, which defaults under `$XDG_CACHE_HOME` — set that instead
if you would rather keep the weights out of a shared Hugging Face cache.

The failure without this arrives as a Hugging Face connection error from inside
the embedding library, which reads as a bug rather than as "you need to pre-seed
a cache" — so if that is what you are looking at, this is the section you want.

**A node that can reach an embeddings API but not Hugging Face** is the one
case remote embeddings are genuinely for — but only where the pre-seeded cache
above is out of reach, because that keeps embeddings local, free and unmetered
on exactly the same node. Reach for it first. Remote is not a
drop-in switch: it changes the collection name, makes the published asset
unusable so `index build` becomes mandatory on every upgrade, bills you per
query as well as per build, and sends the corpus — including the run-artifact
collection's `env.json` and `matrix.json` — to that API on a path redaction does
not cover. The full procedure and the rest of the trade-off is
[configuring a remote embedding provider by hand](configuration.md#configuring-a-remote-embedding-provider-by-hand).

A genuinely air-gapped node cannot reach the embeddings API either, so for that
one the answer is the pre-seeded model cache above, not this.
