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
aorta chat index status          # compare the local index against the published one
aorta chat doctor                # extras, backend reachability, index freshness
```

`index runs` is the only one that touches the second collection, and the only
one you need after a sweep. `build`, `fetch` and `--from` all replace the source
collection and leave it alone.

`index status` writes nothing. It reads the local manifest, downloads the
published one — about a kilobyte, not the index — and prints both sides plus a
one-line verdict, with `--json` for scripting. A published manifest it cannot
read is reported as *no baseline* rather than as *up to date*, and the verdict
names which asset it compared against, because a `.dev` install resolves to the
rolling `main` tag rather than to a release.

Two verdicts exit non-zero, and both mean no comparison was made: *no baseline*,
and *unreadable local index* — a file at the index path with no manifest this
build can read, which is also what the first query would refuse. An **absent**
local index exits zero; that is a normal answer for someone who has not
installed one yet. When the local sidecar is unreadable *and* the published
index is one this install could not use, the local failure is the verdict: it is
the half the reader can act on, and the half that exits non-zero.

### What replaces what

Overwriting is guarded, and deliberately not symmetrically. A fetched index
costs a download to replace; a locally built one may not be reproducible at all,
because the tree it indexed may have moved and a node with no egress cannot
re-download the embedding weights.

| You run | Over an index that was | Result |
|---|---|---|
| `fetch` | downloaded, and identical | *already up to date*; no asset is transferred |
| `fetch` | downloaded, and different | replaced, printing what changed |
| `fetch` or `--from` | built locally | refused; pass `--force` |
| `build` | built locally | rebuilt, as usual |
| `build` (any corpus but the published one) | downloaded, and usable | refused; pass `--force` |
| `build --public-only` | downloaded | rebuilt; it is the same corpus, so nothing is lost |
| `build` | downloaded, but not usable by this install | rebuilt; it cannot answer anything, so nothing is lost |
| `fetch` or `--from` | a manifest whose `corpus_roots` cannot be read | refused; pass `--force` |
| `build` | a manifest whose `corpus_roots` cannot be read | refused, unless a row above already exempts it; pass `--force` |
| any of them, `--public-only` included | *not an index at all* — a path that exists with no manifest beside it | refused; pass `--force` |

`fetch` also opens the store before deciding a refresh would change nothing.
A matching `index_sha256` says the right index was installed, not that the file
is still one, so a damaged store under an untouched sidecar is re-fetched rather
than reported as *already up to date* — by the one command that would repair it.
"Damaged" includes a store that opens cleanly but holds no chunks for the
collection this install queries, which is what an index built by the other
embedding provider and a build that stopped early both look like.

When the published manifest itself is malformed, the refusal says so and does
**not** offer a re-fetch: the same release yields the same bytes, so the remedies
are another release (`--version`) or a local build. A malformed sidecar carried
in with `--from` asks for the file to be re-staged, for the same reason. This is
a rule rather than three messages — an error raised by `index fetch` may not
advise an `index fetch` that would fail identically.

A refusal names what would be lost and the flag that proceeds anyway, following
`config init --force` rather than prompting, so a script and a terminal behave
identically. The command it prints carries any non-default `--path` and
`--output` the refused invocation used, so pasting it acts on the index in
question rather than on the cache; over the default paths it prints the short
form, because there the two are the same index. An index the running
configuration cannot *use* is never protected:
rebuilding one you cannot query loses nothing, and `doctor` tells you to rebuild
it. "Cannot use" is the same question the first query asks, store included — an
embedding-model or identity mismatch, a manifest that disagrees with the
contents, or a `.sqlite` that cannot be opened at all. A merely *stale* index is
not in that set; it still answers, so a narrowing build over it is still refused.

Which side built an index is read off its manifest's `corpus_roots`, so an index
whose manifest records *no* roots — one built before the field existed — is not
protected either way. One that records roots this build cannot read is a broken
sidecar rather than an old one, and is refused rather than guessed at.

No manifest *at all* is the last row, and it is about the destination rather
than about provenance. Every write path here leaves its sidecars, so a path that
exists without them is not an index this tool finished installing: it is a
mistyped `--output`, a file something else owns, or an index whose sidecars were
lost. Only the first two are unrecoverable, and nothing on disk tells them
apart, so all three are refused. A path that does *not* exist is a first
install and is always permitted — that distinction is the whole rule.

This row is checked **before every exemption above it**, `--public-only`
included. The exemptions answer "is this a narrowing, or an index that cannot
be queried anyway" — questions that presume an index is what is there, which is
the one thing this case does not establish. A typo is a typo on the CI path
too. It costs the published build nothing: `nightly.yml` and `release.yml` run
on a fresh workspace and never restore `index-out/`, so their first write is to
a path that does not exist and every later one is over complete sidecars.

The exemptions come first, which is why the `build` row above is qualified. An
unreadable `corpus_roots` means the index is either a published one or a local
one, and both of the `build` exemptions give the same answer down both branches:
a `--public-only` build records the published corpus whichever it replaced, and
an index this install cannot use is unusable whoever built it — while a local
index is never protected from `build` in the first place. So proceeding there is
not a guess about which one is on disk; it is what both possibilities agree on.
`fetch` and `--from` carry no such exemption, so for them the refusal is
unconditional.

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
against this install's provider identity before replacing the index you already
have — the comparison is the model name and the collection identity, not only
the flow. So `fetch` works exactly as long as your install still embeds with
the default local model: a hand-set `embedding_model` is refused just as a
remote provider is, before anything is installed. That is the reason [choosing a
remote
embedder](configuration.md#configuring-a-remote-embedding-provider-by-hand)
means taking over the build.

What is protected is the destination, not the transfer. A fetch downloads the
asset into a staging directory beside the destination first and validates the
manifest after, so a refused fetch has still spent the download — it just
leaves the index you had untouched, because installing is a rename of the
staged file and it never happens. Expect the bandwidth, not a fail-fast.

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
*failing query* prints are not conditioned yet and can still name the fetch.
On a customised local model, read them as `build`. On a remote provider, do not
substitute either command blindly — whether `build` can run at all depends on
whether `remote_embedding_api_key` is set (see below), so run `aorta chat
doctor`, which conditions its remedy on that.

Note that a fetch *replaces* whatever index is at the configured path. Nothing
in the manifest is *consulted* when choosing that wording. `build_index` does
record provenance — `corpus_roots` and `corpus_digest` — and `corpus_roots`
distinguishes a local build over an absolute root from a published build over
its subpaths. Neither `doctor` nor the validation messages read it yet when
they pick a refresh command, and a manifest written before those fields existed
carries neither, so the advice cannot currently tell a published index from one
you built over a different `aorta_path` — if you have one of those, `build` is
your refresh command, not `fetch`.

Withholding the fetch is only half of it, because a remote embedder is often
not a decision anyone made — a profile for a remote LLM is where the setting
usually comes from, and no template selects a remote embedder any more.
Nothing rewrites a `chat.toml` that already exists, though, so a profile
written by an older install still carries it until it is edited or
regenerated. So `aorta chat doctor` also warns when the
configured provider is remote *and* there is no index this install can query,
and names the edit: `embedding_provider = "local"`, the
`export AORTA_CHAT_EMBEDDING_PROVIDER=local` spelling for one shell
session, or `aorta chat
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
needs neither a key nor a rebuild — on the default `embedding_model`. That
setting is independent of the provider and survives the switch, and the local
provider reads it verbatim, so a profile carrying a custom one still has the
published asset refused on it and does need a build (or the model set back to
its default). The remedy lists condition on this, which is why one of them
offers `build` where the others offer `fetch`.

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
collection registered, a vector for each chunk beside it, and values in those
chunks a document can be built out of — that is
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
