"""The BGE query prefix that survived the langchain-huggingface migration.

``langchain_community.embeddings.HuggingFaceBgeEmbeddings`` did two things for
us that its replacement does not: it prepended a query instruction, and it
collapsed newlines. BGE models are trained asymmetrically -- queries carry the
prefix, passages do not -- so dropping it changes what gets embedded.

Nothing would have failed. Documents already in Chroma stay readable and every
test still passes; queries simply stop pairing with them the way they were
indexed, and retrieval quietly degrades. Measured drift from a naive swap was
0.026 per component on a 384-dimension unit vector.

These tests do not load the model. They assert the text handed to the parent
class, which is the whole of the behaviour worth pinning.
"""

from __future__ import annotations

from unittest.mock import patch

from langchain_huggingface import HuggingFaceEmbeddings

from aorta.chat.rag.embeddings.local_bge import (
    BGE_QUERY_INSTRUCTION,
    BgeEmbeddings,
    LocalBgeProvider,
)


def _embeddings() -> BgeEmbeddings:
    """Build without touching the network or loading weights."""
    return BgeEmbeddings.model_construct()


class TestQueryInstruction:
    def test_the_prefix_matches_langchain_communitys_default_verbatim(self):
        """Any drift here changes vectors against an existing index."""
        assert BGE_QUERY_INSTRUCTION == (
            "Represent this question for searching relevant passages: "
        )

    def test_a_query_is_prefixed(self):
        with patch.object(HuggingFaceEmbeddings, "embed_query") as parent:
            _embeddings().embed_query("how do I profile?")
        parent.assert_called_once_with(
            BGE_QUERY_INSTRUCTION + "how do I profile?"
        )

    def test_newlines_in_a_query_become_spaces(self):
        with patch.object(HuggingFaceEmbeddings, "embed_query") as parent:
            _embeddings().embed_query("line one\nline two")
        parent.assert_called_once_with(BGE_QUERY_INSTRUCTION + "line one line two")


class TestDocuments:
    def test_documents_are_not_prefixed(self):
        """Asymmetric by design: the prefix belongs on queries only."""
        with patch.object(HuggingFaceEmbeddings, "embed_documents") as parent:
            _embeddings().embed_documents(["def run(): ..."])
        parent.assert_called_once_with(["def run(): ..."])

    def test_newlines_in_documents_become_spaces(self):
        with patch.object(HuggingFaceEmbeddings, "embed_documents") as parent:
            _embeddings().embed_documents(["def run():\n    return 1\n"])
        parent.assert_called_once_with(["def run():     return 1 "])

    def test_every_document_is_handled(self):
        with patch.object(HuggingFaceEmbeddings, "embed_documents") as parent:
            _embeddings().embed_documents(["a\nb", "c\nd"])
        parent.assert_called_once_with(["a b", "c d"])


class TestProviderStillUsesIt:
    def test_the_local_provider_returns_the_prefixing_class(self):
        """Guards against a future edit reverting to plain HuggingFaceEmbeddings."""
        import inspect

        source = inspect.getsource(LocalBgeProvider.get_embeddings)
        assert "BgeEmbeddings(" in source

    def test_the_collection_name_is_unchanged_by_the_migration(self):
        """A different name would orphan every existing index."""
        assert LocalBgeProvider().collection_name() == "aorta"
