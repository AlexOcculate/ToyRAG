# Line 1:  Import re module for tokenization (splitting text into words).
import re

# Line 3:  Define constants for retrieval configuration.
TOP_K = 3  # Line 4:  Maximum number of documents to retrieve per query.
MIN_SCORE_THRESHOLD = (
    0.15  # Line 5:  Minimum relevance score (below this = not relevant).
)


# Line 8:  Define the Retriever class that scores and ranks documents by relevance.
class Retriever:
    # Line 10:  Constructor accepts a CorpusLoader to access the document collection.
    def __init__(self, corpus):
        # Line 12:  Store the corpus reference for document access.
        self._corpus = corpus

    # Line 15:  Main retrieval method: given a query, return top-K scored documents.
    def retrieve(self, query: str) -> list[dict]:
        # Line 17:  Tokenize the query into individual words for matching.
        query_tokens = self._tokenize(query)
        # Line 18:  If the query has no meaningful tokens, return empty results.
        if not query_tokens:
            # Line 20:  An empty token set means we can't match anything.
            return []

        # Line 23:  Score each document in the corpus against the query.
        scored_docs = []
        # Line 24:  Iterate over all (doc_id, doc_text) pairs from the corpus.
        for doc_id, doc_text in self._corpus.get_all():
            # Line 26:  Compute the relevance score for this document.
            score = self._score_document(query_tokens, doc_text)
            # Line 27:  Only keep documents that meet the minimum threshold.
            if score >= MIN_SCORE_THRESHOLD:
                # Line 29:  Append a result dict with id, score, and text for later use.
                scored_docs.append(
                    {"id": doc_id, "score": round(score, 4), "text": doc_text}
                )

        # Line 32:  Sort by score descending (most relevant first).
        scored_docs.sort(key=lambda d: d["score"], reverse=True)
        # Line 33:  Return only the top-K results.
        return scored_docs[:TOP_K]

    # Line 36:  Compute a relevance score between query tokens and a document's text.
    def _score_document(self, query_tokens: list[str], doc_text: str) -> float:
        # Line 38:  Tokenize the document text into words.
        doc_tokens = self._tokenize(doc_text)
        # Line 39:  If the document is empty, it can't match anything.
        if not doc_tokens:
            # Line 41:  Return zero score for empty documents.
            return 0.0

        # Line 44:  Count how many query tokens appear in the document (case-insensitive).
        matches = sum(1 for qt in query_tokens if qt in doc_tokens)
        # Line 45:  Score = fraction of query terms found in the document.
        score = matches / len(query_tokens)

        # Line 47:  Bonus: if the document is short and has high match ratio, boost slightly.
        # Line 48:  This favors concise, focused documents over long general ones.
        if len(doc_tokens) < 100 and score > 0.5:
            # Line 50:  Apply a small bonus for concise, highly-matching docs.
            score = min(1.0, score * 1.1)

        # Line 52:  Return the final relevance score (capped at 1.0).
        return min(1.0, score)

    # Line 55:  Tokenize text into lowercase words (simple whitespace + punctuation split).
    def _tokenize(self, text: str) -> list[str]:
        # Line 57:  Use regex to find all word characters (letters, digits, underscores).
        tokens = re.findall(r"\b[a-zA-Z_]+\b", text.lower())
        # Line 58:  Filter out common stop words that don't carry meaning.
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "and",
            "or",
            "it",
            "this",
            "that",
            "what",
            "how",
            "why",
            "does",
            "do",
            "can",
            "could",
            "would",
            "should",
        }
        # Line 61:  Keep only tokens that are not stop words.
        return [t for t in tokens if t not in stop_words]
