# Line 1:  Import CorpusLoader to load documents from disk.
from RAG.corpus import CorpusLoader

# Line 2:  Import Retriever for scoring and ranking documents.
from RAG.retriever import Retriever


# Line 5:  Define the RAGPipeline class that orchestrates retrieve → augment → (generate is external).
class RAGPipeline:
    # Line 7:  Constructor wires up the corpus loader and retriever.
    def __init__(self, corpus_dir: str = None):
        # Line 9:  Create a CorpusLoader pointing to the given (or default) directory.
        self.corpus = CorpusLoader(corpus_dir)
        # Line 10:  Create a Retriever that uses this corpus for scoring.
        self.retriever = Retriever(self.corpus)

    # Line 13:  Core method: retrieve relevant docs and build an augmented context string.
    def retrieve_and_augment(self, query: str) -> tuple[str, list[dict]]:
        """
        Returns:
            augmented_context (str): The retrieved document text to inject into the LM prompt.
            retrieved_docs (list[dict]): Metadata about which docs were retrieved (for citations).
        """
        # Line 20:  Call the retriever to get top-K documents for this query.
        results = self.retriever.retrieve(query)

        # Line 22:  If no documents met the relevance threshold, return empty context.
        if not results:
            # Line 24:  No relevant docs found — LM will get no context and should say "I don't know".
            return "", []

        # Line 27:  Build the augmented context by concatenating retrieved documents with headers.
        context_parts = []
        # Line 28:  Iterate over each retrieved document result.
        for doc in results:
            # Line 30:  Format each doc as "[doc_id] content" so the LM can reference it.
            context_parts.append(
                f"[{doc['id']} (relevance: {doc['score']})]\n{doc['text']}"
            )

        # Line 33:  Join all document parts with a separator for the final context string.
        augmented_context = "\n\n---\n\n".join(context_parts)

        # Line 35:  Return both the context string and the metadata list (for citations).
        return augmented_context, results

    # Line 38:  Utility: get corpus statistics for diagnostic output.
    def stats(self) -> dict:
        # Line 40:  Build a dictionary with corpus metadata.
        return {
            "total_documents": self.corpus.count(),  # Line 42:  How many docs are loaded.
            "document_ids": self.corpus.list_documents(),  # Line 43:  Their filenames.
        }
