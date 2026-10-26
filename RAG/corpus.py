# Line 1:  Import os module for file path operations and directory listing.
import os


# Line 3:  Define the CorpusLoader class that manages a collection of text documents.
class CorpusLoader:
    # Line 5:  Constructor accepts an optional corpus directory path.
    def __init__(self, corpus_dir: str = None):
        # Line 7:  If no directory given, default to ./corpus relative to this file's parent.
        if corpus_dir is None:
            # Line 9:  Compute the absolute path to the corpus/ folder (sibling of rag/).
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            # Line 10:  Join with "corpus" to get the full path.
            corpus_dir = os.path.join(base, "corpus")
        # Line 12:  Store the resolved corpus directory path.
        self.corpus_dir = corpus_dir
        # Line 13:  Dictionary mapping document IDs (filenames) to their text content.
        self._documents: dict[str, str] = {}
        # Line 14:  Load all documents from disk into memory.
        self._load()

    # Line 17:  Load all .txt files from the corpus directory into the internal store.
    def _load(self) -> None:
        # Line 19:  Check if the corpus directory actually exists on disk.
        if not os.path.isdir(self.corpus_dir):
            # Line 21:  If missing, leave the store empty (graceful degradation for tests).
            return
        # Line 23:  List all files in the corpus directory.
        for filename in sorted(os.listdir(self.corpus_dir)):
            # Line 25:  Only process .txt files (skip hidden files, subdirs, etc.).
            if not filename.endswith(".txt"):
                # Line 27:  Skip non-text files.
                continue
            # Line 29:  Build the full file path.
            filepath = os.path.join(self.corpus_dir, filename)
            # Line 30:  Read the file content as UTF-8 text.
            with open(filepath, "r", encoding="utf-8") as f:
                # Line 32:  Store the content keyed by filename (the document ID).
                self._documents[filename] = f.read().strip()

    # Line 35:  Return a sorted list of all document IDs (filenames) in the corpus.
    def list_documents(self) -> list[str]:
        # Line 37:  Return sorted keys of the internal documents dictionary.
        return sorted(self._documents.keys())

    # Line 40:  Retrieve the full text content of a document by its ID (filename).
    def get_document(self, doc_id: str) -> str | None:
        # Line 42:  Look up the document; return None if it doesn't exist.
        return self._documents.get(doc_id)

    # Line 45:  Return all documents as a list of (id, text) tuples for batch operations.
    def get_all(self) -> list[tuple[str, str]]:
        # Line 47:  Convert the dictionary items to a sorted list of tuples.
        return sorted(self._documents.items())

    # Line 50:  Return the total number of documents loaded (useful for diagnostics).
    def count(self) -> int:
        # Line 52:  Simply return the length of the documents dictionary.
        return len(self._documents)
