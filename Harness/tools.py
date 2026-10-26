# Line 1:  Import Memory class from sibling module for tool use.
from Harness.memory import Memory


# Line 3:  Define the Tools registry that maps tool names to callable functions.
class Tools:
    # Line 5:  Constructor receives a Memory instance so tools can read/write it.
    def __init__(self, memory: Memory):
        # Line 7:  Store the shared memory reference for use by tool functions.
        self._memory = memory
        # Line 8:  Build the dispatch dictionary mapping tool names to methods.
        self._registry: dict[str, callable] = {
            "memory_store": self._tool_memory_store,  # Line 10:  Store a fact in agent memory.
            "memory_recall": self._tool_memory_recall,  # Line 11:  Retrieve a previously stored fact.
            "list_corpus": self._tool_list_corpus,  # Line 12:  List available corpus documents.
        }

    # Line 15:  Execute a tool by name with given arguments; returns (success, result_string).
    def execute(self, tool_name: str, args: dict) -> tuple[bool, str]:
        # Line 17:  Look up the tool in the registry.
        handler = self._registry.get(tool_name)
        # Line 18:  If tool not found, return a graceful error (don't crash).
        if handler is None:
            # Line 20:  Return failure with a helpful message listing available tools.
            return (
                False,
                f"Unknown tool '{tool_name}'. Available: {list(self._registry.keys())}",
            )
        # Line 22:  Wrap execution in try/except so a buggy tool can't kill the agent loop.
        try:
            # Line 24:  Call the tool handler with its arguments dictionary.
            result = handler(args)
            # Line 25:  Return success and the string result from the tool.
            return True, result
        except Exception as e:
            # Line 27:  Catch any exception and return it as a failure (graceful degradation).
            return False, f"Tool '{tool_name}' raised: {type(e).__name__}: {e}"

    # Line 31:  Tool implementation: store a key-value pair in memory.
    def _tool_memory_store(self, args: dict) -> str:
        # Line 33:  Extract 'key' and 'value' from the arguments dictionary.
        key = args.get("key", "")
        # Line 34:  If no key provided, return an error message.
        if not key:
            # Line 36:  Signal that required argument is missing.
            return "ERROR: memory_store requires 'key' and 'value' arguments."
        # Line 38:  Store the value under the given key in the shared memory.
        self._memory.store(key, args.get("value", ""))
        # Line 39:  Return a confirmation message (the agent will see this as tool output).
        return f"Stored '{key}' in memory."

    # Line 42:  Tool implementation: recall a value from memory by key.
    def _tool_memory_recall(self, args: dict) -> str:
        # Line 44:  Extract the 'key' argument for lookup.
        key = args.get("key", "")
        # Line 45:  If no key provided, return an error message.
        if not key:
            # Line 47:  Signal that required argument is missing.
            return "ERROR: memory_recall requires a 'key' argument."
        # Line 49:  Attempt to recall the value from memory.
        value = self._memory.recall(key)
        # Line 50:  If the key was not found, return a not-found message.
        if value is None:
            # Line 52:  Inform the agent that this key has no stored value.
            return f"No memory found for key '{key}'."
        # Line 54:  Return the recalled value as the tool result.
        return value

    # Line 57:  Tool implementation: list all corpus documents (delegates to corpus module).
    def _tool_list_corpus(self, args: dict) -> str:
        # Line 59:  Import CorpusLoader here to avoid circular import at module load time.
        from rag.corpus import CorpusLoader

        # Line 60:  Create a loader pointing to the default corpus directory.
        loader = CorpusLoader()
        # Line 61:  Get the list of document filenames available in the corpus.
        docs = loader.list_documents()
        # Line 62:  Return a formatted string listing all available documents.
        return f"Available documents ({len(docs)}): {', '.join(docs)}"

    # Line 65:  Return the list of registered tool names (used for LM prompt building).
    def get_tool_names(self) -> list[str]:
        # Line 67:  Return keys of the registry as a sorted list.
        return sorted(self._registry.keys())
