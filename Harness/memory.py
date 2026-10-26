# Line 1:  Import dataclass decorator to define structured memory entries.
from dataclasses import dataclass, field

# Line 2:  Import time module to timestamp each memory entry.
import time


# Line 4:  Define a dataclass representing a single stored memory item.
@dataclass
class MemoryEntry:
    # Line 6:  The key under which this memory is stored (user-chosen identifier).
    key: str
    # Line 7:  The value/content of the memory (free-form text).
    value: str
    # Line 8:  Unix timestamp of when this entry was created.
    timestamp: float = field(default_factory=time.time)


# Line 11:  Define the Memory class that provides key-value storage for the agent.
class Memory:
    # Line 13:  Constructor initializes an empty dictionary to hold entries.
    def __init__(self):
        # Line 15:  Internal dict maps string keys to MemoryEntry objects.
        self._store: dict[str, MemoryEntry] = {}

    # Line 18:  Store a value under a given key; overwrites if key already exists.
    def store(self, key: str, value: str) -> None:
        # Line 20:  Create a new MemoryEntry with the current timestamp and insert it.
        self._store[key] = MemoryEntry(key=key, value=value)

    # Line 23:  Retrieve a value by key; returns None if the key does not exist.
    def recall(self, key: str) -> str | None:
        # Line 25:  Look up the entry and return its value, or None if absent.
        entry = self._store.get(key)
        # Line 27:  Return the value if found, otherwise None to signal "not stored".
        return entry.value if entry else None

    # Line 30:  List all stored keys for introspection (used by list_memory tool).
    def list_keys(self) -> list[str]:
        # Line 32:  Return a sorted list of all keys currently in memory.
        return sorted(self._store.keys())

    # Line 35:  Return the total number of stored entries (useful for debugging).
    def count(self) -> int:
        # Line 37:  Simply return the length of the internal dictionary.
        return len(self._store)
