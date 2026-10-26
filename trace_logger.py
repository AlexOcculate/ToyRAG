# Line 1:  ANSI escape codes for terminal colors (no external packages needed).
class _C:
    # Line 3:  Reset all formatting back to default.
    RESET = "\033[0m"
    # Line 5:  Bold text modifier.
    BOLD = "\033[1m"
    # ─── Foreground colors ─────────────────────────────────────────────
    RED = "\033[91m"  # Line 8:  Bright red for guardrails/blocks.
    GREEN = "\033[92m"  # Line 9:  Bright green for loop/agent flow.
    YELLOW = "\033[93m"  # Line 10: Bright yellow for tool execution.
    BLUE = "\033[94m"  # Line 11: Bright blue for corpus operations.
    MAGENTA = "\033[95m"  # Line 12: Bright magenta for RAG/retrieval.
    CYAN = "\033[96m"  # Line 13: Bright cyan for LM responses.
    WHITE = "\033[97m"  # Line 14: White for general info.
    # ─── Background colors (for status badges) ─────────────────────────
    BG_LIGHT_GREEN = "\033[102m"  # Line 16: Light green background for "valid=True".
    BG_LIGHT_RED = "\033[101m"  # Line 17: Light red background for "BLOCKED"/errors.
    BG_LIGHT_YELLOW = (
        "\033[103m"  # Line 18: Light yellow background for warnings/stops.
    )
    BLACK_FG = "\033[30m"  # Line 19: Black foreground (pairs with light bg).


# Line 22:  Map component names to their assigned ANSI color codes.
COMPONENT_COLORS: dict[str, str] = {
    "guardrails": _C.RED,  # Line 24:  Guardrail events are red (safety/critical).
    "loop": _C.GREEN,  # Line 25:  Agent loop flow is green (healthy operation).
    "lm": _C.CYAN,  # Line 26:  LM responses are cyan (the "brain" speaking).
    "tools": _C.YELLOW,  # Line 27:  Tool execution is yellow (side effects).
    "rag": _C.MAGENTA,  # Line 28:  RAG retrieval is magenta (new in this module).
    "corpus": _C.BLUE,  # Line 29:  Corpus operations are blue (data layer).
}


# Line 32:  Define the TraceLogger class that collects and displays execution traces.
class TraceLogger:
    # Line 34:  Constructor stores the module name for the report header.
    def __init__(self, module_name: str = "ToyRAG"):
        # Line 36:  Store the module name for display in reports.
        self.module_name = module_name
        # Line 37:  Internal list to accumulate all log entries as (component, message) tuples.
        self._entries: list[tuple[str, str]] = []

    # Line 40:  Log a single event with its component tag and message.
    def log(self, component: str, message: str) -> None:
        # Line 42:  Store the entry internally for later report() output.
        self._entries.append((component, message))
        # Line 43:  Also print immediately for real-time terminal feedback.
        self._print_entry(component, message)

    # Line 46:  Print a single log entry with appropriate coloring.
    def _print_entry(self, component: str, message: str) -> None:
        # Line 48:  Look up the color for this component (default to white if unknown).
        color = COMPONENT_COLORS.get(component, _C.WHITE)
        # Line 49:  Format and print the colored log line.
        print(f"  {color}[{component}]{_C.RESET} {message}")

    # Line 52:  Print the full execution report (called after agent.run() completes).
    def report(self) -> None:
        # Line 54:  Print a header line for the report section.
        print(f"\n{_C.BOLD}{'=' * 60}{_C.RESET}")
        # Line 55:  Print the module name in bold.
        print(f"{_C.BOLD}  TRACE REPORT — {self.module_name}{_C.RESET}")
        # Line 56:  Print another separator line.
        print(f"{_C.BOLD}{'=' * 60}{_C.RESET}")
        # Line 57:  Iterate over all collected entries and print them with colors.
        for component, message in self._entries:
            # Line 59:  Get the color for this component.
            color = COMPONENT_COLORS.get(component, _C.WHITE)
            # Line 60:  Print each entry with its colored component tag.
            print(f"  {color}[{component:<12s}]{_C.RESET} {message}")
        # Line 62:  Print a footer with total entry count.
        print(f"{_C.BOLD}{'=' * 60}{_C.RESET}")
        # Line 63:  Show how many trace events were recorded.
        print(f"  Total events: {len(self._entries)}")
        # Line 64:  Print a final separator.
        print(f"{_C.BOLD}{'=' * 60}{_C.RESET}\n")

    # Line 67:  Clear all logged entries (useful between test cases).
    def clear(self) -> None:
        # Line 69:  Empty the internal entries list.
        self._entries.clear()
