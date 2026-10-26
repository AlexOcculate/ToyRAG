# Line 1:  Import re module for pattern-based input validation.
import re

# Line 3:  Define constants for guardrail limits (tunable without changing logic).
MAX_INPUT_LENGTH = 500  # Line 4:  Maximum characters allowed in a single user input.
MAX_ITERATIONS = 10  # Line 5:  Maximum agent loop iterations before forced stop.
FORBIDDEN_PATTERNS = (
    [  # Line 6:  List of regex patterns that indicate hostile/malicious input.
        r"rm\s+-rf",  # Line 7:  Block shell destructive commands.
        r";\s*DROP\s+TABLE",  # Line 8:  Block SQL injection attempts.
        r"\bimport\s+os\b",  # Line 9:  Block attempts to inject Python imports.
        r"__import__",  # Line 10: Block dynamic import tricks.
    ]
)


# Line 14:  Define the Guardrails class that validates inputs and tracks loop state.
class Guardrails:
    # Line 16:  Constructor initializes iteration counter and a list of blocked events.
    def __init__(self, max_iterations: int = MAX_ITERATIONS):
        # Line 18:  Store the maximum allowed iterations for this agent session.
        self.max_iterations = max_iterations
        # Line 19:  Track how many loop iterations have been consumed so far.
        self.iteration_count = 0
        # Line 20:  Log of all blocked/rejected inputs for audit purposes.
        self.blocked_events: list[str] = []

    # Line 23:  Validate a user input string; returns (is_valid, reason).
    def validate_input(self, text: str) -> tuple[bool, str]:
        # Line 25:  Strip whitespace to get the effective content length.
        stripped = text.strip()
        # Line 26:  Check if input exceeds the maximum allowed length.
        if len(stripped) > MAX_INPUT_LENGTH:
            # Line 28:  Record this block event with a descriptive reason.
            self.blocked_events.append(
                f"Length exceeded: {len(stripped)} > {MAX_INPUT_LENGTH}"
            )
            # Line 29:  Return invalid with the specific reason for logging.
            return False, f"Input too long ({len(stripped)}/{MAX_INPUT_LENGTH})"
        # Line 31:  Check if input is empty after stripping (reject blank submissions).
        if len(stripped) == 0:
            # Line 33:  Record the empty-input block event.
            self.blocked_events.append("Empty input rejected")
            # Line 34:  Return invalid with reason "empty".
            return False, "Input is empty"
        # Line 36:  Iterate over all forbidden patterns to check for matches.
        for pattern in FORBIDDEN_PATTERNS:
            # Line 38:  Use case-insensitive regex search to find the pattern in input.
            if re.search(pattern, stripped, re.IGNORECASE):
                # Line 40:  Record which pattern was triggered for the audit log.
                self.blocked_events.append(f"Forbidden pattern matched: {pattern}")
                # Line 41:  Return invalid with the specific pattern that was blocked.
                return False, f"Blocked: matches forbidden pattern '{pattern}'"
        # Line 43:  All checks passed; input is valid.
        return True, "OK"

    # Line 46:  Check if the agent has exceeded its iteration budget.
    def check_iterations(self) -> bool:
        # Line 48:  Increment the iteration counter for this call.
        self.iteration_count += 1
        # Line 49:  Return True if we are still within the allowed budget.
        return self.iteration_count <= self.max_iterations

    # Line 52:  Reset the iteration counter (called at start of each new user turn).
    def reset_iterations(self) -> None:
        # Line 54:  Zero out the counter so a new query gets a fresh budget.
        self.iteration_count = 0
