# Line 1:  Import re for parsing tool call patterns from LM output.
import re


# Line 3:  Define the SimulatedLM class that mimics an LLM using keyword rules.
class SimulatedLM:
    # Line 5:  Constructor accepts a name label for logging purposes.
    def __init__(self, name: str = "SimLM-v1"):
        # Line 7:  Store the model name (used in trace logs to identify which LM responded).
        self.name = name

    # Line 10:  Main generation method: takes a prompt and optional context, returns text.
    def generate(self, prompt: str, context: str = "") -> str:
        # Line 12:  Normalize the prompt to lowercase for keyword matching.
        lower = prompt.lower()

        # ─── Rule 1: Memory store request ───────────────────────────────
        # Line 15:  If user asks to store/remember something, emit a tool call.
        if any(kw in lower for kw in ["store", "remember", "save"]):
            # Line 17:  Extract the key-value pair using a simple pattern.
            match = re.search(
                r"(?:store|remember|save)\s+(?:the\s+)?(\w+)\s*[:=]?\s*(.+)", lower
            )
            # Line 18:  If we could parse key and value, format the tool call.
            if match:
                # Line 20:  Build the TOOL_CALL string that the agent loop will parse.
                return f"TOOL_CALL: memory_store(key={match.group(1)}, value={match.group(2).strip()})"
            # Line 22:  If parsing failed, ask for clarification (simulates LM uncertainty).
            return (
                "I'd like to store that for you. Please format as: store <key> <value>"
            )

        # ─── Rule 2: Memory recall request ──────────────────────────────
        # Line 25:  If user asks to recall/what did I store, emit a tool call.
        if any(kw in lower for kw in ["recall", "what did i store", "retrieve memory"]):
            # Line 27:  Try to extract the key they want to recall.
            match = re.search(r"(?:recall|retrieve)\s+(?:the\s+)?(\w+)", lower)
            # Line 28:  If key found, emit the tool call.
            if match:
                # Line 30:  Format the TOOL_CALL for memory_recall.
                return f"TOOL_CALL: memory_recall(key={match.group(1)})"
            # Line 32:  If no key specified, ask which one they want.
            return "Which memory would you like to recall? Please specify the key."

        # ─── Rule 3: List corpus request ────────────────────────────────
        # Line 35:  If user asks to list/show documents, emit a tool call.
        if any(kw in lower for kw in ["list", "show documents", "what documents"]):
            # Line 37:  Emit the TOOL_CALL for listing corpus documents.
            return "TOOL_CALL: list_corpus()"

        # ─── Rule 4: Grounded answer from RAG context ───────────────────
        # Line 40:  If we have retrieved context, generate a grounded answer.
        if context and context.strip():
            # Line 42:  Extract the most relevant sentences from the context.
            relevant = self._extract_relevant(context, lower)
            # Line 43:  If we found relevant content, compose a grounded response.
            if relevant:
                # Line 45:  Build the answer with attribution to the knowledge base.
                return f"Based on my knowledge base:\n{relevant}\n\n[Grounded in retrieved context]"
            # Line 47:  Context exists but nothing matched the query specifically.
            return "I have some context but couldn't find specific information about that topic."

        # ─── Rule 5: No context, no match — honest refusal ──────────────
        # Line 50:  If we reach here, no rules matched and no useful context exists.
        return "I don't have enough information to answer that question. Try asking about Python, databases, or networking."

    # Line 54:  Helper: extract sentences from context that share keywords with the query.
    def _extract_relevant(self, context: str, query_lower: str) -> str:
        # Line 56:  Split the context into individual sentences (simple period split).
        sentences = [s.strip() for s in context.split(". ") if s.strip()]
        # Line 57:  Extract meaningful words from the query (length > 3 to skip articles).
        query_words = {w for w in query_lower.split() if len(w) > 3}
        # Line 58:  Collect sentences that share at least one keyword with the query.
        matched = []
        # Line 60:  Iterate over each sentence in the context.
        for sent in sentences:
            # Line 62:  Lowercase the sentence for comparison.
            sent_lower = sent.lower()
            # Line 63:  Count how many query words appear in this sentence.
            overlap = sum(1 for w in query_words if w in sent_lower)
            # Line 64:  If there's at least one matching word, include this sentence.
            if overlap > 0:
                # Line 66:  Append the matched sentence to our results list.
                matched.append(sent + ".")
        # Line 68:  Join all matched sentences into a single string (or empty if none).
        return " ".join(matched) if matched else ""
