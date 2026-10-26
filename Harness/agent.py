# Line 1:  Import SimulatedLM for the brain.
from Harness.lm import SimulatedLM

# Line 2:  Import Tools registry for tool execution.
from Harness.tools import Tools

# Line 3:  Import Memory for persistent agent state.
from Harness.memory import Memory

# Line 4:  Import Guardrails for input validation and iteration limits.
from Harness.guardrails import Guardrails

# Line 5:  Import RAGPipeline — the new component that grounds answers.
from RAG.pipeline import RAGPipeline

# Line 6:  Import TraceLogger for colored execution tracing.
import sys, os

# Line 7:  Add parent directory to path so trace_logger is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Line 8:  Import the trace logger module.
from trace_logger import TraceLogger


# Line 11:  Define the Agent class that runs the main loop (same structure as #0).
class Agent:
    # Line 13:  Constructor wires up all components: LM, tools, memory, guardrails, RAG.
    def __init__(self, lm: SimulatedLM = None, rag_pipeline: RAGPipeline = None):
        # Line 15:  Create or accept the simulated language model (the "brain").
        self.lm = lm or SimulatedLM()
        # Line 16:  Create a fresh memory store for this agent instance.
        self.memory = Memory()
        # Line 17:  Create the tools registry, passing in the shared memory.
        self.tools = Tools(self.memory)
        # Line 18:  Create guardrails with default limits (10 max iterations).
        self.guardrails = Guardrails()
        # Line 19:  Create or accept the RAG pipeline (retrieval + augmentation).
        self.rag = rag_pipeline or RAGPipeline()
        # Line 20:  Create the trace logger for this agent session.
        self.trace = TraceLogger(module_name="ToyRAG")

    # Line 23:  Main entry point: process a single user query through the full loop.
    def run(self, user_input: str) -> str:
        # Line 25:  Log the start of a new agent turn with the user's input.
        self.trace.log("loop", f"USER INPUT: {user_input}")

        # ─── Step 1: Guardrails validation ──────────────────────────────
        # Line 28:  Validate the input against length, emptiness, and forbidden patterns.
        valid, reason = self.guardrails.validate_input(user_input)
        # Line 29:  If input failed validation, log and return a rejection message.
        if not valid:
            # Line 31:  Log the guardrail block in red (visible in trace).
            self.trace.log("guardrails", f"BLOCKED: {reason}")
            # Line 32:  Return a user-facing error message (agent does not proceed).
            return f"[REJECTED] {reason}"

        # ─── Step 2: Reset iteration counter for this turn ──────────────
        # Line 35:  Each new user query gets a fresh iteration budget.
        self.guardrails.reset_iterations()

        # ─── Step 3: Agent loop (same structure as module #0) ───────────
        # Line 38:  Accumulate conversation context across iterations within this turn.
        conversation = user_input
        # Line 39:  Track citations collected during this turn for final output.
        citations: list[str] = []

        # Line 41:  Begin the main agent loop (bounded by guardrails).
        while True:
            # Line 43:  Check if we've exceeded the iteration budget.
            if not self.guardrails.check_iterations():
                # Line 45:  Log that we hit the max iterations and are stopping.
                self.trace.log(
                    "guardrails",
                    f"STOPPED: Max iterations ({self.guardrails.max_iterations}) reached.",
                )
                # Line 46:  Return a timeout message to the user.
                return "[TIMEOUT] Agent exceeded maximum iterations."

            # ─── RAG Step: Retrieve relevant documents ──────────────────
            # Line 49:  Call the RAG pipeline to retrieve docs and build augmented prompt.
            augmented_prompt, retrieved_docs = self.rag.retrieve_and_augment(
                conversation
            )
            # Line 50:  Log which documents were retrieved (magenta for RAG component).
            doc_names = [d["id"] for d in retrieved_docs]
            # Line 51:  Record the retrieval step in the trace.
            self.trace.log("rag", f"RETRIEVED {len(retrieved_docs)} docs: {doc_names}")

            # ─── LM Step: Generate response with augmented context ──────
            # Line 54:  Build the full prompt for the LM (user query + tool descriptions).
            prompt = self._build_prompt(conversation)
            # Line 55:  Call the LM with the prompt and the retrieved context.
            response = self.lm.generate(prompt, context=augmented_prompt)
            # Line 56:  Log the raw LM output (cyan for LM component).
            self.trace.log(
                "lm",
                f"RESPONSE: {response[:120]}{'...' if len(response) > 120 else ''}",
            )

            # ─── Parse Step: Determine if this is a tool call or final answer ──
            # Line 59:  Check if the LM output starts with "TOOL_CALL:" prefix.
            if response.startswith("TOOL_CALL:"):
                # Line 61:  Extract the tool name and arguments from the TOOL_CALL string.
                tool_name, args = self._parse_tool_call(response)
                # Line 62:  Log the tool call decision (yellow for tools component).
                self.trace.log("tools", f"CALLING: {tool_name}({args})")

                # ─── Execute Step: Run the tool ─────────────────────────
                # Line 65:  Execute the tool through the registry (handles errors internally).
                success, result = self.tools.execute(tool_name, args)
                # Line 66:  Log the tool execution result.
                self.trace.log(
                    "tools", f"RESULT ({'OK' if success else 'FAIL'}): {result[:80]}"
                )

                # ─── Feed tool result back into conversation ────────────
                # Line 69:  Append the tool result to the conversation so next iteration sees it.
                conversation = (
                    f"{conversation}\n[Tool result from {tool_name}]: {result}"
                )
                # Line 70:  Continue the loop (LM will now see the tool output).
                continue

            # ─── Final Answer: No tool call, so this is the response ────
            # Line 73:  Collect citations from the retrieved documents.
            for doc in retrieved_docs:
                # Line 75:  Add each document ID to the citations list (deduplicated).
                if doc["id"] not in citations:
                    # Line 77:  Append the citation source.
                    citations.append(doc["id"])

            # Line 79:  Log that we're returning a final answer.
            self.trace.log("loop", f"FINAL ANSWER (citations: {citations})")
            # Line 80:  Format and return the final answer with citations appended.
            answer = response
            # Line 81:  If there are citations, append them as a source line.
            if citations:
                # Line 83:  Build the citation footer string.
                answer += f"\n\n[Citations: {', '.join(citations)}]"
            # Line 84:  Return the complete grounded answer to the caller.
            return answer

    # Line 87:  Build the full prompt sent to the LM (includes tool descriptions).
    def _build_prompt(self, conversation: str) -> str:
        # Line 89:  Get the list of available tools for the system prompt section.
        tool_names = self.tools.get_tool_names()
        # Line 90:  Build a system instruction block describing available tools.
        system = f"Available tools: {', '.join(tool_names)}. Use TOOL_CALL: name(args) to invoke."
        # Line 91:  Combine system instructions with the conversation history.
        return f"{system}\n\n{conversation}"

    # Line 94:  Parse a "TOOL_CALL: name(key=value, ...)" string into (name, args_dict).
    def _parse_tool_call(self, response: str) -> tuple[str, dict]:
        # Line 96:  Strip the "TOOL_CALL:" prefix to get the raw call string.
        raw = response[len("TOOL_CALL:") :]
        # Line 97:  Use regex to extract tool name (before first paren).
        import re

        # Line 98:  Match pattern: tool_name(arg1=val1, arg2=val2)
        match = re.match(r"(\w+)\((.*)\)", raw.strip())
        # Line 99:  If no match (malformed), return empty name and empty args.
        if not match:
            # Line 101:  Return a placeholder that will trigger an "unknown tool" error gracefully.
            return "", {}
        # Line 103:  Extract the tool name from group 1.
        tool_name = match.group(1)
        # Line 104:  Parse the arguments string into a dictionary.
        args_str = match.group(2)
        # Line 105:  Initialize an empty dictionary for parsed arguments.
        args: dict[str, str] = {}
        # Line 106:  If there are arguments to parse (non-empty string inside parens).
        if args_str.strip():
            # Line 108:  Split on commas to get individual key=value pairs.
            pairs = args_str.split(",")
            # Line 109:  Iterate over each pair and parse into the args dict.
            for pair in pairs:
                # Line 111:  Split each pair on the first '=' to get key and value.
                if "=" in pair:
                    # Line 113:  Separate key and value, stripping whitespace.
                    k, v = pair.split("=", 1)
                    # Line 114:  Store in args dict with stripped values.
                    args[k.strip()] = v.strip()
        # Line 116:  Return the parsed tool name and arguments dictionary.
        return tool_name, args
