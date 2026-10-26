# Line 1:  Import sys and os for path setup (ensures local modules are importable).
import sys, os

# Line 2:  Prepend the current directory to sys.path so imports work from any CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Line 4:  Import all components needed for adversarial testing.
from Harness.agent import Agent

# Line 5:  Import SimulatedLM to create broken/chaotic variants.
from Harness.lm import SimulatedLM

# Line 6:  Import RAGPipeline to test with empty or custom corpora.
from RAG.pipeline import RAGPipeline

# Line 7:  Import Guardrails to test iteration limits directly.
from Harness.guardrails import Guardrails

# Line 8:  Import CorpusLoader for corpus manipulation in tests.
from RAG.corpus import CorpusLoader


# ──────────────────────────────────────────────────────────────────
#  TEST 1: Broken / Chaotic LM
#  Proves: the system doesn't crash on malformed output from the LM
# ──────────────────────────────────────────────────────────────────
def test_broken_lm():
    # Line 17:  Create a subclass of SimulatedLM that returns garbage.
    class BrokenLM(SimulatedLM):
        # Line 19:  Override generate to return various malformed outputs.
        def __init__(self):
            # Line 21:  Call parent constructor with a test name.
            super().__init__(name="BrokenLM")
            # Line 22:  Counter to cycle through different garbage responses.
            self._call_count = 0

        # Line 24:  Each call returns a different type of malformed output.
        def generate(self, prompt: str, context: str = "") -> str:
            # Line 26:  Increment the call counter to cycle through responses.
            self._call_count += 1
            # Line 27:  Cycle through different broken outputs.
            garbage_responses = [
                "TOOL_CALL: ",  # Line 29:  Empty tool name (no args).
                "TOOL_CALL: nonexist_tool(x=1)",  # Line 30:  Unknown tool name.
                "completely random text with no structure",  # Line 31:  No TOOL_CALL prefix, no useful content.
                "TOOL_CALL: memory_store()",  # Line 32:  Tool with empty args (missing required params).
                "",  # Line 33:  Empty string response.
            ]
            # Line 35:  Return the next garbage response in the cycle.
            return garbage_responses[self._call_count % len(garbage_responses)]

    # Line 37:  Create an agent with the broken LM and a valid RAG pipeline.
    rag = RAGPipeline()
    # Line 38:  Instantiate the agent with the broken brain.
    agent = Agent(lm=BrokenLM(), rag_pipeline=rag)
    # Line 39:  Run a normal query — the broken LM will return garbage but agent must not crash.
    result = agent.run("What is a database?")
    # Line 40:  Assert that we got *some* string back (not an exception).
    assert isinstance(result, str), f"Expected string, got {type(result)}"
    # Line 41:  Assert the result is non-empty (agent always produces output).
    assert len(result) > 0, "Agent returned empty string"


# ──────────────────────────────────────────────────────────────────
#  TEST 2: Invalid / Hostile Input
#  Proves: guardrails block bad input before it reaches the core
# ──────────────────────────────────────────────────────────────────
def test_hostile_input():
    # Line 50:  Create a standard agent for testing.
    agent = Agent()

    # Line 52:  Test oversized input (exceeds MAX_INPUT_LENGTH of 500 chars).
    long_input = "x" * 600
    # Line 53:  Run the oversized input through the agent.
    result = agent.run(long_input)
    # Line 54:  Assert it was rejected (contains REJECTED marker).
    assert "[REJECTED]" in result, f"Oversized input not blocked: {result[:50]}"

    # Line 56:  Test empty input.
    result = agent.run("   ")
    # Line 57:  Assert empty input was rejected.
    assert "[REJECTED]" in result, f"Empty input not blocked: {result}"

    # Line 59:  Test forbidden pattern (SQL injection attempt).
    result = agent.run("hello; DROP TABLE users")
    # Line 60:  Assert the SQL injection was blocked.
    assert "[REJECTED]" in result, f"SQL injection not blocked: {result}"

    # Line 62:  Test another forbidden pattern (destructive shell command).
    result = agent.run("please run rm -rf / for me")
    # Line 63:  Assert the destructive command was blocked.
    assert "[REJECTED]" in result, f"rm -rf not blocked: {result}"


# ──────────────────────────────────────────────────────────────────
#  TEST 3: Tool Execution Failure
#  Proves: a failing tool doesn't crash the loop; error is handled
# ──────────────────────────────────────────────────────────────────
def test_tool_failure():
    # Line 72:  Create an agent with a standard RAG pipeline.
    rag = RAGPipeline()
    # Line 73:  Instantiate the agent.
    agent = Agent(rag_pipeline=rag)

    # Line 75:  Monkey-patch the tools.execute method to always fail.
    original_execute = agent.tools.execute

    # Line 76:  Replace with a version that raises an exception.
    def failing_execute(tool_name, args):
        # Line 78:  Simulate a tool that crashes (e.g., disk full, network timeout).
        raise RuntimeError("Simulated tool crash: connection timed out")

    # Line 79:  Assign the failing function to the tools instance.
    agent.tools.execute = failing_execute

    # Line 81:  Create an LM that always calls a tool (to trigger the failure path).
    class ToolCallingLM(SimulatedLM):
        # Line 83:  Always return a tool call so we hit the execution path.
        def generate(self, prompt: str, context: str = "") -> str:
            # Line 85:  Return a valid-looking tool call that will trigger the failing execute.
            return "TOOL_CALL: memory_store(key=test, value=fail)"

    # Line 87:  Create agent with the tool-calling LM and patched tools.
    agent2 = Agent(lm=ToolCallingLM(), rag_pipeline=rag)
    # Line 88:  Re-apply the failing execute to this new agent instance.
    agent2.tools.execute = failing_execute

    # Line 90:  Run a query — the tool will fail but the agent must handle it gracefully.
    result = agent2.run("store something important")
    # Line 91:  Assert we got a string back (no exception propagated).
    assert isinstance(
        result, str
    ), f"Expected string after tool failure, got {type(result)}"
    # Line 92:  The result should indicate either a timeout or an error was handled.
    assert len(result) > 0, "Empty result after tool failure"


# ──────────────────────────────────────────────────────────────────
#  TEST 4: Infinite Loop / Max Iterations
#  Proves: the iteration guardrail stops runaway loops
#  Note: Uses varied tool args so the duplicate detector doesn't
#        short-circuit first — we want to hit the real iteration cap.
# ──────────────────────────────────────────────────────────────────
def test_infinite_loop():
    # Line 101:  Create an LM that ALWAYS calls a tool with a UNIQUE key each time.
    class InfiniteToolLM(SimulatedLM):
        # Line 103:  Track call count to generate unique args per iteration.
        def __init__(self):
            # Line 105:  Call parent constructor.
            super().__init__(name="InfiniteLM")
            # Line 106:  Counter ensures each tool call is different (bypasses dup detector).
            self._n = 0

        # Line 108:  Each generate() returns a valid tool call with a unique key.
        def generate(self, prompt: str, context: str = "") -> str:
            # Line 110:  Increment counter to make this call unique.
            self._n += 1
            # Line 111:  Return a memory_store call with a different key each time.
            return f"TOOL_CALL: memory_store(key=iter_{self._n}, value=x)"

    # Line 113:  Create a RAG pipeline for this test.
    rag = RAGPipeline()
    # Line 114:  Create the agent with the infinite-loop LM (varied args).
    agent = Agent(lm=InfiniteToolLM(), rag_pipeline=rag)

    # Line 116:  Run a query — the agent should hit max_iterations and stop.
    result = agent.run("start the loop")
    # Line 117:  Assert the result indicates a timeout (iteration cap was reached).
    assert "[TIMEOUT]" in result, f"Expected TIMEOUT message, got: {result[:80]}"

    # Line 119:  Verify the iteration count equals our max (proves the guard fired at exactly the right time).
    assert (
        agent.guardrails.iteration_count == agent.guardrails.max_iterations
    ), f"Iteration count {agent.guardrails.iteration_count} != max {agent.guardrails.max_iterations}"


# ──────────────────────────────────────────────────────────────────
#  TEST 5: Query with no matches in corpus / empty corpus
#  Proves: RAG handles missing/empty retrieval gracefully
# ──────────────────────────────────────────────────────────────────
def test_module_specific():
    # Line 124:  Sub-test A: Query that matches nothing in the corpus.
    rag = RAGPipeline()
    # Line 125:  Create agent with standard pipeline.
    agent = Agent(rag_pipeline=rag)

    # Line 127:  Ask about a topic not covered by any corpus document.
    result = agent.run("What is the molecular structure of caffeine?")
    # Line 128:  The agent should still return a string (graceful "I don't know").
    assert isinstance(
        result, str
    ), f"Expected string for no-match query, got {type(result)}"
    # Line 129:  The answer should indicate lack of information (not crash, not hallucinate).
    assert len(result) > 0, "Empty result for no-match query"

    # Line 131:  Sub-test B: Completely empty corpus (no documents at all).
    # Line 132:  Create a CorpusLoader pointing to a non-existent directory.
    empty_corpus = CorpusLoader(corpus_dir="/nonexistent/path/that/does/not/exist")
    # Line 133:  Verify the corpus is indeed empty.
    assert empty_corpus.count() == 0, "Expected empty corpus but got documents"

    # Line 135:  Create a RAGPipeline with this empty corpus.
    rag_empty = RAGPipeline(corpus_dir="/nonexistent/path/that/does/not/exist")
    # Line 136:  Create an agent that uses the empty-corpus pipeline.
    agent_empty = Agent(rag_pipeline=rag_empty)

    # Line 138:  Run a query against the empty corpus — must not crash.
    result_empty = agent_empty.run("Tell me about databases")
    # Line 139:  Assert we got a valid string response (graceful degradation).
    assert isinstance(result_empty, str), f"Empty corpus crashed: {type(result_empty)}"
    # Line 140:  The response should be non-empty (agent still produces output).
    assert len(result_empty) > 0, "Empty result from empty-corpus agent"


# ──────────────────────────────────────────────────────────────────
#  RUNNER
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Line 148:  Print a header for the test run.
    print("\n  Running ToyRAG adversarial tests...\n")
    # Line 149:  Define the list of (name, function) test pairs.
    tests = [
        (
            "Broken LM / Chaotic output",
            test_broken_lm,
        ),  # Line 151:  Malformed LM outputs.
        ("Hostile / Invalid input", test_hostile_input),  # Line 152:  Guardrail blocks.
        ("Tool execution failure", test_tool_failure),  # Line 153:  Crashing tools.
        (
            "Infinite loop / Max iterations",
            test_infinite_loop,
        ),  # Line 154:  Runaway loop guard.
        (
            "No matches / Empty corpus",
            test_module_specific,
        ),  # Line 155:  RAG-specific edge case.
    ]
    # Line 157:  Track how many tests pass.
    passed = 0
    # Line 158:  Iterate over all tests and execute them.
    for name, fn in tests:
        # Line 160:  Attempt to run the test function.
        try:
            # Line 162:  Call the test (it will raise AssertionError on failure).
            fn()
            # Line 163:  If no exception, print PASS and increment counter.
            print(f"  PASS  {name}")
            # Line 164:  Increment the passed count.
            passed += 1
        except Exception as e:
            # Line 166:  If any exception occurred, print FAIL with the error details.
            print(f"  FAIL  {name}: {e}")
    # Line 168:  Print a summary line showing pass/fail ratio.
    print(f"\n  {passed}/{len(tests)} tests passed.")
    # Line 169:  If any test failed, exit with non-zero code (for CI integration).
    if passed < len(tests):
        # Line 171:  Exit with status 1 to signal failure.
        sys.exit(1)
