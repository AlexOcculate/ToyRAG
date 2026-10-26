# ToyRAG — Retrieve, Augment, Generate (Toy Implementation)

A zero-dependency Python project that teaches how RAG (Retrieval-Augmented Generation) works: search a small document collection, paste the results into the prompt, and produce a grounded answer with citations. No LLM. No GPU. No external packages.

**Prerequisite:** understand [ToyVectorDB](https://github.com/AlexOcculate/TOYAGENTS/tree/main/ToyVectorDB) (Module #6). You should be comfortable with the idea that text can be represented as numbers and compared by similarity. This module uses a simpler keyword-based retrieval, but the concept is the same: find the most relevant documents before answering.

---

## What Is This?

Imagine you are a librarian who also writes answers. When someone asks you a question, you do not answer from memory alone — you first look up the most relevant books on your shelf, read the key passages, and then write an answer that cites those books. RAG does exactly this for an AI agent: it retrieves relevant documents, injects them into the prompt (the "context"), and the model produces an answer grounded in what was actually retrieved.

The critical insight: **the agent loop does not change.** The same loop from Module #0 (validate → think → act → repeat) still runs. What changes is that before each "think" step, we retrieve documents and paste them into the prompt. The quality improvement comes from grounding, not from a smarter model.

Think of it this way: without RAG, the agent is a student taking an open-book exam but forgot to bring the book. With RAG, the agent gets the relevant pages handed to it before each question. Same brain, better results.

---

## Folder Structure

```
ToyRAG/
|-- main.py                  # Entry point: runs 6 demo queries through the agent
|-- trace_logger.py          # Colored execution tracing (ANSI codes, no packages)
|-- test_adversarial.py      # 5 failure-mode tests proving system resilience
|-- README.md                # This file
|
|-- harness/                 # The agent "body" (same structure as Module #0)
|   |-- __init__.py          # Package marker
|   |-- agent.py             # Main agent loop: validate -> RAG retrieve -> LM -> parse
|   |-- lm.py                # SimulatedLM: keyword rules + context-aware grounded answers
|   |-- tools.py             # Tool registry: memory_store, memory_recall, list_corpus
|   |-- guardrails.py        # Input validation + iteration limit + stuck detection
|   |-- memory.py            # Key-value store for agent state across turns
|
|-- rag/                     # The NEW component (the "RAG" in ToyRAG)
|   |-- __init__.py          # Package marker
|   |-- pipeline.py          # RAGPipeline: orchestrates retrieve -> augment -> cite
|   |-- retriever.py         # Keyword-overlap scoring + top-K ranking
|   |-- corpus.py            # CorpusLoader: reads .txt files from disk into memory
|
|-- corpus/                  # The "knowledge base" (3 small text documents)
    |-- doc1_python.txt      # About the Python programming language
    |-- doc2_databases.txt   # About database systems and indexing
    |-- doc3_networking.txt  # About computer networking (TCP, UDP, HTTP, DNS)
```

---

## Dependencies

**External:** None. Zero. This project runs on a bare Python 3.10+ installation.

**Standard library modules used:**

| Module        | Where used                                    | Purpose                                          |
| ------------- | --------------------------------------------- | ------------------------------------------------ |
| `os`          | corpus.py, main.py, test_adversarial.py       | File paths, directory listing                    |
| `re`          | lm.py, retriever.py, guardrails.py, agent.py  | Tokenization, pattern matching, input validation |
| `time`        | memory.py                                     | Timestamping memory entries                      |
| `sys`         | main.py, test_adversarial.py, trace_logger.py | Path setup, exit codes                           |
| `dataclasses` | memory.py                                     | Structured MemoryEntry definition                |

**Internal import graph:**

```
main.py
  |-- harness/agent.py
  |     |-- harness/lm.py
  |     |-- harness/tools.py
  |     |     |-- harness/memory.py
  |     |-- harness/guardrails.py
  |     |-- rag/pipeline.py
  |     |     |-- rag/corpus.py
  |     |     |-- rag/retriever.py
  |     |-- trace_logger.py
  |-- harness/lm.py
  |-- rag/pipeline.py

test_adversarial.py
  |-- (same imports as main.py)
```

No circular dependencies. The `rag/` package is self-contained and only depends on the standard library. The `harness/` package depends on `rag/` (the agent calls the pipeline), but never the reverse.

---

## Execution

```bash
# Navigate to the project folder
cd ToyRAG

# Run the interactive demo (6 queries showing different RAG behaviors)
python main.py

# Run adversarial tests (proves the system survives failure conditions)
python test_adversarial.py
```

No `pip install`. No virtual environment needed. Just Python 3.10 or later.

---

## Expected Output

Running `python main.py` produces output like this (colors omitted for readability):

```
============================================================
  TOYRAG - Retrieve, Augment, Generate (Toy Implementation)
  RAG = search + paste into prompt. Grounded answers with citations.
============================================================

  Corpus: 3 documents loaded
  Documents: doc1_python.txt, doc2_databases.txt, doc3_networking.txt

------------------------------------------------------------
  RUNNING DEMO QUERIES
------------------------------------------------------------

  [1/6] Q: What is a database and how do indexes work?
  [loop    ] USER INPUT: What is a database and how do indexes work?
  [rag     ] RETRIEVED 2 docs: ['doc2_databases.txt', 'doc3_networking.txt']
  [lm      ] RESPONSE: Based on my knowledge base: A database is an organized...
  [loop    ] FINAL ANSWER (citations: ['doc2_databases.txt', 'doc3_networking.txt'])
  A: Based on my knowledge base:
A database is an organized collection of structured data stored electronically.
Relational databases store data in tables with rows and columns, using SQL for queries.
Indexes speed up read operations by creating lookup structures similar to a book's index page.

[Grounded in retrieved context]

[Citations: doc2_databases.txt, doc3_networking.txt]

  [2/6] Q: Explain the difference between TCP and UDP
  ...
  A: Based on my knowledge base:
TCP provides reliable, ordered delivery with acknowledgments;
UDP offers faster but unreliable datagram transmission.

[Grounded in retrieved context]

[Citations: doc3_networking.txt]

  [4/6] Q: store favorite_language Python
  ...
  A: Done! I've stored that for you.

  [5/6] Q: recall favorite_language
  ...
  A: Here's what I had stored: python

  [6/6] Q: What is the capital of Australia?
  ...
  A: I don't have enough information to answer that question.
     Try asking about Python, databases, or networking.
```

Running `python test_adversarial.py` produces:

```
  Running ToyRAG adversarial tests...

  PASS  Broken LM / Chaotic output
  PASS  Hostile / Invalid input
  PASS  Tool execution failure
  PASS  Infinite loop / Max iterations
  PASS  No matches / Empty corpus

  5/5 tests passed.
```

---

## Architecture Diagram

```
+------------------------------------------------------------------+
|                        USER (main.py)                            |
|                     "What is a database?"                        |
+------------------------------------------------------------------+
              |
              v
+------------------------------------------------------------------+
|                    GUARDRAILS (harness/guardrails.py)            |
|  [ ] Length check (max 500 chars)                                |
|  [ ] Empty input check                                           |
|  [ ] Forbidden pattern scan (regex list)                         |
|  [ ] Iteration budget (max 10 per turn)                          |
+------------------------------------------------------------------+
              |  (valid input passes through)
              v
+------------------------------------------------------------------+
|                 AGENT LOOP (harness/agent.py)                    |
|                                                                  |
|  +------------------------------------------------------------+  |
|  |  WHILE iterations < max:                                    |  |
|  |                                                             |  |
|  |  +----------------------------------------+                 |  |
|  |  |  RAG PIPELINE (rag/pipeline.py)        |                 |  |
|  |  |                                        |                 |  |
|  |  |  1. RETRIEVE                           |                 |  |
|  |  |     rag/retriever.py                   |                 |  |
|  |  |     - Tokenize query                   |                 |  |
|  |  |     - Score each doc (keyword overlap) |                 |
|  |  |     - Return top-K above threshold     |                 |  |
|  |  |                                        |                 |  |
|  |  |  2. AUGMENT                            |                 |  |
|  |  |     - Format retrieved docs as context |                 |  |
|  |  |     - Return (context_string, metadata)|                 |  |
|  |  +----------------------------------------+                 |  |
|  |              |                                               |  |
|  |              v                                               |  |
|  |  +----------------------------------------+                 |  |
|  |  |  SIMULATED LM (harness/lm.py)          |                 |  |
|  |  |                                        |                 |  |
|  |  |  Input: prompt + retrieved context     |                 |  |
|  |  |  Output: TOOL_CALL or final answer     |                 |  |
|  |  |                                        |                 |  |
|  |  |  Rules:                                |                 |  |
|  |  |   - If tool result in context -> done  |                 |
|  |  |   - If "store" keyword -> TOOL_CALL    |                 |
|  |  |   - If "recall" keyword -> TOOL_CALL   |                 |
|  |  |   - If context matches -> grounded ans |                 |
|  |  |   - Else -> "I don't know"             |                 |  |
|  |  +----------------------------------------+                 |  |
|  |              |                                               |  |
|  |              v                                               |  |
|  |     +-----------+       +-----------+                         |  |
|  |     | TOOL_CALL |       | FINAL     |                         |
|  |     +-----------+       | ANSWER    |                         |
|  |          |               +-----------+                         |
|  |          v                     |                              |
|  |  +------------------+          |                              |  |
|  |  | TOOLS (harness/  |          |                              |  |
|  |  |   tools.py)      |          |                              |  |
|  |  |                  |          |                              |  |
|  |  | memory_store()   |          v                              |  |
|  |  | memory_recall()  |    +------------------+                 |  |
|  |  | list_corpus()    |    | ADD CITATIONS    |                 |  |
|  |  +------------------+    | (doc IDs from    |                 |  |
|  |          |               |  retrieval)      |                 |  |
|  |          v               +------------------+                 |  |
|  |  Feed result back to     |                              |  |
|  |  conversation, LOOP      |                              |  |
|  +------------------------------------------------------------+  |
+------------------------------------------------------------------+
              |
              v
+------------------------------------------------------------------+
|                    USER SEES ANSWER                               |
|  "Based on my knowledge base: ... [Citations: doc2_databases.txt]"|
+------------------------------------------------------------------+
```

**Key design decision:** The RAG pipeline sits *inside* the agent loop, not outside it. This means every iteration (including post-tool-call iterations) gets fresh retrieval based on the updated conversation. In production systems, you might retrieve once per user turn — but retrieving per-iteration is simpler and demonstrates that the loop structure is truly unchanged from Module #0.

---

## State Machine Diagram

```
States:
  IDLE          - Waiting for user input
  VALIDATING    - Guardrails checking the input
  RETRIEVING    - RAG pipeline searching the corpus
  GENERATING    - LM producing a response (tool call or answer)
  EXECUTING     - Running a tool (if LM emitted TOOL_CALL)
  RESPONDING    - Formatting final answer with citations
  BLOCKED       - Input rejected by guardrails
  TIMEOUT       - Max iterations reached
  STUCK         - Duplicate tool call detected

Transitions:

  IDLE --[user input received]--> VALIDATING

  VALIDATING --[input valid]--> RETRIEVING
  VALIDATING --[input invalid]--> BLOCKED --> IDLE (return rejection)

  RETRIEVING --[docs found or not]--> GENERATING

  GENERATING --[LM says TOOL_CALL]--> EXECUTING
  GENERATING --[LM says final answer]--> RESPONDING

  EXECUTING --[tool succeeds]--> RETRIEVING (loop back, iteration+1)
  EXECUTING --[tool fails]--> RETRIEVING (loop back, error in context)
  EXECUTING --[same call as last iter]--> STUCK --> IDLE

  RESPONDING --[answer + citations ready]--> IDLE (return to user)

  Any state --[iteration >= max]--> TIMEOUT --> IDLE


Visual:

  +-------+     input      +------------+    valid     +------------+
  | IDLE  |--------------->| VALIDATING |-----------> | RETRIEVING |
  +-------+                +------------+              +------------+
      ^                        | invalid                   |
      |                        v                           v
      |                   +---------+              +------------+
      +-------------------| BLOCKED |              | GENERATING |
                          +---------+              +------------+
                                                   /          \
                                          TOOL_CALL/            \final answer
                                                  v              v
                                            +-----------+  +------------+
                                            | EXECUTING |  | RESPONDING |
                                            +-----------+  +------------+
                                                  |              |
                                                  | (loop)       |
                                                  v              v
                                           back to          return to
                                         RETRIEVING         IDLE

  Guards:
    VALIDATING -> BLOCKED:     len > 500 OR empty OR forbidden pattern
    EXECUTING -> STUCK:        current_call == last_tool_call
    Any -> TIMEOUT:            iteration_count >= max_iterations (10)
```

**Loop structure note:** The inner loop (RETRIEVING -> GENERATING -> EXECUTING -> RETRIEVING) is the same pattern as Module #0. RAG adds the RETRIEVING state before every GENERATING step. The outer boundary is the iteration guardrail (max 10 cycles per user turn).

---

## File Reference

**`main.py`** — The entry point. Creates the LM, RAG pipeline, and agent, then runs 6 demo queries that showcase different behaviors: grounded answers (queries 1-3), tool calls with memory (queries 4-5), and graceful "I don't know" when no documents match (query 6). This is what you run first to see the system in action.

**`trace_logger.py`** — A lightweight logging utility that colors terminal output by component. Guardrails are red, the LM is cyan, tools are yellow, RAG retrieval is magenta, and the agent loop is green. After execution completes, calling `report()` prints the full event log so you can see exactly what happened internally. Think of it as a "flight recorder" for your agent.

**`harness/agent.py`** — The main orchestrator. Contains the `Agent` class with its `run()` method that implements the loop: validate input, retrieve documents via RAG, call the LM, parse the response, execute tools if needed, and return a final answer with citations. Also contains the duplicate-tool-call detection (the "stuck" guard) that prevents infinite loops from a broken LM. This is the file to read first if you want to understand the control flow.

**`harness/lm.py`** — The simulated brain. Uses keyword matching to decide whether to emit a tool call or produce a final answer. The key RAG-specific behavior: it checks if the prompt already contains a `[Tool result from ...]` line (meaning the task is done) before deciding to call another tool. When generating answers, it extracts sentences from the retrieved context that share keywords with the query — this simulates "grounding" without an actual language model.

**`harness/tools.py`** — The tool registry. Maps tool names (`memory_store`, `memory_recall`, `list_corpus`) to their implementations. Each tool is a simple function that takes a dictionary of arguments and returns a string result. The `execute()` method wraps each call in try/except so a crashing tool cannot kill the agent loop. Think of this as the agent's "hands" — the things it can physically do.

**`harness/guardrails.py`** — The safety layer. Validates user input (length, emptiness, forbidden patterns like SQL injection) and enforces the iteration budget. The `check_iterations()` method uses a "check before increment" pattern so the counter accurately reflects consumed iterations. Also tracks blocked events for audit purposes. This is the seatbelt: it does not drive the car, but it prevents crashes.

**`harness/memory.py`** — A simple key-value store using Python's `dataclasses`. Each entry has a key, value, and timestamp. Supports `store`, `recall`, `list_keys`, and `count` operations. This persists across tool calls within a single agent session (not across program runs — for that you would add file I/O).

**`rag/pipeline.py`** — The RAG orchestrator. Its main method `retrieve_and_augment()` takes a query, calls the retriever to get top-K documents, formats them into a context string with document IDs and relevance scores, and returns both the context (to inject into the prompt) and the metadata list (for citations). This is the "glue" that connects retrieval to generation.

**`rag/retriever.py`** — The scoring engine. Tokenizes the query and each document, counts keyword overlaps, and computes a relevance score (fraction of query terms found in the document). Documents below a minimum threshold (0.15) are discarded. Results are sorted by score and truncated to top-K (3 by default). A small bonus is applied to short, highly-matching documents to favor focused content over general text. This is intentionally simple — in production you would use vector embeddings (Modules #5-6), but the concept of "score, rank, truncate" is identical.

**`rag/corpus.py`** — The document loader. At initialization, it reads all `.txt` files from the `corpus/` directory into an in-memory dictionary keyed by filename. Provides `list_documents()`, `get_document()`, `get_all()`, and `count()` methods. If the directory does not exist, it gracefully returns an empty corpus (no crash). This represents the "ingestion" step of a real RAG system — in production, documents come from databases, APIs, or file systems.

**`corpus/doc1_python.txt`** — A 5-line document about the Python programming language: its history, design philosophy, supported paradigms, standard library, and typing system. Used to demonstrate retrieval when the user asks about Python.

**`corpus/doc2_databases.txt`** — A 5-line document about database systems: what they are, relational vs NoSQL, ACID properties, and how indexes work. The "star" document for the demo — it gets retrieved for database-related queries.

**`corpus/doc3_networking.txt`** — A 5-line document about computer networking: the TCP/IP model, TCP vs UDP, HTTP, and DNS. Demonstrates that retrieval is topic-specific (networking queries do not retrieve the Python document).

**`test_adversarial.py`** — Five tests that prove the system survives failure conditions without crashing: (1) a broken LM that returns garbage, (2) hostile inputs that should be blocked, (3) a tool that raises an exception, (4) an LM that loops forever (hits the iteration cap), and (5) queries with no matches in an empty corpus. If all 5 pass, you know the harness is robust enough to handle real-world messiness.

---

## What This Teaches

| Concept                                     | Where to look                                                                         | Key insight                                                                                                                                                                              |
| ------------------------------------------- | ------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RAG = search + paste                        | `rag/pipeline.py` line 20-35                                                          | The entire "AI" trick is: find relevant text, put it in the prompt. No magic.                                                                                                            |
| Grounding prevents hallucination            | `harness/lm.py` Rule 4 vs Rule 5                                                      | With context, the LM answers from evidence. Without context, it says "I don't know." That honesty IS the grounding guarantee.                                                            |
| Citations = traceability                    | `harness/agent.py` lines 85-96                                                        | Every answer carries its sources. A user can verify: "Did the agent actually read that document?"                                                                                        |
| Retrieval quality determines answer quality | `rag/retriever.py` scoring logic                                                      | If you retrieve the wrong documents, no amount of model intelligence fixes it. Garbage in, garbage out.                                                                                  |
| The agent loop is unchanged                 | Compare `harness/agent.py` to Module #0                                               | RAG adds a RETRIEVE step before each LM call. The loop structure (validate, think, act, repeat) is identical. You can add RAG to any existing agent by inserting one method call.        |
| Graceful degradation                        | `rag/corpus.py` empty dir handling; `harness/lm.py` Rule 5                            | When the corpus is empty or nothing matches, the system says "I don't know" instead of crashing or making things up.                                                                     |
| Defense in depth for tool execution         | `harness/agent.py` try/except around execute + `harness/tools.py` internal try/except | Two layers catch tool failures: the registry catches handler exceptions, the agent catches registry-level exceptions. Neither can kill the loop.                                         |
| Stuck detection prevents wasted compute     | `harness/agent.py` duplicate call check                                               | If the LM emits the same tool call twice in a row, something is wrong. Break early instead of burning all 10 iterations.                                                                 |
| Keyword retrieval vs vector retrieval       | `rag/retriever.py` vs Module #6                                                       | Both do "score, rank, truncate." Keyword matching is transparent and debuggable; vector matching handles semantic similarity. The architecture is identical — swap the scoring function. |

---

## Intuitive Summary (for non-IT readers)

Here is the whole system as a single analogy:

You are a **customer service agent** at a company with a **knowledge base** (the `corpus/` folder). When a customer asks a question:

1. **Security check** (guardrails): Is this a real question, or someone trying to hack the system? If it looks malicious, reject it politely.
2. **Look up the manual** (RAG retrieval): Search the knowledge base for pages relevant to the question. You find 1-3 pages that seem most related.
3. **Read and answer** (LM generation): Using those pages as reference, write an answer. If the pages clearly answer the question, you cite them. If no relevant pages exist, you honestly say "I don't have that information."
4. **Use your tools if needed** (tool execution): Sometimes the customer asks you to do something (remember a preference, check available documents). You use your tools, then come back to step 2 with the updated context.
5. **Safety limit** (iteration cap): If you find yourself going in circles (calling the same tool repeatedly), a supervisor steps in and stops you after 10 attempts.

That is the entire system. No neural networks. No GPUs. No API keys. Just structured information flow with guardrails at every step.

---

## Curriculum Position

This is **Module #7** in the TOYAGENTS curriculum. It builds on:

- **#5 ToyEmbeddings** — taught you that text can become numbers and similarity can be measured
- **#6 ToyVectorDB** — taught you how to store those numbers and search by nearest neighbor

This module uses a simpler keyword-based retrieval (no vectors needed) to keep the focus on the **RAG pipeline architecture** rather than the retrieval algorithm. In Module #10 (Capstone), you will swap the keyword retriever for the vector-based one from #6 — the pipeline code does not change at all. That is the point: the architecture is independent of the retrieval method.

Next up: **#8 ToySandBox** — isolated tool execution with resource limits and audit logging.

---

Parent curriculum: [TOYAGENTS](https://github.com/AlexOcculate/TOYAGENTS)