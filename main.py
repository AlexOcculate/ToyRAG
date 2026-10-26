# Line 1:  Import os for path setup (ensures relative imports work from any CWD).
import os

# Line 2:  Import sys for exit codes and path manipulation.
import sys

# Line 3:  Add the current directory to sys.path so local modules are importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Line 5:  Import the Agent class (the main orchestrator).
from Harness.agent import Agent

# Line 6:  Import SimulatedLM for explicit brain creation.
from Harness.lm import SimulatedLM

# Line 7:  Import RAGPipeline for explicit pipeline creation.
from RAG.pipeline import RAGPipeline


# Line 10:  Main function that runs an interactive demo session.
def main():
    # Line 12:  Print a header banner for the module.
    print("=" * 60)
    # Line 13:  Print the module title.
    print("  TOYRAG — Retrieve, Augment, Generate (Toy Implementation)")
    # Line 14:  Print the subtitle explaining the concept.
    print("  RAG = search + paste into prompt. Grounded answers with citations.")
    # Line 15:  Print a separator line.
    print("=" * 60)

    # Line 17:  Create the simulated language model (the "brain").
    lm = SimulatedLM(name="SimLM-RAG-v1")
    # Line 18:  Create the RAG pipeline (loads corpus, creates retriever).
    rag_pipeline = RAGPipeline()
    # Line 19:  Print corpus statistics so the user knows what's available.
    stats = rag_pipeline.stats()
    # Line 20:  Display the number of documents loaded.
    print(f"\n  Corpus: {stats['total_documents']} documents loaded")
    # Line 21:  Display the document IDs.
    print(f"  Documents: {', '.join(stats['document_ids'])}")

    # Line 23:  Create the agent with our LM and RAG pipeline.
    agent = Agent(lm=lm, rag_pipeline=rag_pipeline)

    # Line 25:  Define a set of demo queries to showcase different RAG behaviors.
    demo_queries = [
        # Line 27:  Query about databases — should retrieve doc2_databases.txt.
        "What is a database and how do indexes work?",
        # Line 28:  Query about networking — should retrieve doc3_networking.txt.
        "Explain the difference between TCP and UDP",
        # Line 29:  Query about Python — should retrieve doc1_python.txt.
        "What programming paradigms does Python support?",
        # Line 30:  Memory operation — should trigger a tool call.
        "store favorite_language Python",
        # Line 31:  Recall the stored memory — should trigger another tool call.
        "recall favorite_language",
        # Line 32:  Query with no relevant docs — should get a graceful "I don't know".
        "What is the capital of Australia?",
    ]

    # Line 34:  Print a line before starting the demo queries.
    print(f"\n{'─' * 60}")
    # Line 35:  Print the demo header.
    print("  RUNNING DEMO QUERIES")
    # Line 36:  Print another separator.
    print(f"{'─' * 60}\n")

    # Line 38:  Iterate over each demo query and run it through the agent.
    for i, query in enumerate(demo_queries, 1):
        # Line 40:  Print which query number we're on.
        print(f"  [{i}/{len(demo_queries)}] Q: {query}")
        # Line 41:  Run the agent with this query (triggers full RAG pipeline).
        answer = agent.run(query)
        # Line 42:  Print the agent's answer.
        print(f"  A: {answer}")
        # Line 43:  Print a blank line between queries for readability.
        print()

    # Line 45:  Print the full trace report showing all internal events.
    agent.trace.report()

    # Line 47:  Print a closing message.
    print("  Demo complete. Try running test_adversarial.py for failure-mode tests.\n")


# Line 50:  Standard Python entry point guard.
if __name__ == "__main__":
    # Line 52:  Call the main function to start the demo.
    main()
