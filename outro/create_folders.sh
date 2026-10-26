#!/bin/bash

# chmod +x create_folders.sh

touch main.py
touch trace_logger.py
touch test_adversarial.py
touch README.md
mkdir -p Harness/
touch Harness/__init__.py
touch Harness/agent.py
touch Harness/lm.py
touch Harness/tools.py
touch Harness/guardrails.py
touch Harness/memory.py
mkdir -p RAG/
touch RAG/__init__.py
touch RAG/pipeline.py
touch RAG/retriever.py
touch RAG/corpus.py
mkdir -p Corpus/
touch Corpus/doc1_python.txt
touch Corpus/doc2_databases.txt
touch Corpus/doc3_networking.txt