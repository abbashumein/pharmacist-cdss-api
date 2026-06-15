import os
import time

def print_memory():
    # This reads the exact RAM usage of the current Python process
    import psutil
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    print(f"Current Memory Usage: {mem_mb:.2f} MB")

print("--- Step 1: Base Python process ---")
print_memory()

print("\n--- Step 2: Loading Heavy Frameworks ---")
from fastapi import FastAPI
from langgraph.graph import StateGraph
import chromadb
print_memory()

print("\n--- Step 3: Simulating default Chroma Client init ---")
# If psutil isn't installed, run 'pip install psutil' first
try:
    from chromadb.utils import embedding_functions
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    # This triggers the onnx runtime model load
    default_ef(["Test text structural load"])
    print_memory()
except Exception as e:
    print(f"Could not load embedding model details: {e}")