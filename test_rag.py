"""Test script for app.py RAG pipeline."""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app import load_index, retrieve_top_k, generate_answer

def test_rag_pipeline():
    print("1. Testing load_index()...")
    model, index, chunks = load_index()
    print(f"   Chunks loaded ({len(chunks)} chunks):")
    for i, c in enumerate(chunks):
        print(f"   [{i}] {c}")

    print("\n2. Testing retrieve_top_k()...")
    test_queries = [
        "นมหมีฮอกไกโดราคาเท่าไหร่",
        "มีเมนูคนแพ้ถั่วทานได้ไหม",
        "ร้านเปิดกี่โมง",
        "ขายแฮมเบอร์เกอร์ไหม"
    ]

    for q in test_queries:
        retrieved = retrieve_top_k(q, model, index, chunks, k=3)
        print(f"\nQuery: '{q}'")
        for idx, chunk in enumerate(retrieved, 1):
            print(f"   Top-{idx}: {chunk}")
        
        print("   Generating Answer from Gemini...")
        ans = generate_answer(q, retrieved)
        print(f"   Answer: {ans}")

if __name__ == "__main__":
    test_rag_pipeline()
