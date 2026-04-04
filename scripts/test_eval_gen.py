# scripts/test_eval_gen.py
import sys
import random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
random.seed(42)

from src.storage.db import get_connection
from scripts.generate_eval_set import sample_chunks, generate_qa_pair

conn   = get_connection()
chunks = sample_chunks(conn, 3, 2000, 2010)
print(f"Sampled {len(chunks)} chunks\n")

for c in chunks:
    print(f"[{c['year']}] {c['title'][:55]}")
    print(f"  arxiv   : {c['arxiv_id']}")
    print(f"  section : {c['section_title']}")
    print(f"  text    : {c['text'][:200]}...")
    print()
    qa = generate_qa_pair(c)
    if qa:
        print(f"  Q : {qa['question']}")
        print(f"  A : {qa['answer']}")
        print(f"  KT: {qa['key_terms']}")
    else:
        print("  ✗ Generation failed or low quality")
    print("-" * 60)

conn.close()