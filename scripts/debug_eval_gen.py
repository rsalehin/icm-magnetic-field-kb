# scripts/debug_eval_gen.py
import sys
import json
import re
import random
import ollama
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
random.seed(42)

from src.storage.db import get_connection
from scripts.generate_eval_set import sample_chunks, GENERATION_PROMPT, QWEN_MODEL

conn   = get_connection()
chunks = sample_chunks(conn, 1, 2000, 2010)
conn.close()

chunk = chunks[0]
print(f"Chunk: {chunk['title'][:60]}")
print(f"Text : {chunk['text'][:200]}\n")

prompt = GENERATION_PROMPT.format(
    title   = chunk["title"][:80],
    year    = chunk["year"],
    section = chunk["section_title"],
    text    = chunk["text"][:800],
)

response = ollama.chat(
    model    = QWEN_MODEL,
    messages = [{"role": "user", "content": prompt}],
    options  = {"temperature": 0.3, "num_predict": 300, "think": False},
)

raw = response["message"]["content"].strip()
print(f"Raw response:\n{raw}\n")

# Try parsing
raw_clean = re.sub(r"```json|```", "", raw).strip()
print(f"After cleanup:\n{raw_clean}\n")

try:
    data = json.loads(raw_clean)
    print(f"Parsed OK: {data}")
except json.JSONDecodeError as e:
    print(f"JSON parse error: {e}")