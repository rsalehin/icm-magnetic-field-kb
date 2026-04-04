# scripts/debug_ollama.py
import ollama

# Test 1 — minimal call, no options
print("Test 1 — minimal call...")
response = ollama.chat(
    model    = "qwen3:14b",
    messages = [{"role": "user", "content": "Reply with just this JSON: {\"test\": \"ok\"}"}],
)
print(f"Response: {response['message']['content'][:200]}")

# Test 2 — with options
print("\nTest 2 — with num_predict option...")
response = ollama.chat(
    model    = "qwen3:14b",
    messages = [{"role": "user", "content": "Reply with just this JSON: {\"test\": \"ok\"}"}],
    options  = {"temperature": 0.3, "num_predict": 200},
)
print(f"Response: {response['message']['content'][:200]}")

# Test 3 — check think option
print("\nTest 3 — with think=False...")
try:
    response = ollama.chat(
        model    = "qwen3:14b",
        messages = [{"role": "user", "content": "Reply with just this JSON: {\"test\": \"ok\"}"}],
        think    = False,
    )
    print(f"Response: {response['message']['content'][:200]}")
except Exception as e:
    print(f"Error: {e}")