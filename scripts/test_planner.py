# scripts/test_planner.py
"""
Test Step 10c — Query planner on diverse query types.
Verifies intent detection, entity extraction, explicit vs
inferred constraint separation, and budget assignment.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.query.planner import plan_query

test_queries = [
    # Fact queries
    ("What spectral index n did Murgia 2004 find for A119?",
     "fact"),
    ("What is the sigma_RM value for the Coma cluster?",
     "fact"),
    ("How strong is the magnetic field B_0 in Abell 2255?",
     "fact"),

    # Comparison queries
    ("Compare the GRF and BxC magnetic field models for galaxy clusters",
     "comparison"),
    ("What is the difference between analytical and MHD simulation methods?",
     "comparison"),

    # Synthesis queries
    ("What does the literature say about Faraday rotation in galaxy clusters?",
     "synthesis"),
    ("Summarize what is known about depolarization in the intracluster medium",
     "synthesis"),

    # Discovery queries
    ("What papers studied A119 using VLA observations?",
     "discovery"),
    ("Find papers related to turbulent magnetic fields after 2015",
     "discovery"),

    # Explicit constraint queries
    ("MeerKAT papers after 2020 on galaxy cluster magnetic fields",
     "discovery"),
    ("MNRAS papers about RM synthesis since 2018",
     "fact"),
]

print(f"{'Query':<55} {'Expected':<12} {'Got':<12} {'Match'}")
print("-" * 90)

all_pass = True
for query, expected_intent in test_queries:
    result = plan_query(query)
    got    = result["intent"]
    match  = "✓" if got == expected_intent else "✗"
    if got != expected_intent:
        all_pass = False
    print(f"{query[:54]:<55} {expected_intent:<12} {got:<12} {match}")

print(f"\n{'All intent detections passed ✓' if all_pass else 'Some intents wrong ✗'}")

# ── Detailed output for one query ─────────────────────────────────────────────
print("\n" + "="*60)
print("Detailed output for complex query:")
q = "Compare GRF and BxC magnetic field models in galaxy clusters using VLA after 2010"
result = plan_query(q)
print(f"Query: {q}")
print(f"\nintent          : {result['intent']}")
print(f"synthesis_mode  : {result['synthesis_mode']}")
print(f"\nentities:")
for k, v in result["entities"].items():
    if v:
        print(f"  {k:<15}: {v}")
print(f"\nexplicit_filters:")
for k, v in result["explicit_filters"].items():
    if v:
        print(f"  {k:<15}: {v}")
print(f"\nsoft_boosts:")
for k, v in result["soft_boosts"].items():
    if v:
        print(f"  {k:<15}: {v}")
print(f"\nbudgets:")
for k, v in result["budgets"].items():
    print(f"  {k:<15}: {v}")