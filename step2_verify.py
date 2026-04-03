# step2_verify.py
packages = {
    "duckdb": "duckdb",
    "faiss": "faiss",
    "networkx": "networkx",
    "sentence_transformers": "sentence_transformers",
    "transformers": "transformers",
    "fitz (pymupdf)": "fitz",
    "pdfplumber": "pdfplumber",
}
for label, pkg in packages.items():
    try:
        mod = __import__(pkg)
        print(f"  ✓ {label:<25} {getattr(mod, '__version__', '?')}")
    except ImportError as e:
        print(f"  ✗ {label:<25} {e}")

# Confirm torch still intact
import torch
print(f"\n  torch still: {torch.__version__}  |  CUDA: {torch.cuda.is_available()}")