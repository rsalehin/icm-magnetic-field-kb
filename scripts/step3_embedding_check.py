# step3_embedding_check.py
import torch
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
import os

MODEL_ID = "allenai/specter2_base"

print("=" * 55)
print(f"CUDA  : {torch.cuda.is_available()}")
print(f"GPU   : {torch.cuda.get_device_name(0)}")
print("=" * 55)

# ── Step 1: Explicit download with progress ───────────────
print(f"\n[1/3] Downloading {MODEL_ID} ...")
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
snapshot_download(
    repo_id=MODEL_ID,
    cache_dir=cache_dir,
)
print("      Download complete.")

# ── Step 2: Load onto GPU ─────────────────────────────────
print(f"\n[2/3] Loading model onto GPU ...")
model = SentenceTransformer(MODEL_ID, device="cuda")
print(f"      Model device : {next(model.parameters()).device}")

# ── Step 3: Encode test sentences ────────────────────────
test_sentences = [
    "The rotation measure dispersion sigma_RM constrains the intracluster magnetic field strength.",
    "We simulate turbulent magnetic fields using a Gaussian random field with a power-law power spectrum.",
    "The Burn law describes exponential depolarization as a function of wavelength squared.",
]

print(f"\n[3/3] Encoding {len(test_sentences)} test sentences ...")
embeddings = model.encode(test_sentences, show_progress_bar=True)

# ── Summary ───────────────────────────────────────────────
print("\n" + "=" * 55)
print(f"Embedding shape : {embeddings.shape}")
print(f"Dtype           : {embeddings.dtype}")
print(f"Model device    : {next(model.parameters()).device}")
device_str = str(next(model.parameters()).device)
if "cuda" in device_str:
    print(f"VRAM used       : {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print("\n✓ GPU embedding confirmed.")
else:
    print("\n✗ WARNING: model is running on CPU, not GPU.")
print("=" * 55)