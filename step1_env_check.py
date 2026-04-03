# step1_env_check.py
import sys, platform

print("=== Python & OS ===")
print(f"Python : {sys.version}")
print(f"OS     : {platform.platform()}")

print("\n=== GPU ===")
try:
    import torch
    print(f"PyTorch : {torch.__version__}")
    print(f"CUDA available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  GPU {i} : {p.name}  |  VRAM: {p.total_memory/1e9:.1f} GB")
except ImportError:
    print("PyTorch not installed")

print("\n=== RAM ===")
try:
    import psutil
    vm = psutil.virtual_memory()
    print(f"Total : {vm.total/1e9:.1f} GB  |  Available : {vm.available/1e9:.1f} GB")
except ImportError:
    print("psutil not installed — run: pip install psutil")

print("\n=== Key packages (installed?) ===")
packages = [
    "faiss", "duckdb", "networkx", "sentence_transformers",
    "transformers", "pdfplumber", "pymupdf", "requests", "tqdm"
]
for pkg in packages:
    try:
        mod = __import__(pkg.replace("-","_"))
        ver = getattr(mod, "__version__", "?")
        print(f"  ✓ {pkg:<25} {ver}")
    except ImportError:
        print(f"  ✗ {pkg:<25} NOT INSTALLED")