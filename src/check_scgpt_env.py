#!/usr/bin/env python
"""Check that scgpt imports inside the container; if not, reinstall the dependency
stack documented in obesity_scGPT_perturbation_prediction/notebooks/Install_deps.ipynb
into user site-packages (~/.local), then re-check.

Deliberately omits two things from the notebook:
- the `rm -rf ~/.local/lib/python3.11/site-packages/*` wipe (too destructive/broad)
- the local flash_attn wheel install (file no longer exists; optional dep --
  scgpt imports fine without it, just without the fused attention kernel)
"""
import sys
import subprocess


def print_env():
    import torch
    import numpy
    print(f"python: {sys.executable}")
    print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}  "
          f"cuda_available: {torch.cuda.is_available()}  numpy: {numpy.__version__}")


def try_import_scgpt():
    try:
        import scgpt
        print(f"scgpt: {scgpt.__version__}  location: {scgpt.__file__}")
        return True
    except Exception as e:
        print(f"scgpt import FAILED: {e!r}")
        return False


def pip_user(*args):
    cmd = [sys.executable, "-m", "pip", "install", "--user", *args]
    print("+ " + " ".join(cmd))
    subprocess.check_call(cmd)


print("=== Initial environment check ===")
print_env()

print("=== Attempting `import scgpt` ===")
ok = try_import_scgpt()

if ok:
    print("RESULT: scgpt import OK, no reinstall needed.")
    sys.exit(0)

print("=== scgpt import failed; reinstalling deps per Install_deps.ipynb ===")
pip_user("--upgrade", "pip", "setuptools", "wheel")
pip_user("scvi-tools==0.20.3", "scanpy==1.9.3", "anndata==0.9.2", "scib==1.1.7")
pip_user("torchtext==0.17.0", "torchdata==0.7.1", "--no-deps")
pip_user(
    "numba>=0.55.1", "scikit-misc>=0.1.4", "umap-learn>=0.5.3",
    "leidenalg>=0.8.10", "datasets>=2.3.0,<3.0",
    "typing-extensions>=4.2.0,<5.0", "cell-gears<0.0.3", "orbax<0.1.8",
)
pip_user(
    "--no-deps", "torch-scatter", "torch-sparse", "torch-cluster",
    "torch-spline-conv", "torch-geometric",
    "-f", "https://data.pyg.org/whl/torch-2.2.1+cu121.html", "--no-cache-dir",
)
pip_user("scgpt", "--no-deps")
pip_user("wandb", "faiss-cpu", "nbformat")

print("=== Re-checking scgpt import after reinstall ===")
ok2 = try_import_scgpt()
print("RESULT:", "scgpt import OK after reinstall." if ok2 else "scgpt import STILL FAILING after reinstall.")
sys.exit(0 if ok2 else 1)
