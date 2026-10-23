"""
scripts/check_env.py
Environment sanity checks for ThriftNAS (STEP 0).
Each test prints PASS or FAIL with a short reason.
Run with:  python scripts/check_env.py
"""

import sys
import time
import shutil
import subprocess
import importlib

SEP = "-" * 60


def _pf(label: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")


# ---------------------------------------------------------------------------
# T1 – PyTorch + CUDA + device name + VRAM
# ---------------------------------------------------------------------------
def test_t1() -> None:
    print(f"\n{SEP}")
    print("T1  PyTorch / CUDA / device / VRAM")
    print(SEP)
    try:
        import torch  # noqa: F401
        _pf("torch imports", True, f"version {torch.__version__}")
    except ImportError as exc:
        _pf("torch imports", False, str(exc))
        return  # remaining T1 sub-tests are meaningless

    cuda_ok = torch.cuda.is_available()
    _pf("cuda.is_available()", cuda_ok)
    if not cuda_ok:
        _pf("device name contains '3050'", False, "CUDA not available")
        _pf("total VRAM 5.5–6.5 GB", False, "CUDA not available")
        return

    props = torch.cuda.get_device_properties(0)
    name = props.name
    name_ok = "3050" in name
    _pf("device name contains '3050'", name_ok, name)

    vram_gb = props.total_memory / 2**30
    vram_ok = 5.5 <= vram_gb <= 6.5
    _pf("total VRAM 5.5–6.5 GB", vram_ok, f"{vram_gb:.2f} GB")


# ---------------------------------------------------------------------------
# T2 – Package imports
# ---------------------------------------------------------------------------
def test_t2() -> None:
    print(f"\n{SEP}")
    print("T2  Python package imports")
    print(SEP)
    packages = [
        ("torchvision", "torchvision"),
        ("fvcore", "fvcore"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
        ("pandas", "pandas"),
        ("gradio", "gradio"),
        ("pytest", "pytest"),
        ("yaml (PyYAML)", "yaml"),
        ("graphviz", "graphviz"),
    ]
    for label, mod in packages:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            _pf(label, True, f"version {ver}")
        except ImportError as exc:
            _pf(label, False, str(exc))


# ---------------------------------------------------------------------------
# T3 – System Graphviz (`dot -V`)
# ---------------------------------------------------------------------------
def test_t3() -> None:
    print(f"\n{SEP}")
    print("T3  System Graphviz (`dot -V`)")
    print(SEP)
    try:
        result = subprocess.run(
            ["dot", "-V"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # `dot -V` writes to stderr
        version_line = (result.stderr or result.stdout).strip().splitlines()[0]
        _pf("`dot -V` runs", result.returncode == 0, version_line)
    except FileNotFoundError:
        _pf(
            "`dot -V` runs",
            False,
            "graphviz not found – install: winget install graphviz.graphviz  "
            "(or: choco install graphviz)",
        )
    except Exception as exc:
        _pf("`dot -V` runs", False, str(exc))


# ---------------------------------------------------------------------------
# T4 – GPU fp16 matmul throughput
# ---------------------------------------------------------------------------
def test_t4() -> None:
    print(f"\n{SEP}")
    print("T4  GPU fp16 2048×2048 matmul throughput")
    print(SEP)
    try:
        import torch

        if not torch.cuda.is_available():
            _pf("fp16 matmul", False, "CUDA not available")
            return

        device = torch.device("cuda")
        N = 2048
        a = torch.randn(N, N, device=device, dtype=torch.float16)
        b = torch.randn(N, N, device=device, dtype=torch.float16)

        # warm-up
        for _ in range(3):
            _ = a @ b
        torch.cuda.synchronize()

        RUNS = 10
        t0 = time.perf_counter()
        for _ in range(RUNS):
            c = a @ b  # noqa: F841
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / RUNS  # seconds per matmul

        # TFLOPs: matmul of (N,N)@(N,N) = 2*N^3 FLOPs
        tflops = 2 * N**3 / elapsed / 1e12
        # Memory bandwidth: read a + b, write c = 3 * N^2 * 2 bytes (fp16)
        bw_gbs = 3 * N * N * 2 / elapsed / 1e9

        _pf(
            "fp16 matmul completes",
            True,
            f"{tflops:.2f} TFLOPs  |  {bw_gbs:.1f} GB/s  |  {elapsed*1000:.2f} ms/run",
        )
    except Exception as exc:
        _pf("fp16 matmul", False, str(exc))


# ---------------------------------------------------------------------------
# T5 – Thread count
# ---------------------------------------------------------------------------
def test_t5() -> None:
    print(f"\n{SEP}")
    print("T5  torch thread control")
    print(SEP)
    try:
        import torch

        before = torch.get_num_threads()
        print(f"  torch.get_num_threads() before set: {before}")
        torch.set_num_threads(1)
        after = torch.get_num_threads()
        print(f"  torch.get_num_threads() after  set: {after}")
        _pf("set_num_threads(1) -> get_num_threads() == 1", after == 1)
    except Exception as exc:
        _pf("thread control", False, str(exc))


# ---------------------------------------------------------------------------
# T6 – DataLoader with num_workers=2 (must live inside __main__)
# ---------------------------------------------------------------------------
def _run_dataloader_test() -> None:
    """Called only from __main__ block to satisfy Windows spawn requirement."""
    print(f"\n{SEP}")
    print("T6  DataLoader num_workers=2 over TensorDataset")
    print(SEP)
    try:
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        ds = TensorDataset(torch.arange(64, dtype=torch.float32))
        dl = DataLoader(ds, batch_size=8, num_workers=2, shuffle=False)
        batches = [b[0] for b in dl]
        total = sum(bt.numel() for bt in batches)
        all_ok = (len(batches) == 8) and (total == 64)
        _pf(
            "DataLoader num_workers=2",
            all_ok,
            f"{len(batches)} batches, {total} items total",
        )
    except Exception as exc:
        _pf("DataLoader num_workers=2", False, str(exc))


# ---------------------------------------------------------------------------
# T7 – Free disk space
# ---------------------------------------------------------------------------
def test_t7() -> None:
    print(f"\n{SEP}")
    print("T7  Free disk space on project drive")
    print(SEP)
    try:
        usage = shutil.disk_usage(".")
        free_gb = usage.free / 2**30
        ok = free_gb > 5.0
        _pf("free disk > 5 GB", ok, f"{free_gb:.1f} GB free")
    except Exception as exc:
        _pf("free disk", False, str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("ThriftNAS – Environment Check (STEP 0)")
    print("=" * 60)

    test_t1()
    test_t2()
    test_t3()
    test_t4()
    test_t5()
    _run_dataloader_test()   # T6 – must be inside __main__ for Windows spawn
    test_t7()

    print(f"\n{'=' * 60}")
    print("Environment check complete.")
    print("=" * 60)
