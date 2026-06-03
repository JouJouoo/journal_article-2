from __future__ import annotations

import torch


def cuda_device_count() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def resolve_device(requested: str | None = None) -> torch.device:
    """Resolve cpu/cuda/auto into a torch.device with a CPU fallback for auto."""

    name = (requested or "auto").strip().lower()
    if name in {"auto", "gpu"}:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda:0")
    if name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"{requested} was requested, but CUDA is unavailable.")
        return torch.device(name)
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be one of: auto, cpu, cuda, cuda:<index>")


def assign_parallel_device(requested: str | None, worker_index: int) -> str:
    """Assign a device string for parallel experiment workers."""

    name = (requested or "auto").strip().lower()
    count = cuda_device_count()
    if name in {"auto", "gpu", "cuda"} and count > 0:
        return f"cuda:{worker_index % count}"
    if name in {"auto", "gpu"}:
        return "cpu"
    return name
