"""Runtime resource helpers for GANular."""

from typing import Callable

import torch


def recommend_accelerator_batch_size(
    run_batch: Callable[[int], None],
    max_batch: int = 32,
    headroom: float = 0.75,
) -> int:
    """Estimate a conservative accelerator batch size from a batch-of-one probe."""
    if max_batch < 1:
        raise ValueError("max_batch must be positive")
    if not 0 < headroom <= 1:
        raise ValueError("headroom must be in the interval (0, 1]")
    if not torch.accelerator.is_available():
        return max_batch

    memory = torch.accelerator.memory
    memory.empty_cache()
    torch.accelerator.synchronize()
    baseline = memory.memory_allocated()
    memory.reset_peak_memory_stats()
    run_batch(1)
    torch.accelerator.synchronize()

    probe_bytes = memory.max_memory_allocated() - baseline
    memory.empty_cache()
    free_bytes, _ = memory.get_memory_info()
    if probe_bytes <= 0:
        return 1
    return max(1, min(max_batch, int(free_bytes * headroom / probe_bytes)))
