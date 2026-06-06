"""In-process checkpoint/restore with NCCL-aware hybrid teardown/rebuild.

Unlike MultiGPUCheckpointer (external orchestration), this runs INSIDE each
worker process. Each worker checkpoints its own GPU state via os.getpid().

Tested on:
  - A5000 PCIe TP=2: NCCL survives in place, ~1s cycle
  - H100 NVLink TP=2: hybrid destroy/reinit, ~2.3s cycle
    (destroy 444ms + ckpt/restore 1515ms + reinit 332ms)
  - GPU memory preserved across checkpoint boundary

Usage:
    ckpt = InProcessCheckpointer(
        pre_checkpoint=destroy_nccl,
        post_restore=reinit_nccl,
    )
    ckpt.suspend()
    # ... GPU memory freed ...
    ckpt.resume()
"""

import os
import time
from typing import Callable, Optional

from gpu_checkpoint_orchestrator.api import CudaCheckpointAPI


class InProcessCheckpointer:
    """Checkpoint/restore from within the GPU worker process.

    Each worker calls suspend()/resume() on itself. For multi-GPU with
    NVLink, the caller must provide pre/post hooks that handle NCCL
    teardown/rebuild (collective operations across all ranks).

    The hybrid approach avoids requiring NCCL_P2P_DISABLE=1, so NCCL
    gets full NVLink bandwidth during inference.
    """

    def __init__(
        self,
        pre_checkpoint: Optional[Callable[[], None]] = None,
        post_restore: Optional[Callable[[], None]] = None,
    ):
        self._api = CudaCheckpointAPI()
        self._pre_checkpoint = pre_checkpoint
        self._post_restore = post_restore
        self._state = "RUNNING"

    @property
    def state(self) -> str:
        return self._state

    def suspend(self) -> dict:
        """Teardown NCCL -> lock -> checkpoint.

        Returns timing dict with per-step breakdown in milliseconds.
        """
        if self._state != "RUNNING":
            raise RuntimeError(f"Cannot suspend from state {self._state}")

        pid = os.getpid()
        timings = {}

        if self._pre_checkpoint:
            t0 = time.perf_counter()
            self._pre_checkpoint()
            timings["pre_checkpoint_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        self._api.lock(pid)
        timings["lock_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        try:
            self._api.checkpoint(pid)
        except Exception:
            try:
                self._api.unlock(pid)
            except Exception:
                pass
            raise
        timings["checkpoint_ms"] = (time.perf_counter() - t0) * 1000

        self._state = "SUSPENDED"
        return timings

    def resume(self) -> dict:
        """Restore -> unlock -> rebuild NCCL.

        Returns timing dict with per-step breakdown in milliseconds.
        """
        if self._state != "SUSPENDED":
            raise RuntimeError(f"Cannot resume from state {self._state}")

        pid = os.getpid()
        timings = {}

        t0 = time.perf_counter()
        self._api.restore(pid)
        timings["restore_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        self._api.unlock(pid)
        timings["unlock_ms"] = (time.perf_counter() - t0) * 1000

        self._state = "RUNNING"

        if self._post_restore:
            t0 = time.perf_counter()
            self._post_restore()
            timings["post_restore_ms"] = (time.perf_counter() - t0) * 1000

        return timings

    def cycle(self) -> dict:
        """Full suspend + resume cycle."""
        suspend = self.suspend()
        resume = self.resume()
        return {"suspend": suspend, "resume": resume}

    @staticmethod
    def required_env() -> dict[str, str]:
        """Environment variables required for in-process checkpoint.

        Unlike MultiGPUCheckpointer.required_env(), NCCL_P2P_DISABLE
        is NOT required — the hybrid teardown/rebuild handles NVLink
        P2P state by destroying NCCL before checkpoint.
        """
        return {
            "CUDA_MODULE_LOADING": "EAGER",
            "NCCL_NVLS_ENABLE": "0",
        }
