"""NCCL lifecycle helpers for use with InProcessCheckpointer.

Provides pre_checkpoint/post_restore hooks that destroy and reinitialize
NCCL process groups around the checkpoint boundary.

Usage:
    from gpu_checkpoint_orchestrator.nccl import make_nccl_hooks
    from gpu_checkpoint_orchestrator import InProcessCheckpointer

    destroy_nccl, reinit_nccl = make_nccl_hooks()
    ckpt = InProcessCheckpointer(
        pre_checkpoint=destroy_nccl,
        post_restore=reinit_nccl,
    )
"""

import os
from typing import Optional


def make_nccl_hooks(
    backend: str = "nccl",
    init_method: Optional[str] = None,
    store_factory: Optional[callable] = None,
):
    """Create matched destroy/reinit hooks for torch.distributed NCCL.

    Captures rank and world_size before destroy, replays them on reinit.
    The hooks are closures over shared state — use one pair per
    InProcessCheckpointer instance.

    Args:
        backend: torch.distributed backend (default "nccl")
        init_method: init_method for init_process_group (default "env://")
        store_factory: optional callable returning a Store for reinit
            (e.g. for TCPStore with a fixed port). If None, uses init_method.

    Returns:
        (destroy_fn, reinit_fn) tuple
    """
    import torch.distributed as dist

    _state = {}

    def destroy():
        if not dist.is_initialized():
            return

        _state["rank"] = dist.get_rank()
        _state["world_size"] = dist.get_world_size()

        dist.destroy_process_group()

    def reinit():
        if dist.is_initialized():
            return

        rank = _state.get("rank")
        world_size = _state.get("world_size")
        if rank is None or world_size is None:
            raise RuntimeError(
                "NCCL reinit called without prior destroy — "
                "rank/world_size not captured"
            )

        kwargs = {
            "backend": backend,
            "rank": rank,
            "world_size": world_size,
        }

        if store_factory is not None:
            kwargs["store"] = store_factory()
        else:
            kwargs["init_method"] = init_method or "env://"

        dist.init_process_group(**kwargs)

    return destroy, reinit


def make_raw_nccl_hooks():
    """Create hooks that use ncclCommDestroy/ncclCommInitRank directly.

    For environments without torch.distributed (e.g. raw NCCL via ctypes
    or C extensions). The caller must set the comm handles via set_comms()
    before the first suspend.

    Returns:
        Object with destroy(), reinit(), set_comms(comms), get_comms() methods.
    """

    class RawNCCLHooks:
        def __init__(self):
            self._comms = []
            self._nccl_lib = None
            self._rank = None
            self._nranks = None
            self._unique_id = None

        def set_comms(self, comms, rank, nranks, unique_id=None):
            self._comms = list(comms)
            self._rank = rank
            self._nranks = nranks
            self._unique_id = unique_id

        def get_comms(self):
            return list(self._comms)

        def _load_nccl(self):
            if self._nccl_lib is None:
                import ctypes

                for name in ["libnccl.so.2", "libnccl.so"]:
                    try:
                        self._nccl_lib = ctypes.CDLL(name)
                        return
                    except OSError:
                        continue
                raise RuntimeError("Cannot load libnccl.so")

        def destroy(self):
            self._load_nccl()
            for comm in self._comms:
                rc = self._nccl_lib.ncclCommDestroy(comm)
                if rc != 0:
                    raise RuntimeError(f"ncclCommDestroy failed: {rc}")
            self._comms = []

        def reinit(self):
            import ctypes

            self._load_nccl()
            if self._unique_id is None or self._rank is None:
                raise RuntimeError("set_comms() must be called before reinit")

            new_comm = ctypes.c_void_p()
            rc = self._nccl_lib.ncclCommInitRank(
                ctypes.byref(new_comm), self._nranks, self._unique_id, self._rank
            )
            if rc != 0:
                raise RuntimeError(f"ncclCommInitRank failed: {rc}")
            self._comms = [new_comm]

    return RawNCCLHooks()
