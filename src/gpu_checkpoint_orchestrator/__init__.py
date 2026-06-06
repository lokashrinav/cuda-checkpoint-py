"""Multi-GPU checkpoint/restore orchestration for any CUDA process.

Works with vLLM, TensorRT-LLM, SGLang, PyTorch training, etc.
Wraps NVIDIA's cuCheckpointProcess* driver API with multi-GPU coordination.

Three layers:
  1. CudaCheckpointAPI — direct ctypes bindings to cuCheckpointProcess* 4-step API
  2. MultiGPUCheckpointer — parallel checkpoint/restore across multiple CUDA PIDs
     (external orchestration, requires NCCL_P2P_DISABLE=1)
  3. InProcessCheckpointer — self-checkpoint from within worker processes
     (hybrid NCCL teardown/rebuild, full NVLink bandwidth)
"""

from gpu_checkpoint_orchestrator.api import CudaCheckpointAPI
from gpu_checkpoint_orchestrator.multi_gpu import MultiGPUCheckpointer
from gpu_checkpoint_orchestrator.in_process import InProcessCheckpointer
from gpu_checkpoint_orchestrator.discover import discover_cuda_pids, find_cuda_pids_for_process

__all__ = [
    "CudaCheckpointAPI",
    "MultiGPUCheckpointer",
    "InProcessCheckpointer",
    "discover_cuda_pids",
    "find_cuda_pids_for_process",
]
