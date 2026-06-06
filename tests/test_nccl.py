"""Tests for gpu_checkpoint_orchestrator.nccl — NCCL lifecycle hooks."""

import sys
from unittest.mock import MagicMock, patch
import pytest


class TestMakeNCCLHooks:

    def _make_hooks(self, mock_dist, **kwargs):
        """Create hooks with mock_dist injected via sys.modules patch."""
        old_torch = sys.modules.get("torch")
        old_dist = sys.modules.get("torch.distributed")
        mock_torch = MagicMock()
        mock_torch.distributed = mock_dist
        try:
            sys.modules["torch"] = mock_torch
            sys.modules["torch.distributed"] = mock_dist

            if "gpu_checkpoint_orchestrator.nccl" in sys.modules:
                del sys.modules["gpu_checkpoint_orchestrator.nccl"]
            from gpu_checkpoint_orchestrator.nccl import make_nccl_hooks
            return make_nccl_hooks(**kwargs)
        finally:
            if old_torch is not None:
                sys.modules["torch"] = old_torch
            elif "torch" in sys.modules:
                del sys.modules["torch"]
            if old_dist is not None:
                sys.modules["torch.distributed"] = old_dist
            elif "torch.distributed" in sys.modules:
                del sys.modules["torch.distributed"]
            if "gpu_checkpoint_orchestrator.nccl" in sys.modules:
                del sys.modules["gpu_checkpoint_orchestrator.nccl"]

    def test_destroy_captures_rank_and_world_size(self):
        mock_dist = MagicMock()
        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 3
        mock_dist.get_world_size.return_value = 8

        destroy, _ = self._make_hooks(mock_dist)
        destroy()
        mock_dist.destroy_process_group.assert_called_once()

    def test_reinit_replays_captured_state(self):
        mock_dist = MagicMock()
        mock_dist.is_initialized.side_effect = [True, False]
        mock_dist.get_rank.return_value = 1
        mock_dist.get_world_size.return_value = 4

        destroy, reinit = self._make_hooks(mock_dist)
        destroy()
        reinit()

        init_call = mock_dist.init_process_group.call_args
        assert init_call[1]["rank"] == 1
        assert init_call[1]["world_size"] == 4
        assert init_call[1]["backend"] == "nccl"

    def test_reinit_without_destroy_raises(self):
        mock_dist = MagicMock()
        mock_dist.is_initialized.return_value = False

        _, reinit = self._make_hooks(mock_dist)
        with pytest.raises(RuntimeError, match="rank/world_size not captured"):
            reinit()

    def test_destroy_skips_if_not_initialized(self):
        mock_dist = MagicMock()
        mock_dist.is_initialized.return_value = False

        destroy, _ = self._make_hooks(mock_dist)
        destroy()
        mock_dist.destroy_process_group.assert_not_called()

    def test_reinit_skips_if_already_initialized(self):
        mock_dist = MagicMock()
        mock_dist.is_initialized.side_effect = [True, True]
        mock_dist.get_rank.return_value = 0
        mock_dist.get_world_size.return_value = 2

        destroy, reinit = self._make_hooks(mock_dist)
        destroy()
        reinit()
        mock_dist.init_process_group.assert_not_called()

    def test_custom_backend(self):
        mock_dist = MagicMock()
        mock_dist.is_initialized.side_effect = [True, False]
        mock_dist.get_rank.return_value = 0
        mock_dist.get_world_size.return_value = 2

        destroy, reinit = self._make_hooks(mock_dist, backend="gloo")
        destroy()
        reinit()

        init_call = mock_dist.init_process_group.call_args
        assert init_call[1]["backend"] == "gloo"

    def test_store_factory(self):
        mock_dist = MagicMock()
        mock_dist.is_initialized.side_effect = [True, False]
        mock_dist.get_rank.return_value = 0
        mock_dist.get_world_size.return_value = 2

        mock_store = MagicMock()
        factory = MagicMock(return_value=mock_store)

        destroy, reinit = self._make_hooks(mock_dist, store_factory=factory)
        destroy()
        reinit()

        factory.assert_called_once()
        init_call = mock_dist.init_process_group.call_args
        assert init_call[1]["store"] is mock_store
        assert "init_method" not in init_call[1]
