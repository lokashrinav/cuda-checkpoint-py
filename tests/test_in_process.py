"""Tests for gpu_checkpoint_orchestrator.in_process — self-checkpoint with hooks."""

import os
from unittest.mock import MagicMock, patch, call
import pytest


class TestInProcessCheckpointer:

    def _make_ckpt(self, pre_checkpoint=None, post_restore=None):
        with patch("ctypes.CDLL") as mock_cdll:
            mock_lib = MagicMock()
            mock_cdll.return_value = mock_lib
            for name in ["Lock", "Checkpoint", "Restore", "Unlock"]:
                fn = MagicMock()
                fn.return_value = 0
                setattr(mock_lib, f"cuCheckpointProcess{name}", fn)

            from gpu_checkpoint_orchestrator.in_process import InProcessCheckpointer
            ckpt = InProcessCheckpointer(
                pre_checkpoint=pre_checkpoint,
                post_restore=post_restore,
            )
            return ckpt, mock_lib

    def test_suspend_calls_lock_then_checkpoint(self):
        ckpt, lib = self._make_ckpt()
        result = ckpt.suspend()
        lib.cuCheckpointProcessLock.assert_called_once()
        lib.cuCheckpointProcessCheckpoint.assert_called_once()
        assert "lock_ms" in result
        assert "checkpoint_ms" in result

    def test_suspend_uses_own_pid(self):
        ckpt, lib = self._make_ckpt()
        ckpt.suspend()
        pid = os.getpid()
        lock_call = lib.cuCheckpointProcessLock.call_args
        assert lock_call[0][0] == pid

    def test_resume_calls_restore_then_unlock(self):
        ckpt, lib = self._make_ckpt()
        ckpt.suspend()
        result = ckpt.resume()
        lib.cuCheckpointProcessRestore.assert_called_once()
        lib.cuCheckpointProcessUnlock.assert_called_once()
        assert "restore_ms" in result
        assert "unlock_ms" in result

    def test_pre_checkpoint_hook_called_before_lock(self):
        order = []
        hook = MagicMock(side_effect=lambda: order.append("hook"))
        ckpt, lib = self._make_ckpt(pre_checkpoint=hook)
        lib.cuCheckpointProcessLock.side_effect = lambda *a: order.append("lock") or 0
        ckpt.suspend()
        assert order == ["hook", "lock"]

    def test_post_restore_hook_called_after_unlock(self):
        order = []
        hook = MagicMock(side_effect=lambda: order.append("hook"))
        ckpt, lib = self._make_ckpt(post_restore=hook)
        lib.cuCheckpointProcessUnlock.side_effect = lambda *a: order.append("unlock") or 0
        ckpt.suspend()
        ckpt.resume()
        assert order == ["unlock", "hook"]

    def test_state_tracking(self):
        ckpt, _ = self._make_ckpt()
        assert ckpt.state == "RUNNING"
        ckpt.suspend()
        assert ckpt.state == "SUSPENDED"
        ckpt.resume()
        assert ckpt.state == "RUNNING"

    def test_double_suspend_raises(self):
        ckpt, _ = self._make_ckpt()
        ckpt.suspend()
        with pytest.raises(RuntimeError, match="Cannot suspend"):
            ckpt.suspend()

    def test_resume_without_suspend_raises(self):
        ckpt, _ = self._make_ckpt()
        with pytest.raises(RuntimeError, match="Cannot resume"):
            ckpt.resume()

    def test_checkpoint_failure_unlocks(self):
        ckpt, lib = self._make_ckpt()
        lib.cuCheckpointProcessCheckpoint.return_value = 801
        with pytest.raises(RuntimeError, match="rc=801"):
            ckpt.suspend()
        lib.cuCheckpointProcessUnlock.assert_called_once()

    def test_cycle_returns_both(self):
        ckpt, _ = self._make_ckpt()
        result = ckpt.cycle()
        assert "suspend" in result
        assert "resume" in result
        assert "lock_ms" in result["suspend"]
        assert "restore_ms" in result["resume"]

    def test_required_env_no_p2p_disable(self):
        from gpu_checkpoint_orchestrator.in_process import InProcessCheckpointer
        env = InProcessCheckpointer.required_env()
        assert env["CUDA_MODULE_LOADING"] == "EAGER"
        assert env["NCCL_NVLS_ENABLE"] == "0"
        assert "NCCL_P2P_DISABLE" not in env

    def test_no_hooks_works(self):
        ckpt, _ = self._make_ckpt()
        result = ckpt.cycle()
        assert "pre_checkpoint_ms" not in result["suspend"]
        assert "post_restore_ms" not in result["resume"]

    def test_timings_are_positive(self):
        hook = MagicMock()
        ckpt, _ = self._make_ckpt(pre_checkpoint=hook, post_restore=hook)
        result = ckpt.cycle()
        for v in result["suspend"].values():
            assert v >= 0
        for v in result["resume"].values():
            assert v >= 0
