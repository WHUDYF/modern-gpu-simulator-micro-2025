import ctypes

import pytest

from scripts import run_resnet50_gate0_formal_trace


def test_gate0_formal_trace_runner_falls_back_to_versioned_libcudart(monkeypatch):
    attempts = []

    class FakeCudaRuntime:
        pass

    expected = FakeCudaRuntime()

    def fake_cdll(name):
        attempts.append(name)
        if name == "libcudart.so.12":
            return expected
        raise OSError(f"missing {name}")

    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)

    assert run_resnet50_gate0_formal_trace._load_cuda_runtime() is expected
    assert attempts == ["libcudart.so", "libcudart.so.12"]


def test_gate0_formal_trace_runner_reports_all_libcudart_attempts(monkeypatch):
    def fake_cdll(name):
        raise OSError(f"missing {name}")

    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)

    with pytest.raises(OSError, match="libcudart.so.11.0"):
        run_resnet50_gate0_formal_trace._load_cuda_runtime()
