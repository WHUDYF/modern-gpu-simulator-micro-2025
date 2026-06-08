import ctypes
import sys
from types import SimpleNamespace

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


def test_gate0_formal_trace_runner_uses_offline_resnet_weights(monkeypatch):
    calls = []

    class FakeTorch:
        float16 = "float16"

        class cuda:
            @staticmethod
            def init():
                calls.append(("cuda.init",))

            @staticmethod
            def synchronize():
                calls.append(("cuda.synchronize",))

        class amp:
            @staticmethod
            def autocast(device, dtype):
                calls.append(("autocast", device, dtype))

                class Context:
                    def __enter__(self):
                        return None

                    def __exit__(self, *args):
                        return False

                return Context()

        @staticmethod
        def randn(*shape, device):
            calls.append(("randn", shape, device))
            return "sample"

        @staticmethod
        def no_grad():
            calls.append(("no_grad",))

            class Context:
                def __enter__(self):
                    return None

                def __exit__(self, *args):
                    return False

            return Context()

    class FakeModel:
        def cuda(self):
            calls.append(("model.cuda",))
            return self

        def eval(self):
            calls.append(("model.eval",))
            return self

        def __call__(self, sample):
            calls.append(("model", sample))
            return SimpleNamespace(shape=(1, 1000))

    class FakeModels:
        class ResNet50_Weights:
            DEFAULT = object()

        @staticmethod
        def resnet50(*, weights):
            calls.append(("resnet50", weights))
            if weights is FakeModels.ResNet50_Weights.DEFAULT:
                raise AssertionError("formal trace runner must not download pretrained weights")
            return FakeModel()

    class FakeCudaRuntime:
        @staticmethod
        def cudaProfilerStart():
            calls.append(("cudaProfilerStart",))
            return 0

        @staticmethod
        def cudaProfilerStop():
            calls.append(("cudaProfilerStop",))
            return 0

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(sys.modules, "torchvision", SimpleNamespace(models=FakeModels))
    monkeypatch.setitem(sys.modules, "torchvision.models", FakeModels)
    monkeypatch.setattr(
        run_resnet50_gate0_formal_trace,
        "_load_cuda_runtime",
        lambda: FakeCudaRuntime(),
    )

    run_resnet50_gate0_formal_trace.main()

    assert ("resnet50", None) in calls
