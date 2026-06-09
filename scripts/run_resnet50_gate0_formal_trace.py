import ctypes
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path


def _load_cuda_runtime():
    errors = []
    for library_name in (
        "libcudart.so",
        "libcudart.so.12",
        "libcudart.so.12.0",
        "libcudart.so.11.0",
    ):
        try:
            return ctypes.CDLL(library_name)
        except OSError as exc:
            errors.append(f"{library_name}: {exc}")
    raise OSError("unable to load CUDA runtime library: " + "; ".join(errors))


def _build_offline_resnet50(models):
    try:
        return models.resnet50(weights=None)
    except TypeError:
        return models.resnet50(pretrained=False)


def _runtime_proof_hash(
    collector_session_id: str,
    collector_runtime_nonce: str,
    output_root: str,
) -> str:
    payload = {
        "collector_session_id": collector_session_id,
        "collector_runtime_nonce": collector_runtime_nonce,
        "output_root": output_root,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_runtime_proof() -> None:
    output_root = os.environ.get("GCL_RESNET50_TRACE_OUT") or os.environ.get("TRACES_FOLDER")
    session_id = os.environ.get("GCL_RESNET50_COLLECTOR_SESSION_ID")
    nonce = os.environ.get("GCL_RESNET50_COLLECTOR_RUNTIME_NONCE")
    if not output_root or not session_id or not nonce:
        return
    evidence_path = Path(output_root) / "nvbit_collection_evidence.json"
    if not evidence_path.is_file():
        return
    evidence = json.loads(evidence_path.read_text())
    evidence["collector_runtime_proof_hash"] = _runtime_proof_hash(
        session_id,
        nonce,
        output_root,
    )
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")


@contextmanager
def _cuda_autocast(torch):
    amp = getattr(torch, "amp", None)
    autocast = getattr(amp, "autocast", None)
    if autocast is not None:
        with autocast("cuda", dtype=torch.float16):
            yield
        return
    with torch.cuda.amp.autocast(dtype=torch.float16):
        yield


def main() -> None:
    import torch
    import torchvision.models as models

    torch.cuda.init()
    model = _build_offline_resnet50(models).cuda().eval()
    sample = torch.randn(1, 3, 224, 224, device="cuda")

    with torch.no_grad(), _cuda_autocast(torch):
        model(sample)
    torch.cuda.synchronize()

    cudart = _load_cuda_runtime()
    start_status = cudart.cudaProfilerStart()
    if start_status != 0:
        raise RuntimeError(f"cudaProfilerStart failed with status {start_status}")

    with torch.no_grad(), _cuda_autocast(torch):
        output = model(sample)
    torch.cuda.synchronize()

    stop_status = cudart.cudaProfilerStop()
    if stop_status != 0:
        raise RuntimeError(f"cudaProfilerStop failed with status {stop_status}")
    _write_runtime_proof()

    print("resnet50_formal_trace_done", tuple(output.shape))


if __name__ == "__main__":
    main()
