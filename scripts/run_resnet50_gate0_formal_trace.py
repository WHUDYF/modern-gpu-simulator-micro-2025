import ctypes


def _load_cuda_runtime():
    errors = []
    for library_name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0"):
        try:
            return ctypes.CDLL(library_name)
        except OSError as exc:
            errors.append(f"{library_name}: {exc}")
    raise OSError("unable to load CUDA runtime library: " + "; ".join(errors))


def main() -> None:
    import torch
    import torchvision.models as models

    torch.cuda.init()
    model = models.resnet50(weights=None).cuda().eval()
    sample = torch.randn(1, 3, 224, 224, device="cuda")

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        model(sample)
    torch.cuda.synchronize()

    cudart = _load_cuda_runtime()
    start_status = cudart.cudaProfilerStart()
    if start_status != 0:
        raise RuntimeError(f"cudaProfilerStart failed with status {start_status}")

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        output = model(sample)
    torch.cuda.synchronize()

    stop_status = cudart.cudaProfilerStop()
    if stop_status != 0:
        raise RuntimeError(f"cudaProfilerStop failed with status {stop_status}")

    print("resnet50_formal_trace_done", tuple(output.shape))


if __name__ == "__main__":
    main()
