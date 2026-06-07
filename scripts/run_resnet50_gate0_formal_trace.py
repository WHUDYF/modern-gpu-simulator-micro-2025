import ctypes

import torch
import torchvision.models as models


def main() -> None:
    torch.cuda.init()
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).cuda().eval()
    sample = torch.randn(1, 3, 224, 224, device="cuda")

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        model(sample)
    torch.cuda.synchronize()

    cudart = ctypes.CDLL("libcudart.so")
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
