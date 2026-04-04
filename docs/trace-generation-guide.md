# AI 模型推理 Trace 生成指南

本文档用于在 RTX 5090 机器上通过 NVBit tracer 生成三个 AI 模型的推理 trace，供 GPU 模拟器验证使用。

## 环境前提

以下已确认就绪，无需操作：
- CUDA 已安装
- NVBit tracer 已编译（位于 `simulator-remodeled/util/tracer_nvbit/`）

## Step 1：安装 Python 环境和依赖

```bash
# 如果没有 conda/pip，先安装 miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

# 创建独立环境
conda create -n trace_gen python=3.10 -y
conda activate trace_gen

# 安装 PyTorch（根据实际 CUDA 版本选择，以下以 CUDA 12.x 为例）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 安装 HuggingFace transformers
pip install transformers
```

验证安装：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

预期输出：`True NVIDIA GeForce RTX 5090` 或类似。

## Step 2：创建推理脚本

创建文件 `run_inference.py`：

```python
import torch
import sys
import os

def run_resnet50():
    """ResNet-50 CNN inference, FP16, random input"""
    import torchvision.models as models

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).cuda().eval()
    dummy_input = torch.randn(1, 3, 224, 224).cuda()

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        output = model(dummy_input)

    print(f"ResNet-50 output shape: {output.shape}")


def run_bert_base():
    """BERT-base Transformer Encoder inference, FP16, random token ids"""
    from transformers import BertModel

    model = BertModel.from_pretrained("bert-base-uncased").cuda().eval()
    dummy_input = torch.randint(0, 30522, (1, 128)).cuda()  # vocab_size=30522
    attention_mask = torch.ones(1, 128, dtype=torch.long).cuda()

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        output = model(input_ids=dummy_input, attention_mask=attention_mask)

    print(f"BERT-base output shape: {output.last_hidden_state.shape}")


def run_gpt2_small():
    """GPT-2 small Transformer Decoder inference, FP16, random token ids"""
    from transformers import GPT2Model

    model = GPT2Model.from_pretrained("gpt2").cuda().eval()
    dummy_input = torch.randint(0, 50257, (1, 128)).cuda()  # vocab_size=50257

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        output = model(input_ids=dummy_input)

    print(f"GPT-2 output shape: {output.last_hidden_state.shape}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("resnet50", "bert", "gpt2"):
        print("Usage: python run_inference.py <resnet50|bert|gpt2>")
        sys.exit(1)

    # Warmup: ensure CUDA context is initialized before tracing
    torch.cuda.init()
    torch.zeros(1).cuda()

    model_name = sys.argv[1]
    if model_name == "resnet50":
        run_resnet50()
    elif model_name == "bert":
        run_bert_base()
    elif model_name == "gpt2":
        run_gpt2_small()

    # Sync to ensure all GPU operations complete
    torch.cuda.synchronize()
    print(f"Done: {model_name}")
```

## Step 3：预下载模型权重

在 trace 生成之前先下载模型，避免 NVBit tracer 捕获下载过程中的无关 CUDA 操作：

```bash
conda activate trace_gen

python -c "import torchvision.models as m; m.resnet50(weights=m.ResNet50_Weights.DEFAULT)"
python -c "from transformers import BertModel; BertModel.from_pretrained('bert-base-uncased')"
python -c "from transformers import GPT2Model; GPT2Model.from_pretrained('gpt2')"
```

## Step 4：验证推理脚本可正常运行

不挂载 NVBit 的情况下，先确认三个模型都能正常推理：

```bash
conda activate trace_gen

python run_inference.py resnet50
python run_inference.py bert
python run_inference.py gpt2
```

每个都应输出对应的 output shape 和 "Done"。如果报错，先解决再继续。

## Step 5：使用 NVBit Tracer 生成 Trace

NVBit tracer 通过 `LD_PRELOAD` 注入，捕获所有 CUDA kernel 执行。

```bash
conda activate trace_gen

# 设定项目根目录（根据实际路径修改）
export PROJECT_ROOT=/path/to/modern-gpu-simulator-micro-2025
export TRACER_DIR=$PROJECT_ROOT/simulator-remodeled/util/tracer_nvbit
export TRACER_TOOL=$TRACER_DIR/tracer_tool/tracer_tool.so

# 创建输出目录
mkdir -p traces/resnet50 traces/bert traces/gpt2
```

逐个模型生成 trace：

### ResNet-50

```bash
cd traces/resnet50
LD_PRELOAD=$TRACER_TOOL python ../../run_inference.py resnet50
cd ../..
```

### BERT-base

```bash
cd traces/bert
LD_PRELOAD=$TRACER_TOOL python ../../run_inference.py bert
cd ../..
```

### GPT-2 small

```bash
cd traces/gpt2
LD_PRELOAD=$TRACER_TOOL python ../../run_inference.py gpt2
cd ../..
```

每个模型运行完毕后，对应目录下应出现：
- `traces/` 子目录，包含 `.pb` 文件（per-threadblock 动态 trace）
- `extra_info/enhanced_execution_info.json`（静态指令元数据）
- `stats.csv`

## Step 6：验证 Trace 完整性

```bash
echo "=== ResNet-50 ==="
find traces/resnet50 -name "*.pb" | wc -l
du -sh traces/resnet50

echo "=== BERT-base ==="
find traces/bert -name "*.pb" | wc -l
du -sh traces/bert

echo "=== GPT-2 ==="
find traces/gpt2 -name "*.pb" | wc -l
du -sh traces/gpt2

echo "=== Total ==="
du -sh traces/
```

预期：
- 每个模型目录下有数百到数千个 .pb 文件
- ResNet-50：预估 200-500MB
- BERT-base：预估 500MB-1GB
- GPT-2 small：预估 300-800MB
- 总计：预估 1-2.5GB

如果某个模型的 trace 为空（0 个 .pb 文件），说明 NVBit tracer 未正确捕获，需要检查：
1. `LD_PRELOAD` 路径是否正确
2. tracer_tool.so 是否与当前 CUDA 版本兼容
3. 是否有权限问题

## Step 7：打包

```bash
# 打包所有 trace 和推理脚本
tar czf ai_inference_traces.tar.gz traces/ run_inference.py

# 查看打包后大小
ls -lh ai_inference_traces.tar.gz
```

将 `ai_inference_traces.tar.gz` 传回开发机器。

## Step 8：传回后验证

在开发机器上解压并检查：

```bash
tar xzf ai_inference_traces.tar.gz

# 确认结构完整
find traces/ -name "*.pb" | wc -l
find traces/ -name "*.json" | wc -l
du -sh traces/*/
```

## 注意事项

1. **NVBit 与 CUDA 版本兼容性**：NVBit tracer 需要与 CUDA driver 版本匹配。如果 tracer 报错，检查 NVBit 版本是否支持当前 driver。

2. **PyTorch CUDA 初始化 kernel**：PyTorch 启动时会执行一些初始化 kernel（如 cuBLAS handle 创建）。这些也会被 NVBit 捕获。这是正常的，模拟器在 kernel 过滤阶段会处理。

3. **磁盘空间**：确保生成 trace 的磁盘有至少 **5GB** 可用空间（留余量）。

4. **运行时间**：每个模型的 trace 生成可能需要 5-30 分钟（NVBit instrumentation 开销较大）。总计预留 1-2 小时。

5. **如果 trace 过大（单模型超过 3GB）**：可能是 PyTorch 初始化 kernel 过多。可尝试添加环境变量过滤：
   ```bash
   # 只 trace 特定 kernel（具体过滤方式取决于 tracer 版本）
   export DYNAMIC_KERNEL_LIMIT_END=100
   ```
   或联系开发机器这边讨论处理方案。
