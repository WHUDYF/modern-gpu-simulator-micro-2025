# RTX 3090 处方性诊断闭环 - 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在租用的 RTX 3090 云服务器上，跑通 "真实 workload (GPT-2 decode) → trace + NCU + 模拟器 → 多维 Roofline 特征 → AI 处方 → 模拟器闭环验证" 的完整链路，验证 AI agent 能否输出可度量效果的架构修改建议。

**Architecture:** 在 3090 上同时获取 NVBit trace 和 NCU 硬件 stats，用同一 trace 跑模拟器（SM86_RTXA6000 config）建立可信基线。提取压缩特征 + distance-to-roof 指标，让 AI 输出针对具体模拟器参数的修改处方。通过"改参数 → 重跑模拟器 → 对比"形成闭环验证。

**Tech Stack:** RTX 3090 + CUDA 12.8 (or 11.4) + NVBit 1.7.6 + Nsight Compute 2025.x + Python 3.11 + protobuf + GPGPU-Sim 模拟器 + HuggingFace transformers

---

## Phase 0: 云服务器环境准备（用户主导）

### Task 0: 租用并验证 3090 云服务器

**Files:** N/A (环境准备)

- [ ] **Step 1: 选择云服务商并租用**

推荐平台对比（2026-04-07）：

| 平台 | GPU | 价格估算 | CUDA 预装 |
|------|-----|---------|----------|
| AutoDL | RTX 3090 | ¥1.5-2.5/h | 11.4 / 12.8 可选 |
| GPUMall | RTX 3090 | ¥1.5-3/h | 12.x |
| Lambda | RTX 3090 | $0.5-1/h | 11.x / 12.x |

选择 CUDA 12.8 镜像（README 确认支持）。

- [ ] **Step 2: 登录后基础检查**

```bash
nvidia-smi                                    # 确认 3090 可见
nvcc --version                                # 确认 CUDA 版本
ncu --version                                 # 确认 NCU 存在
gcc --version                                 # CUDA 11.4 需要 gcc 9
```

- [ ] **Step 3: 确认 NCU 权限**

```bash
ncu --set basic python -c "import torch; torch.cuda.init()" 2>&1 | head -5
```

若报 `ERR_NVGPUCTRPERM`，联系云服务商客服开启 GPU counter 权限（多数平台支持）。

- [ ] **Step 4: 克隆仓库**

```bash
git clone <your-repo-url> modern-gpu-simulator-micro-2025
cd modern-gpu-simulator-micro-2025
```

---

## Phase 1: 环境配置

### Task 1: 编译 NVBit tracer

**Files:**
- Use: `simulator-remodeled/util/tracer_nvbit/`

- [ ] **Step 1: 安装 NVBit 依赖**

```bash
cd simulator-remodeled
./util/tracer_nvbit/install_nvbit.sh
```

- [ ] **Step 2: 编译 tracer**

```bash
make -C ./util/tracer_nvbit/
```

- [ ] **Step 3: 验证 tracer_tool.so 存在**

```bash
ls -la ./util/tracer_nvbit/tracer_tool/tracer_tool.so
```

Expected: 文件存在，约几 MB。

### Task 2: 编译模拟器

**Files:**
- Use: `simulator-remodeled/gpu-simulator/`

- [ ] **Step 1: 设置环境**

```bash
source ./gpu-simulator/setup_environment_no_git.sh
```

- [ ] **Step 2: 编译**

```bash
make -j -C ./gpu-simulator/
```

- [ ] **Step 3: 验证可执行文件**

```bash
ls -la ./gpu-simulator/bin/release/accel-sim.out
./gpu-simulator/bin/release/accel-sim.out --help 2>&1 | head -20
```

### Task 3: 安装 Python 环境

**Files:**
- Use: `experiments/gpt2_decode/`

- [ ] **Step 1: 安装 miniconda（如果云服务器未预装）**

```bash
which conda || (wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh && bash /tmp/mc.sh -b -p $HOME/miniconda3)
```

- [ ] **Step 2: 创建 trace_gen 环境**

```bash
$HOME/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
$HOME/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
$HOME/miniconda3/bin/conda create -n trace_gen python=3.11 -y
$HOME/miniconda3/envs/trace_gen/bin/pip install torch transformers "httpx[socks]"
```

- [ ] **Step 3: 安装特征提取依赖**

```bash
$HOME/miniconda3/envs/trace_gen/bin/pip install numpy protobuf
```

- [ ] **Step 4: 验证**

```bash
$HOME/miniconda3/envs/trace_gen/bin/python -c "
import torch
from transformers import AutoModelForCausalLM
print('cuda:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0))
"
```

Expected: `cuda: True`, `device: NVIDIA GeForce RTX 3090`

### Task 4: 编译 proto 模块

**Files:**
- Use: `experiments/baseline_diagnosis/compile_proto.sh` (已有)

- [ ] **Step 1: 检查 protoc 可用**

```bash
which protoc && protoc --version
```

如果没有，安装：`apt-get install protobuf-compiler` 或从 3.21.x 源码编译。

- [ ] **Step 2: 编译 proto**

```bash
bash experiments/baseline_diagnosis/compile_proto.sh
```

- [ ] **Step 3: 验证可导入**

```bash
python3 -c "
import sys
sys.path.insert(0, 'experiments/baseline_diagnosis')
from proto_gen import trace_pb2, compressed_threadblock_pb2
print('OK')
"
```

---

## Phase 2: 建立 Roof 上限参照表

### Task 5: 在 3090 上重新跑微基准 trace

**Files:**
- Use: 现有微基准应用（`gpu-app-collection/src/cuda/GPU_Microbenchmark/`）

- [ ] **Step 1: 找到微基准二进制**

```bash
find simulator-remodeled/gpu-app-collection -name "l1_bw_32f*" -executable 2>/dev/null
find simulator-remodeled/gpu-app-collection -name "MaxFlops*" -executable 2>/dev/null
```

如果没编译好，运行：
```bash
cd simulator-remodeled
source ./gpu-app-collection/src/setup_environment
make -j -C ./gpu-app-collection/src GPU_Microbenchmark
```

- [ ] **Step 2: 用 NVBit tracer 重跑关键微基准**

```bash
./util/tracer_nvbit/run_hw_trace.py -B GPU_Microbenchmark -D 0
```

Expected: trace 写入 `hw_run/traces/device-0/<cuda-version>/`

- [ ] **Step 3: 提取压缩特征**

```bash
BENCH_ROOT=hw_run/traces/device-0/12.8/GPU_Microbenchmark
OUT_ROOT=experiments/baseline_diagnosis/results/microbench_3090

mkdir -p $OUT_ROOT
for bench in "$BENCH_ROOT"/*/; do
  name=$(basename "$bench")
  python3 experiments/baseline_diagnosis/extract_trace_features.py \
    --trace-dir "$bench/traces" \
    --output "$OUT_ROOT/${name}.json"
done
```

### Task 6: 用 NCU 采集微基准硬件 stats 建立 roof 表

**Files:**
- Create: `experiments/baseline_diagnosis/run_ncu_roof.sh`
- Create: `experiments/baseline_diagnosis/build_roof_table.py`
- Create: `experiments/baseline_diagnosis/results/roof_table.json`

- [ ] **Step 1: 写 roof 采集脚本**

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(pwd)
OUT_DIR=$PROJECT_ROOT/experiments/baseline_diagnosis/results/microbench_3090
mkdir -p "$OUT_DIR"

# 关键 roof 对应的微基准
declare -A ROOF_BENCHES=(
  [MaxFlops]="peak_flops"
  [mem_bw]="peak_hbm_bw"
  [l1_bw_32f]="peak_l1_bw"
  [l2_bw_32f]="peak_l2_bw"
  [shared_bw]="peak_shmem_bw"
  [l1_lat]="l1_latency"
  [l2_lat]="l2_latency"
  [mem_lat]="hbm_latency"
)

for bench in "${!ROOF_BENCHES[@]}"; do
  echo "Profiling $bench..."
  BINARY=$(find simulator-remodeled/gpu-app-collection/bin -name "$bench*" -executable | head -1)
  if [ -z "$BINARY" ]; then
    echo "  Binary not found for $bench, skipping"
    continue
  fi
  ncu --set full --csv --target-processes all "$BINARY" \
    > "$OUT_DIR/${bench}_ncu.csv" 2>/dev/null || echo "  NCU failed for $bench"
done
```

- [ ] **Step 2: 运行 roof 采集**

```bash
bash experiments/baseline_diagnosis/run_ncu_roof.sh
```

- [ ] **Step 3: 写 roof 表构建脚本**

```python
#!/usr/bin/env python3
"""Build roof table from microbench NCU data.

Reads microbench NCU CSVs and extracts the peak achievable values for each
dimension to form the denominator in distance-to-roof calculation.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from parse_ncu_metrics import parse_ncu_csv

ROOF_MAP = {
    "MaxFlops": {
        "metric": "compute_utilization",
        "roof_key": "peak_compute_pct",
        "description": "peak sustained compute utilization achievable",
    },
    "mem_bw": {
        "metric": "dram_throughput_pct",
        "roof_key": "peak_dram_bw_pct",
        "description": "peak sustained DRAM bandwidth achievable",
    },
    "l1_bw_32f": {
        "metric": "memory_pipe_utilization",
        "roof_key": "peak_l1_bw_pct",
        "description": "peak sustained L1 bandwidth achievable",
    },
    "l2_bw_32f": {
        "metric": "memory_throughput_pct",
        "roof_key": "peak_l2_bw_pct",
        "description": "peak sustained L2 bandwidth achievable",
    },
    "shared_bw": {
        "metric": "memory_pipe_utilization",
        "roof_key": "peak_shmem_bw_pct",
        "description": "peak sustained shared memory bandwidth achievable",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Dir with *_ncu.csv files")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    roof_table = {
        "hardware": "RTX 3090 (SM_86)",
        "source": "measured from microbenchmarks via NCU",
        "roofs": {},
    }

    for bench, config in ROOF_MAP.items():
        csv_path = os.path.join(args.input_dir, f"{bench}_ncu.csv")
        if not os.path.exists(csv_path):
            print(f"Missing: {csv_path}")
            continue

        kernels = parse_ncu_csv(csv_path)
        if not kernels:
            continue

        # Take the max across kernels (should be single kernel for microbenches)
        max_val = 0.0
        for kname, metrics in kernels.items():
            val = metrics.get(config["metric"])
            if isinstance(val, (int, float)) and val > max_val:
                max_val = val

        roof_table["roofs"][config["roof_key"]] = {
            "value": max_val,
            "unit": "percent_of_peak",
            "source_bench": bench,
            "description": config["description"],
        }

    with open(args.output, "w") as f:
        json.dump(roof_table, f, indent=2)

    print(f"Roof table written to {args.output}")
    for k, v in roof_table["roofs"].items():
        print(f"  {k}: {v['value']:.2f}% (from {v['source_bench']})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 构建 roof 表**

```bash
python3 experiments/baseline_diagnosis/build_roof_table.py \
  --input-dir experiments/baseline_diagnosis/results/microbench_3090 \
  --output experiments/baseline_diagnosis/results/roof_table.json
```

Expected: `roof_table.json` 包含各个 roof 的实测上限值。

---

## Phase 3: GPT-2 数据采集

### Task 7: 生成 GPT-2 decode trace

**Files:**
- Use: `experiments/gpt2_decode/run_trace.sh`
- Modify: conda activation 路径（指向 3090 上的 miniconda）

- [ ] **Step 1: 修改 run_trace.sh 中的 conda 路径**

检查 `run_trace.sh` 第 11 行 `eval "$($HOME/miniconda3/bin/conda shell.bash hook)"` 确保和实际路径一致。

- [ ] **Step 2: 从 ctx=128 开始跑**

```bash
cd /path/to/modern-gpu-simulator-micro-2025
CONTEXTS="128" RUNS="1" bash experiments/gpt2_decode/run_trace.sh
```

Expected: 生成 `experiments/gpt2_decode/results/model_gpt2_ctx128_gen1_run1/traces/` 目录。

- [ ] **Step 3: 验证 trace 结构**

```bash
ls -R experiments/gpt2_decode/results/model_gpt2_ctx128_gen1_run1/traces/ | head -30
cat experiments/gpt2_decode/results/model_gpt2_ctx128_gen1_run1/traces/stats.csv | head
```

Expected: 看到 `dynamic_trace.pb`、`extra_info/`、`threadblocks/device_*/stream_*/kernel_*/` 结构，stats.csv 有多个 kernel 行。

- [ ] **Step 4: 跑其他 context length（可选，耗时）**

```bash
CONTEXTS="512 1024" RUNS="1" bash experiments/gpt2_decode/run_trace.sh
```

### Task 8: 用 NCU 采集 GPT-2 硬件 stats

**Files:**
- Use: `experiments/baseline_diagnosis/run_ncu_gpt2.sh` (已有)

- [ ] **Step 1: 修改脚本中的 conda 路径（如果需要）**

- [ ] **Step 2: 跑 NCU profiling for ctx=128**

```bash
CONTEXTS="128" bash experiments/baseline_diagnosis/run_ncu_gpt2.sh
```

Expected: `experiments/baseline_diagnosis/results/gpt2/ctx128_ncu.csv` 和 `ctx128_hw.json` 生成。

- [ ] **Step 3: 验证 hw stats**

```bash
python3 -c "
import json
with open('experiments/baseline_diagnosis/results/gpt2/ctx128_hw.json') as f:
    d = json.load(f)
print('Kernels:', len(d['hardware_stats']))
for kname, metrics in list(d['hardware_stats'].items())[:3]:
    print(f'  {kname}: {len(metrics)} metrics')
"
```

Expected: 至少几十个 kernel（GPT-2 decode 每层 12 个 transformer block 会产生多个 kernel）。

### Task 9: 用模拟器重放 GPT-2 trace（校准步骤）

**Files:**
- Use: `simulator-remodeled/gpu-simulator/`
- Use: `simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM86_RTXA6000/`

- [ ] **Step 1: 定位 SM86 A6000 配置**

```bash
ls simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM86_RTXA6000/
```

Expected: 看到 `gpgpusim.config` 和 `trace.config`。

- [ ] **Step 2: 跑模拟器重放**

```bash
cd simulator-remodeled
./gpu-simulator/bin/release/accel-sim.out \
  -trace ../experiments/gpt2_decode/results/model_gpt2_ctx128_gen1_run1/traces/dynamic_trace.pb \
  -config ./gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM86_RTXA6000/gpgpusim.config \
  -config ./gpu-simulator/configs/tested-cfgs/SM86_RTXA6000/trace.config \
  > ../experiments/baseline_diagnosis/results/gpt2/ctx128_sim_baseline.log 2>&1
```

Expected: 几分钟到几十分钟跑完，log 文件写入 `ctx128_sim_baseline.log`。

**注意：** 如果报错 GPT-2 trace 格式不兼容，可能需要先跑 `run_simulations.py` 走正规路径。

- [ ] **Step 3: 提取模拟器输出的 per-kernel stats**

```bash
grep -E "kernel_name|gpu_sim_cycle|gpu_sim_insn" \
  experiments/baseline_diagnosis/results/gpt2/ctx128_sim_baseline.log \
  > experiments/baseline_diagnosis/results/gpt2/ctx128_sim_stats.txt
```

- [ ] **Step 4: 对比 sim_stats 和 hw_stats（sanity check）**

校准成功标准：sim 的 kernel 数量和 hw 一致，IPC 数量级相近（不要求精确匹配）。

如果偏差 > 10x，说明模拟器配置需要调整，联系模拟器开发者或切换 config。

---

## Phase 4: 多维 Roofline 特征组装

### Task 10: distance-to-roof 计算脚本

**Files:**
- Create: `experiments/baseline_diagnosis/compute_distance_to_roof.py`

- [ ] **Step 1: 写 distance-to-roof 脚本**

```python
#!/usr/bin/env python3
"""Compute distance-to-roof metrics for each kernel.

Input: workload hw_stats JSON + roof_table JSON
Output: per-kernel utilization for each roof dimension
"""
import argparse
import json
import os


def compute_distance(hw_stats, roof_table):
    """For each kernel, compute utilization against each roof."""
    roofs = roof_table["roofs"]
    results = {}

    for kname, metrics in hw_stats.get("hardware_stats", {}).items():
        utilization = {}

        # Compute utilization
        compute_roof = roofs.get("peak_compute_pct", {}).get("value", 100)
        if "compute_utilization" in metrics:
            utilization["compute"] = {
                "achieved_pct": metrics["compute_utilization"],
                "roof_pct": compute_roof,
                "utilization": metrics["compute_utilization"] / max(compute_roof, 1) * 100,
                "roof_source": roofs.get("peak_compute_pct", {}).get("source_bench"),
            }

        # DRAM bandwidth
        dram_roof = roofs.get("peak_dram_bw_pct", {}).get("value", 100)
        if "dram_throughput_pct" in metrics:
            utilization["hbm_bandwidth"] = {
                "achieved_pct": metrics["dram_throughput_pct"],
                "roof_pct": dram_roof,
                "utilization": metrics["dram_throughput_pct"] / max(dram_roof, 1) * 100,
                "roof_source": roofs.get("peak_dram_bw_pct", {}).get("source_bench"),
            }

        # L2 bandwidth
        l2_roof = roofs.get("peak_l2_bw_pct", {}).get("value", 100)
        if "memory_throughput_pct" in metrics:
            utilization["l2_bandwidth"] = {
                "achieved_pct": metrics["memory_throughput_pct"],
                "roof_pct": l2_roof,
                "utilization": metrics["memory_throughput_pct"] / max(l2_roof, 1) * 100,
                "roof_source": roofs.get("peak_l2_bw_pct", {}).get("source_bench"),
            }

        # Occupancy
        if "occupancy_pct" in metrics:
            utilization["occupancy"] = {
                "achieved_pct": metrics["occupancy_pct"],
                "roof_pct": 100.0,
                "utilization": metrics["occupancy_pct"],
                "roof_source": "theoretical max",
            }

        # Identify dominant bottleneck
        max_util = 0
        dominant = None
        for dim, u in utilization.items():
            if u["utilization"] > max_util:
                max_util = u["utilization"]
                dominant = dim

        results[kname] = {
            "utilizations": utilization,
            "dominant_bottleneck": dominant,
            "dominant_utilization_pct": max_util,
        }

    return {
        "workload": hw_stats.get("workload_name", "unknown"),
        "roof_source": roof_table.get("hardware", "unknown"),
        "per_kernel_distance_to_roof": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hw-stats", required=True)
    parser.add_argument("--roof-table", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.hw_stats) as f:
        hw = json.load(f)
    with open(args.roof_table) as f:
        roof = json.load(f)

    result = compute_distance(hw, roof)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Distance-to-roof written to {args.output}")
    print(f"Kernels analyzed: {len(result['per_kernel_distance_to_roof'])}")

    # Print summary
    bottleneck_counts = {}
    for k, v in result["per_kernel_distance_to_roof"].items():
        b = v.get("dominant_bottleneck", "unknown")
        bottleneck_counts[b] = bottleneck_counts.get(b, 0) + 1
    print("Bottleneck distribution:")
    for b, c in sorted(bottleneck_counts.items(), key=lambda x: -x[1]):
        print(f"  {b}: {c} kernels")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行 distance-to-roof 计算**

```bash
python3 experiments/baseline_diagnosis/compute_distance_to_roof.py \
  --hw-stats experiments/baseline_diagnosis/results/gpt2/ctx128_hw.json \
  --roof-table experiments/baseline_diagnosis/results/roof_table.json \
  --output experiments/baseline_diagnosis/results/gpt2/ctx128_distance.json
```

- [ ] **Step 3: 检视输出**

看看 GPT-2 decode 的 kernel 瓶颈分布——多少是 memory-bound、多少是 compute-bound。

### Task 11: 组装完整特征包

**Files:**
- Create: `experiments/baseline_diagnosis/assemble_full_features.py`

- [ ] **Step 1: 写组装脚本**

```python
#!/usr/bin/env python3
"""Assemble full feature package for one workload.

Combines:
- trace compression features (from extract_trace_features.py)
- hardware stats (from parse_ncu_metrics.py)
- distance-to-roof (from compute_distance_to_roof.py)
- simulator baseline stats (optional, from sim log)

Produces a single JSON per kernel with everything AI needs.
"""
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-features", required=True)
    parser.add_argument("--hw-stats", required=True)
    parser.add_argument("--distance", required=True)
    parser.add_argument("--sim-stats", help="Optional simulator baseline stats")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    def load(path):
        with open(path) as f:
            return json.load(f)

    trace = load(args.trace_features)
    hw = load(args.hw_stats)
    distance = load(args.distance)
    sim = load(args.sim_stats) if args.sim_stats and os.path.exists(args.sim_stats) else {}

    # Build per-kernel view
    full = {
        "workload": distance.get("workload", "unknown"),
        "roof_source": distance.get("roof_source"),
        "summary": {
            "total_kernels": len(distance.get("per_kernel_distance_to_roof", {})),
            "bottleneck_distribution": {},
        },
        "per_kernel_features": {},
    }

    # Compute bottleneck distribution
    for kname, kdist in distance.get("per_kernel_distance_to_roof", {}).items():
        b = kdist.get("dominant_bottleneck", "unknown")
        full["summary"]["bottleneck_distribution"][b] = \
            full["summary"]["bottleneck_distribution"].get(b, 0) + 1

    # Assemble per-kernel features
    tb_features = trace.get("threadblock_features", {})
    hw_stats = hw.get("hardware_stats", {})
    distance_stats = distance.get("per_kernel_distance_to_roof", {})

    all_kernel_names = set(hw_stats.keys()) | set(distance_stats.keys())

    for kname in all_kernel_names:
        full["per_kernel_features"][kname] = {
            "compression": tb_features.get(kname, {}),
            "hardware_stats": hw_stats.get(kname, {}),
            "distance_to_roof": distance_stats.get(kname, {}),
            "simulator_baseline": sim.get(kname, {}) if sim else None,
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(full, f, indent=2)

    print(f"Full features written to {args.output}")
    print(f"Summary: {full['summary']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行组装**

```bash
python3 experiments/baseline_diagnosis/assemble_full_features.py \
  --trace-features experiments/baseline_diagnosis/results/gpt2/ctx128.json \
  --hw-stats experiments/baseline_diagnosis/results/gpt2/ctx128_hw.json \
  --distance experiments/baseline_diagnosis/results/gpt2/ctx128_distance.json \
  --output experiments/baseline_diagnosis/results/gpt2/ctx128_full.json
```

Expected: 单一 JSON 包含 GPT-2 每个 kernel 的四种特征。

---

## Phase 5: 处方性诊断 + 闭环验证

### Task 12: 整理模拟器可修改参数清单

**Files:**
- Create: `experiments/baseline_diagnosis/simulator_param_catalog.md`

- [ ] **Step 1: 提取 A6000 config 中的关键参数**

```bash
cat simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM86_RTXA6000/gpgpusim.config \
  | grep -E "^\-gpgpu_(cache|num_sched|scheduler|dram|shader|shmem)" \
  > /tmp/sm86_params.txt
cat /tmp/sm86_params.txt
```

- [ ] **Step 2: 分类并记录参数清单**

手动整理到 `simulator_param_catalog.md`，分为：

- Cache 参数（L1/L2 大小、关联度、行大小）
- Scheduler 参数（数量、策略）
- Memory 参数（DRAM 带宽、分区数）
- Pipeline 参数（operand collector 单元数）
- Resource 参数（shared memory、寄存器文件）

每个参数记录：当前值 + 合理调整范围 + 预期影响维度。

### Task 13: 处方性诊断 prompt 模板

**Files:**
- Create: `experiments/baseline_diagnosis/prescriptive_prompt.md`

- [ ] **Step 1: 写新 prompt 模板**

```markdown
# GPU 架构处方性诊断 prompt

你是 GPU 架构分析专家。基于提供的工作负载特征包和模拟器可修改参数清单，
你的任务是输出具体的架构修改处方。

## 输入

1. **压缩特征**：workload 的行为结构（指令组成、控制流规则性、TB 一致性等）
2. **硬件性能指标**：当前 workload 在真实硬件上的实测数据
3. **Distance-to-Roof**：当前 workload 距离每个潜在瓶颈的利用率
4. **模拟器可修改参数清单**：你可以建议修改的参数边界

## 输出要求

对每个主要 kernel 或 kernel 组，产出以下结构的处方报告：

### 当前状态
- 主要瓶颈：{roof 名}，当前利用率 {X%}
- 次要瓶颈：{roof 名}，当前利用率 {Y%}
- 关键观察：{结合压缩特征的洞察}

### 处方 N
**修改内容：** 具体参数名 + 原值 → 新值
**诊断依据：** 从哪些特征看出这个修改有价值
**预期效果：**
  - 量化指标：哪个 metric 会变，预期变化方向，数量级
  - 受影响范围：整个 kernel / 特定阶段 / 特定 TB
**预期代价：** 硬件面积 / 功耗 / 其他维度
**验证方法：** 重跑模拟器后看哪个指标的前后对比，什么阈值算"处方成功"
**置信度：** HIGH / MEDIUM / LOW + 理由

## 约束

- 每条处方必须可以映射到清单中的具体参数
- 预期效果必须定量（"提升 10-20%"而不是"提升"）
- 不要建议清单之外的修改
- 如果没有足够证据支持某个修改，明确说"特征不足以提建议，需要 X 类数据"

---

## 输入数据

[INSERT FULL FEATURES JSON HERE]

## 参数清单

[INSERT SIMULATOR PARAM CATALOG HERE]
```

### Task 14: 初次处方诊断（手动对话）

**Files:**
- Create: `experiments/baseline_diagnosis/results/gpt2/ctx128_prescriptions_v1.md`

- [ ] **Step 1: 准备输入数据**

把 `ctx128_full.json` 的内容和 `simulator_param_catalog.md` 的内容，
套用 `prescriptive_prompt.md` 的模板，构造完整 prompt。

- [ ] **Step 2: 让 AI 输出处方**

在对话中把 prompt 输入，得到处方报告，保存到 `ctx128_prescriptions_v1.md`。

- [ ] **Step 3: 人工筛选**

从报告中选 1-3 条：
- 置信度 HIGH
- 修改成本低（单个参数）
- 预期效果明确

### Task 15: 模拟器闭环验证

**Files:**
- Create: `experiments/baseline_diagnosis/results/gpt2/ctx128_sim_after_rx1.log` 等
- Create: `experiments/baseline_diagnosis/results/gpt2/ctx128_closed_loop_results.md`

- [ ] **Step 1: 为每条处方创建修改版 config**

```bash
mkdir -p experiments/baseline_diagnosis/configs/
cp -r simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM86_RTXA6000 \
      experiments/baseline_diagnosis/configs/SM86_RTXA6000_rx1
# 编辑 gpgpusim.config 按处方修改参数
```

- [ ] **Step 2: 重跑模拟器**

```bash
cd simulator-remodeled
./gpu-simulator/bin/release/accel-sim.out \
  -trace ../experiments/gpt2_decode/results/model_gpt2_ctx128_gen1_run1/traces/dynamic_trace.pb \
  -config ../experiments/baseline_diagnosis/configs/SM86_RTXA6000_rx1/gpgpusim.config \
  -config ./gpu-simulator/configs/tested-cfgs/SM86_RTXA6000/trace.config \
  > ../experiments/baseline_diagnosis/results/gpt2/ctx128_sim_after_rx1.log 2>&1
```

- [ ] **Step 3: 对比前后 stats**

```python
# 提取 baseline 和 rx1 的 per-kernel IPC、cache miss、throughput
# 计算变化量
# 输出对比表
```

- [ ] **Step 4: AI 自评估**

把前后对比数据送回 AI，让它评估：
- 方向是否与预期一致
- 量级是否接近预期
- 如果不一致，原因是什么

### Task 16: 记录结果与决策

**Files:**
- Create: `experiments/baseline_diagnosis/results/gpt2/phase5_summary.md`

- [ ] **Step 1: 填写处方评估表**

对每条实际验证的处方，填 Section 4.2.4 的评估表。

- [ ] **Step 2: 决策：是否进入 Squash/Delta 阶段**

依据：
- 如果至少 1 条处方方向对且量级接近：核心方法可行 → 进入 Squash/Delta 强化
- 如果所有处方方向错：核心假设有问题 → 重新审视特征集或 prompt
- 如果处方正确但粒度太粗：Squash/Delta 正好能补 → 进入第三步

- [ ] **Step 3: 写 Phase 5 总结报告**

内容：每条处方的结果、成本收益分析、对后续方向的建议。

---

## 里程碑

- **M1（Phase 0-1 完成）**：环境就绪，能在 3090 上跑 trace 和 NCU
- **M2（Phase 2 完成）**：roof 表建立
- **M3（Phase 3 完成）**：GPT-2 完整数据采集
- **M4（Phase 4 完成）**：完整特征包组装
- **M5（Phase 5 完成）**：至少 1 条处方的闭环验证结果
- **Decision Gate**：基于 M5 结果决定是否进入 Squash/Delta

## 时间估算

- Phase 0-1（环境）：半天到 1 天
- Phase 2（roof）：半天
- Phase 3（GPT-2 数据）：1 天
- Phase 4（特征组装）：半天
- Phase 5（处方 + 闭环）：1-2 天

总计：**3-5 天**云服务器使用时间，按 ¥2/小时算约 ¥150-240。

## 失败预案

| 失败场景 | 应对 |
|---------|------|
| NVBit 不兼容云服务器的 CUDA 版本 | 换云服务器镜像；或尝试 CUDA 11.4 |
| NCU 权限无法开启 | 联系云服务商；或换平台 |
| 模拟器校准偏差太大 | 换 SM75 config 试试；或联系模拟器作者 |
| AI 处方完全在噪声水平 | 重新设计 prompt；或增加参数敏感度预筛选 |
