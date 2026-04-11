# Stage C 闭环验证 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 GPGPU-Sim 模拟器上验证 mini-transformer v4 的三条 Stage C 架构处方（C-1 寄存器文件 / C-2 HBM 带宽 / C-3 L2 容量），量化每条处方前后的 APE 变化，输出 E5_stageC_validation.md 报告。

**Architecture:** 录制 6 个代表 kernel 的 trace → 跑 baseline 模拟 → 逐一修改配置参数验证处方 → 对比 NCU 实测数据计算 APE。每次只改一个参数，用对照 kernel 验证无副作用。

**Tech Stack:** GPGPU-Sim 4.2 (accel-sim.out), NVBit tracer (accel-sim format), Python 3, CUDA 12.x

---

## 文件结构

```
experiments/baseline_diagnosis/
├── parse_ncu_v2.py                    (已有，解析 NCU CSV)
├── parse_sim_output.py                (新建，解析模拟器日志)
└── results/mini_transformer_v4/
    ├── E5_stageC_validation.md        (新建，最终报告)
    └── configs/
        ├── baseline/
        │   ├── gpgpusim.config        (从 SM86_RTX3080_TI 复制)
        │   └── trace.config           (从 SM86_RTX3080_TI 复制)
        ├── rx_C1/
        │   └── gpgpusim.config        (调整 shader_registers)
        ├── rx_C2/
        │   └── gpgpusim.config        (调整 n_mem)
        └── rx_C3/
            └── gpgpusim.config        (调整 dl2)

experiments/mini_transformer/
└── traces/                            (新建，NVBit 录制的 trace)
    ├── gemm_tiled/
    ├── attention_score/
    ├── softmax_kernel/
    ├── context_mul/
    ├── residual_add/
    └── layernorm_kernel/
```

---

## Task 1：准备模拟器配置文件

**Files:**
- Create: `experiments/baseline_diagnosis/results/mini_transformer_v4/configs/baseline/gpgpusim.config`
- Create: `experiments/baseline_diagnosis/results/mini_transformer_v4/configs/baseline/trace.config`

- [ ] **Step 1: 创建 configs 目录结构**

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025
mkdir -p experiments/baseline_diagnosis/results/mini_transformer_v4/configs/baseline
mkdir -p experiments/baseline_diagnosis/results/mini_transformer_v4/configs/rx_C1
mkdir -p experiments/baseline_diagnosis/results/mini_transformer_v4/configs/rx_C2
mkdir -p experiments/baseline_diagnosis/results/mini_transformer_v4/configs/rx_C3
```

- [ ] **Step 2: 复制 baseline 配置**

```bash
SRC=simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM86_RTX3080_TI
DST=experiments/baseline_diagnosis/results/mini_transformer_v4/configs/baseline

cp $SRC/gpgpusim.config $DST/gpgpusim.config
cp simulator-remodeled/gpu-simulator/configs/tested-cfgs/SM86_RTX3080_TI/trace.config \
   $DST/trace.config
```

- [ ] **Step 3: 创建 rx_C1 配置（gpgpu_shader_registers 验证）**

C-1 的 baseline 值已是真实值 65536。rx_C1 用于测试"若配置错误（比如 32768）APE 会如何变化"，从而证明该参数对模拟准确性的敏感性。

```bash
cp experiments/baseline_diagnosis/results/mini_transformer_v4/configs/baseline/gpgpusim.config \
   experiments/baseline_diagnosis/results/mini_transformer_v4/configs/rx_C1/gpgpusim.config

# 将 gpgpu_shader_registers 从 65536 改为 32768（缩小一半，验证敏感性）
sed -i 's/-gpgpu_shader_registers 65536/-gpgpu_shader_registers 32768/' \
   experiments/baseline_diagnosis/results/mini_transformer_v4/configs/rx_C1/gpgpusim.config
```

验证修改：
```bash
grep "gpgpu_shader_registers" \
  experiments/baseline_diagnosis/results/mini_transformer_v4/configs/rx_C1/gpgpusim.config
```
Expected: `-gpgpu_shader_registers 32768`

- [ ] **Step 4: 创建 rx_C2 配置（gpgpu_n_mem 验证）**

```bash
cp experiments/baseline_diagnosis/results/mini_transformer_v4/configs/baseline/gpgpusim.config \
   experiments/baseline_diagnosis/results/mini_transformer_v4/configs/rx_C2/gpgpusim.config

# 将 gpgpu_n_mem 从 24 改为 12（减半，预期 DRAM 吞吐下降，APE 增大）
sed -i 's/-gpgpu_n_mem 24/-gpgpu_n_mem 12/' \
   experiments/baseline_diagnosis/results/mini_transformer_v4/configs/rx_C2/gpgpusim.config
```

验证修改：
```bash
grep "gpgpu_n_mem" \
  experiments/baseline_diagnosis/results/mini_transformer_v4/configs/rx_C2/gpgpusim.config
```
Expected: `-gpgpu_n_mem 12`

- [ ] **Step 5: 创建 rx_C3 配置（L2 cache 容量验证）**

```bash
cp experiments/baseline_diagnosis/results/mini_transformer_v4/configs/baseline/gpgpusim.config \
   experiments/baseline_diagnosis/results/mini_transformer_v4/configs/rx_C3/gpgpusim.config

# 将 L2 大小从 64 sets 改为 256 sets（扩大 4x，预期 softmax DRAM 降低）
# 格式：S:sets:line_size:assoc
sed -i 's/-gpgpu_cache:dl2 S:64:128:16/-gpgpu_cache:dl2 S:256:128:16/' \
   experiments/baseline_diagnosis/results/mini_transformer_v4/configs/rx_C3/gpgpusim.config
```

验证修改：
```bash
grep "gpgpu_cache:dl2" \
  experiments/baseline_diagnosis/results/mini_transformer_v4/configs/rx_C3/gpgpusim.config
```
Expected: `-gpgpu_cache:dl2 S:256:128:16,L:B:m:L:P,A:192:96,32:0,32`

- [ ] **Step 6: Commit**

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025
git add experiments/baseline_diagnosis/results/mini_transformer_v4/configs/
git commit -m "add stageC validation configs for mini_transformer_v4"
```

---

## Task 2：在 RTX 3080 Ti 上录制 trace

**Files:**
- Create: `experiments/mini_transformer/traces/` (NVBit 输出目录)

**前置条件：** 需要 SSH 连接到 RTX 3080 Ti (117.50.75.39)

- [ ] **Step 1: 连接到 RTX 3080 Ti 并确认环境**

```bash
ssh ubuntu@117.50.75.39
cd ~/modern-gpu-simulator-micro-2025

# 确认 v4 binary 存在
ls experiments/mini_transformer/mini_transformer_v4
# Expected: 文件存在

# 确认 NVBit tracer 路径
ls simulator-remodeled/util/tracer_tool/
# Expected: tracer_tool.so 或类似文件
```

- [ ] **Step 2: 确认 accel-sim tracer 的具体路径**

```bash
find ~/modern-gpu-simulator-micro-2025/simulator-remodeled -name "tracer_tool.so" 2>/dev/null
find ~/modern-gpu-simulator-micro-2025/simulator-remodeled -name "*.so" -path "*/tracer*" 2>/dev/null
```

记录 tracer_tool.so 的实际路径，后续步骤用 `$TRACER_PATH` 指代。

- [ ] **Step 3: 录制 mini_transformer_v4 的完整 trace**

```bash
cd ~/modern-gpu-simulator-micro-2025/experiments/mini_transformer
mkdir -p traces

# 设置 tracer 环境变量（accel-sim 格式）
export TRACER_PATH=<Step 2 找到的路径>
export TRACES_FOLDER=./traces

LD_PRELOAD=$TRACER_PATH ./mini_transformer_v4
```

Expected 输出：traces/ 目录下生成 `dynamic_trace.pb` 和每个 kernel 的 threadblock trace 文件

- [ ] **Step 4: 验证 trace 文件完整性**

```bash
ls traces/
# Expected: dynamic_trace.pb  extra_info  threadblocks/

# 检查 kernel 数量
cat traces/extra_info | grep "kernel" | wc -l
# Expected: 接近 78（6层 × 13次 launch）
```

- [ ] **Step 5: 将 trace 同步回本地**

在本地机器执行：
```bash
scp -r ubuntu@117.50.75.39:~/modern-gpu-simulator-micro-2025/experiments/mini_transformer/traces \
    /home/dyf/modern-gpu-simulator-micro-2025/experiments/mini_transformer/traces
```

- [ ] **Step 6: Commit trace 元数据（不 commit .pb 大文件）**

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025
echo "experiments/mini_transformer/traces/*.pb" >> .gitignore
echo "experiments/mini_transformer/traces/threadblocks/" >> .gitignore
git add .gitignore
git add experiments/mini_transformer/traces/extra_info
git commit -m "add mini_transformer_v4 trace metadata"
```

---

## Task 3：编写模拟器输出解析脚本

**Files:**
- Create: `experiments/baseline_diagnosis/parse_sim_output.py`

- [ ] **Step 1: 编写解析脚本**

```bash
cat > /home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/parse_sim_output.py << 'EOF'
#!/usr/bin/env python3
"""Parse GPGPU-Sim log output and extract per-kernel metrics.

Reads a simulator log file, extracts key performance metrics per kernel,
and outputs a JSON file for APE comparison against NCU measurements.
"""
import re
import json
import sys
from collections import defaultdict

# Metrics to extract from simulator log
# Maps log field name -> output JSON key
METRIC_MAP = {
    "gpu_ipc":                      "sim_ipc",
    "gpu_occupancy":                "sim_occupancy_pct",
    "L1D_total_cache_hit_rate":     "sim_l1_hit_rate_pct",
    "dram_bw_util":                 "sim_dram_throughput_pct",
    "gpu_sim_cycle":                "sim_cycles",
    "gpu_sim_insn":                 "sim_insn",
}

def parse_kernel_stats(log_path):
    """Extract per-kernel stats from GPGPU-Sim log."""
    results = {}
    current_kernel = None

    with open(log_path) as f:
        for line in f:
            # Kernel launch marker
            m = re.search(r'kernel_name\s*=\s*(\S+)', line)
            if m:
                current_kernel = m.group(1)
                if current_kernel not in results:
                    results[current_kernel] = {}
                continue

            if current_kernel is None:
                continue

            # Metric lines: "metric_name = value"
            for log_key, json_key in METRIC_MAP.items():
                m = re.search(rf'{re.escape(log_key)}\s*=\s*([0-9.]+)', line)
                if m:
                    results[current_kernel][json_key] = float(m.group(1))

    return results

def short_name(full_name):
    """Extract short kernel name matching NCU output."""
    m = re.match(r'(\w+)\(', full_name)
    return m.group(1) if m else full_name.split('(')[0].strip()

def main(log_path, out_path):
    raw = parse_kernel_stats(log_path)

    # Aggregate by short name (mean across launches)
    aggregated = defaultdict(lambda: defaultdict(list))
    for full_name, metrics in raw.items():
        sname = short_name(full_name)
        for k, v in metrics.items():
            aggregated[sname][k].append(v)

    result = {}
    for sname, metrics in aggregated.items():
        result[sname] = {k: round(sum(v) / len(v), 4) for k, v in metrics.items()}

    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Parsed {len(result)} kernels -> {out_path}", file=sys.stderr)
    for k, v in result.items():
        print(f"  {k:25s}  ipc={v.get('sim_ipc','?'):6}  "
              f"occ={v.get('sim_occupancy_pct','?'):5}  "
              f"dram={v.get('sim_dram_throughput_pct','?'):5}  "
              f"l1_hit={v.get('sim_l1_hit_rate_pct','?'):5}", file=sys.stderr)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
EOF
```

- [ ] **Step 2: 编写 APE 计算脚本**

```bash
cat > /home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/compute_ape.py << 'EOF'
#!/usr/bin/env python3
"""Compute APE between simulator output and NCU measurements.

Usage:
    python3 compute_ape.py <sim_json> <ncu_json> <output_json>

sim_json: output of parse_sim_output.py
ncu_json: output of parse_ncu_v2.py (key_metrics section)
"""
import json
import sys

# NCU key -> sim key mapping
COMPARE_PAIRS = {
    "achieved_occupancy_pct":       "sim_occupancy_pct",
    "dram_throughput_pct":          "sim_dram_throughput_pct",
    "l1_hit_rate_pct":              "sim_l1_hit_rate_pct",
    "ipc_active":                   "sim_ipc",
}

TARGET_KERNELS = [
    "gemm_tiled",
    "attention_score",
    "residual_add",
    "softmax_kernel",
    "context_mul",
    "layernorm_kernel",
]

def compute_ape(ncu_val, sim_val):
    if ncu_val == 0:
        return None
    return abs(sim_val - ncu_val) / ncu_val * 100.0

def main(sim_path, ncu_path, out_path):
    with open(sim_path) as f:
        sim_data = json.load(f)
    with open(ncu_path) as f:
        ncu_raw = json.load(f)

    # ncu_raw structure: {"hardware_stats": {kernel: {"key_metrics": {...}}}}
    ncu_data = {k: v["key_metrics"] for k, v in ncu_raw["hardware_stats"].items()}

    results = {}
    for kernel in TARGET_KERNELS:
        if kernel not in ncu_data or kernel not in sim_data:
            print(f"WARNING: {kernel} missing in sim or NCU data", file=sys.stderr)
            continue

        kernel_result = {}
        for ncu_key, sim_key in COMPARE_PAIRS.items():
            ncu_val = ncu_data[kernel].get(ncu_key)
            sim_val = sim_data[kernel].get(sim_key)
            if ncu_val is None or sim_val is None:
                continue
            ape = compute_ape(ncu_val, sim_val)
            kernel_result[ncu_key] = {
                "ncu": round(ncu_val, 3),
                "sim": round(sim_val, 3),
                "ape_pct": round(ape, 1) if ape is not None else None,
            }
        results[kernel] = kernel_result

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print(f"\n{'Kernel':25s} {'Metric':30s} {'NCU':>8} {'SIM':>8} {'APE%':>7}")
    print("-" * 80)
    for kernel, metrics in results.items():
        for metric, vals in metrics.items():
            ape_str = f"{vals['ape_pct']:6.1f}%" if vals['ape_pct'] is not None else "   N/A"
            print(f"{kernel:25s} {metric:30s} {vals['ncu']:8.2f} {vals['sim']:8.2f} {ape_str}")
    print()

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
EOF
```

- [ ] **Step 3: Commit 脚本**

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025
git add experiments/baseline_diagnosis/parse_sim_output.py
git add experiments/baseline_diagnosis/compute_ape.py
git commit -m "add simulator output parser and APE computation script"
```

---

## Task 4：运行 Baseline 模拟

**Files:**
- Create: `experiments/baseline_diagnosis/results/mini_transformer_v4/sim_logs/baseline/` (模拟器日志)

**前置条件：** Task 2 完成，trace 文件已在本地

- [ ] **Step 1: 配置模拟器环境**

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled
export CUDA_INSTALL_PATH=/usr/local/cuda-12.8
source gpu-simulator/setup_environment_no_git.sh
```

Expected：无报错，`accel-sim.out` 可执行

- [ ] **Step 2: 创建日志输出目录**

```bash
mkdir -p /home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/mini_transformer_v4/sim_logs/baseline
mkdir -p /home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/mini_transformer_v4/sim_logs/rx_C1
mkdir -p /home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/mini_transformer_v4/sim_logs/rx_C2
mkdir -p /home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/mini_transformer_v4/sim_logs/rx_C3
```

- [ ] **Step 3: 运行 baseline 模拟**

```bash
TRACE=/home/dyf/modern-gpu-simulator-micro-2025/experiments/mini_transformer/traces/dynamic_trace.pb
CFG_DIR=/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/mini_transformer_v4/configs/baseline
LOG_DIR=/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/mini_transformer_v4/sim_logs/baseline
SIM=/home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/bin/release/accel-sim.out

$SIM \
  -trace $TRACE \
  -config $CFG_DIR/gpgpusim.config \
  -config $CFG_DIR/trace.config \
  2>&1 | tee $LOG_DIR/sim.log

echo "Exit code: $?"
```

Expected：模拟器运行完毕，`sim.log` 包含每个 kernel 的统计输出

- [ ] **Step 4: 解析 baseline 模拟输出**

```bash
python3 /home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/parse_sim_output.py \
  /home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/mini_transformer_v4/sim_logs/baseline/sim.log \
  /home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/mini_transformer_v4/sim_logs/baseline/sim_metrics.json
```

Expected：6 个 kernel 的指标被解析出来

- [ ] **Step 5: 计算 baseline APE**

```bash
python3 /home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/compute_ape.py \
  /home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/mini_transformer_v4/sim_logs/baseline/sim_metrics.json \
  /tmp/v4_hw_check.json \
  /home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/mini_transformer_v4/sim_logs/baseline/ape.json
```

Expected：打印 APE 汇总表，记录每个 kernel 每个指标的 baseline APE 值

- [ ] **Step 6: Commit baseline 结果**

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025
git add experiments/baseline_diagnosis/results/mini_transformer_v4/sim_logs/baseline/sim_metrics.json
git add experiments/baseline_diagnosis/results/mini_transformer_v4/sim_logs/baseline/ape.json
git commit -m "add baseline simulation results for mini_transformer_v4 stageC"
```

---

## Task 5：验证处方 C-1（寄存器文件）

**Files:**
- Create: `sim_logs/rx_C1/sim.log`, `sim_logs/rx_C1/ape.json`

- [ ] **Step 1: 运行 rx_C1 模拟**

```bash
TRACE=/home/dyf/modern-gpu-simulator-micro-2025/experiments/mini_transformer/traces/dynamic_trace.pb
CFG_DIR=/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/mini_transformer_v4/configs/rx_C1
BASE_CFG=/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/mini_transformer_v4/configs/baseline
LOG_DIR=/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/mini_transformer_v4/sim_logs/rx_C1
SIM=/home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/bin/release/accel-sim.out

$SIM \
  -trace $TRACE \
  -config $CFG_DIR/gpgpusim.config \
  -config $BASE_CFG/trace.config \
  2>&1 | tee $LOG_DIR/sim.log
```

- [ ] **Step 2: 解析并计算 APE**

```bash
python3 /home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/parse_sim_output.py \
  $LOG_DIR/sim.log $LOG_DIR/sim_metrics.json

python3 /home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/compute_ape.py \
  $LOG_DIR/sim_metrics.json \
  /tmp/v4_hw_check.json \
  $LOG_DIR/ape.json
```

- [ ] **Step 3: 验证 C-1 结论**

检查以下条件：

1. `gemm_tiled` 和 `attention_score` 的 `achieved_occupancy_pct` APE：
   - baseline APE < 10%：C-1 直接通过（65536 配置准确）
   - baseline APE > 10%：记录偏差方向，处方有效

2. `residual_add` 的 APE 变化应 < 2%（对照验证）

```bash
python3 - << 'PYEOF'
import json

with open("/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/mini_transformer_v4/sim_logs/baseline/ape.json") as f:
    base = json.load(f)
with open("/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/mini_transformer_v4/sim_logs/rx_C1/ape.json") as f:
    rx = json.load(f)

print("C-1 验证（寄存器文件）")
print("=" * 60)
for kernel in ["gemm_tiled", "attention_score"]:
    for metric in ["achieved_occupancy_pct", "ipc_active"]:
        b = base.get(kernel, {}).get(metric, {}).get("ape_pct", "N/A")
        r = rx.get(kernel, {}).get(metric, {}).get("ape_pct", "N/A")
        print(f"  {kernel:20s} {metric:30s}  baseline={b}%  rx_C1={r}%")
print("\n对照 kernel（应无变化）：")
for metric in ["achieved_occupancy_pct"]:
    b = base.get("residual_add", {}).get(metric, {}).get("ape_pct", "N/A")
    r = rx.get("residual_add", {}).get(metric, {}).get("ape_pct", "N/A")
    print(f"  residual_add {metric}  baseline={b}%  rx_C1={r}%")
PYEOF
```

- [ ] **Step 4: Commit**

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025
git add experiments/baseline_diagnosis/results/mini_transformer_v4/sim_logs/rx_C1/
git commit -m "add rx_C1 simulation results for stageC validation"
```

---

## Task 6：验证处方 C-2（HBM 带宽）

**Files:**
- Create: `sim_logs/rx_C2/sim.log`, `sim_logs/rx_C2/ape.json`

- [ ] **Step 1: 运行 rx_C2 模拟**

```bash
TRACE=/home/dyf/modern-gpu-simulator-micro-2025/experiments/mini_transformer/traces/dynamic_trace.pb
CFG_DIR=/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/mini_transformer_v4/configs/rx_C2
BASE_CFG=/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/mini_transformer_v4/configs/baseline
LOG_DIR=/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/mini_transformer_v4/sim_logs/rx_C2
SIM=/home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/bin/release/accel-sim.out

$SIM \
  -trace $TRACE \
  -config $CFG_DIR/gpgpusim.config \
  -config $BASE_CFG/trace.config \
  2>&1 | tee $LOG_DIR/sim.log
```

- [ ] **Step 2: 解析并计算 APE**

```bash
python3 /home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/parse_sim_output.py \
  $LOG_DIR/sim.log $LOG_DIR/sim_metrics.json

python3 /home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/compute_ape.py \
  $LOG_DIR/sim_metrics.json \
  /tmp/v4_hw_check.json \
  $LOG_DIR/ape.json
```

- [ ] **Step 3: 验证 C-2 结论**

```bash
python3 - << 'PYEOF'
import json

with open("/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/mini_transformer_v4/sim_logs/baseline/ape.json") as f:
    base = json.load(f)
with open("/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/mini_transformer_v4/sim_logs/rx_C2/ape.json") as f:
    rx = json.load(f)

print("C-2 验证（HBM 带宽 gpgpu_n_mem 24→12）")
print("=" * 60)
for metric in ["dram_throughput_pct", "ipc_active"]:
    b = base.get("residual_add", {}).get(metric, {}).get("ape_pct", "N/A")
    r = rx.get("residual_add", {}).get(metric, {}).get("ape_pct", "N/A")
    print(f"  residual_add {metric:30s}  baseline={b}%  rx_C2={r}%")
print("\n对照 kernel（应无变化）：")
for metric in ["dram_throughput_pct"]:
    b = base.get("gemm_tiled", {}).get(metric, {}).get("ape_pct", "N/A")
    r = rx.get("gemm_tiled", {}).get(metric, {}).get("ape_pct", "N/A")
    print(f"  gemm_tiled    {metric:30s}  baseline={b}%  rx_C2={r}%")
PYEOF
```

预期：rx_C2 中 residual_add 的 dram_throughput APE 相对 baseline 增大（n_mem 减半导致带宽模型偏差增大），证明 n_mem 对 DRAM 精度敏感。

- [ ] **Step 4: Commit**

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025
git add experiments/baseline_diagnosis/results/mini_transformer_v4/sim_logs/rx_C2/
git commit -m "add rx_C2 simulation results for stageC validation"
```

---

## Task 7：验证处方 C-3（L2 Cache 容量）

**Files:**
- Create: `sim_logs/rx_C3/sim.log`, `sim_logs/rx_C3/ape.json`

- [ ] **Step 1: 运行 rx_C3 模拟**

```bash
TRACE=/home/dyf/modern-gpu-simulator-micro-2025/experiments/mini_transformer/traces/dynamic_trace.pb
CFG_DIR=/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/mini_transformer_v4/configs/rx_C3
BASE_CFG=/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/mini_transformer_v4/configs/baseline
LOG_DIR=/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/mini_transformer_v4/sim_logs/rx_C3
SIM=/home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/bin/release/accel-sim.out

$SIM \
  -trace $TRACE \
  -config $CFG_DIR/gpgpusim.config \
  -config $BASE_CFG/trace.config \
  2>&1 | tee $LOG_DIR/sim.log
```

- [ ] **Step 2: 解析并计算 APE**

```bash
python3 /home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/parse_sim_output.py \
  $LOG_DIR/sim.log $LOG_DIR/sim_metrics.json

python3 /home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/compute_ape.py \
  $LOG_DIR/sim_metrics.json \
  /tmp/v4_hw_check.json \
  $LOG_DIR/ape.json
```

- [ ] **Step 3: 验证 C-3 结论**

```bash
python3 - << 'PYEOF'
import json

with open("/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/mini_transformer_v4/sim_logs/baseline/ape.json") as f:
    base = json.load(f)
with open("/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/mini_transformer_v4/sim_logs/rx_C3/ape.json") as f:
    rx = json.load(f)

print("C-3 验证（L2 cache S:64→S:256，容量 4x）")
print("=" * 60)
for metric in ["dram_throughput_pct", "l1_hit_rate_pct"]:
    b = base.get("softmax_kernel", {}).get(metric, {}).get("ape_pct", "N/A")
    r = rx.get("softmax_kernel", {}).get(metric, {}).get("ape_pct", "N/A")
    print(f"  softmax_kernel {metric:30s}  baseline={b}%  rx_C3={r}%")
print("\n对照 kernel（应无变化）：")
for metric in ["dram_throughput_pct"]:
    b = base.get("context_mul", {}).get(metric, {}).get("ape_pct", "N/A")
    r = rx.get("context_mul", {}).get(metric, {}).get("ape_pct", "N/A")
    print(f"  context_mul    {metric:30s}  baseline={b}%  rx_C3={r}%")
PYEOF
```

预期：rx_C3 中 softmax 的 dram_throughput APE 变化，证明 L2 容量对 softmax 的 DRAM 建模精度有影响。

- [ ] **Step 4: Commit**

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025
git add experiments/baseline_diagnosis/results/mini_transformer_v4/sim_logs/rx_C3/
git commit -m "add rx_C3 simulation results for stageC validation"
```

---

## Task 8：生成 E5 验证报告

**Files:**
- Create: `results/mini_transformer_v4/E5_stageC_validation.md`

- [ ] **Step 1: 运行报告生成脚本**

```bash
python3 - << 'PYEOF'
import json

BASE_DIR = "/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/mini_transformer_v4/sim_logs"

configs = {
    "baseline": f"{BASE_DIR}/baseline/ape.json",
    "rx_C1":    f"{BASE_DIR}/rx_C1/ape.json",
    "rx_C2":    f"{BASE_DIR}/rx_C2/ape.json",
    "rx_C3":    f"{BASE_DIR}/rx_C3/ape.json",
}

data = {}
for name, path in configs.items():
    with open(path) as f:
        data[name] = json.load(f)

# Print APE comparison table
KERNELS  = ["gemm_tiled", "attention_score", "residual_add", "softmax_kernel", "context_mul"]
METRICS  = ["achieved_occupancy_pct", "dram_throughput_pct", "l1_hit_rate_pct", "ipc_active"]

print("| Kernel | Metric | Baseline APE | rx_C1 APE | rx_C2 APE | rx_C3 APE |")
print("|--------|--------|:------------:|:---------:|:---------:|:---------:|")
for k in KERNELS:
    for m in METRICS:
        row = [k, m]
        for cfg in ["baseline", "rx_C1", "rx_C2", "rx_C3"]:
            val = data[cfg].get(k, {}).get(m, {}).get("ape_pct", "N/A")
            row.append(f"{val:.1f}%" if isinstance(val, float) else str(val))
        print("| " + " | ".join(row) + " |")
PYEOF
```

- [ ] **Step 2: 将输出整理为 E5 报告**

使用上一步的输出，手动或脚本填入以下模板，保存为 `E5_stageC_validation.md`：

报告需包含：
- APE 汇总表（baseline vs rx_C1/C2/C3）
- 每条处方的判定（有效 / 无效 / 模拟器限制）
- 对照 kernel 的副作用验证
- 与 backprop 闭环验证的对比（方法论一致性）

- [ ] **Step 3: Commit 最终报告**

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025
git add experiments/baseline_diagnosis/results/mini_transformer_v4/E5_stageC_validation.md
git commit -m "add E5 stageC validation report for mini_transformer_v4"
```

---

## 自检：Spec 覆盖确认

| Spec 要求 | 对应 Task |
|----------|----------|
| 录制 6 个代表 kernel trace | Task 2 |
| baseline 模拟 + APE 计算 | Task 4 |
| C-1 寄存器文件验证 | Task 5 |
| C-2 HBM 带宽验证 | Task 6 |
| C-3 L2 容量验证 | Task 7 |
| 对照 kernel 副作用验证 | Task 5/6/7 Step 3 |
| E5 验证报告 | Task 8 |
| configs/ 目录结构 | Task 1 |

