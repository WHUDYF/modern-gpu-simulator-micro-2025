# 微基准 AI 诊断 Sanity Check 报告

日期：2026-04-07
评估者: 丁逸夫

## 背景

本报告是 baseline diagnosis 第一步的 sanity check 产出。使用已有的
5090 微基准 trace 数据（`hw_run/traces/device-0/12.8/benchstudy-20260403/`），
通过 `extract_trace_features.py` 提取压缩特征和静态元数据，再将特征包输入
AI agent 做架构诊断。

注意：由于 NVBit 1.7.6 不支持 Blackwell SM_120，GPT-2 trace 暂未生成；
NCU 也因为 GPU 性能计数器权限问题无法在当前账号下采集硬件统计。
因此本次诊断**仅基于 trace 压缩特征和静态元数据**，没有硬件 stats 输入。

评估的核心问题：AI 仅凭 trace 侧的特征，能否产出有意义的架构诊断？

## 输入特征来源

每份诊断输入包含：
- `dynamic_trace.pb` 中的 kernel 元数据（grid/block dim、寄存器、shared memory）
- `stats.csv` 的动态指令计数
- `enhanced_execution_info.json` 的静态指令分析（opcode 分布、control bits）
- 各 threadblock `.pb` 文件的压缩格式和 warp 级统计

所有微基准在当前数据中均使用原始 `compressed_threadblock` 格式，没有
触发 v6（RLE）、v7（warp shared PC）或 v8（cross-TB delta）编码。

## 选取的代表性工作负载

| 名称 | 类型 | 目的 | grid × block |
|------|------|------|-------------|
| `l1_bw_32f` | 访存密集 | L1 带宽隔离测试 | 1 × 1024 |
| `mem_bw` | 访存密集 | HBM 带宽饱和测试 | 160 × 1024 |
| `MaxFlops` | 计算密集 | 峰值 FP32 吞吐测试 | 1 × 1024 |

---

## 1. l1_bw_32f

### 1.1 关键输入特征

- 单 TB，32 warps，1024 线程
- 40 寄存器/线程，0 shared memory
- 静态指令 440 条，动态每 warp 3234 条（动态/静态比 ≈ 7.35）
- Top opcodes:
  - `FADD` × 132
  - `LDG.E.STRONG.SM` × 129
  - `MOV` × 35, `IADD3` × 34, `IADD.64` × 33, `LOP3.LUT` × 33
- Control bits summary:
  - `stall_count`: mean=2.01, std=1.43, p50=1, p75=3, max=9
  - barrier_waits=48, yields=100, write_barriers=138, read_barriers=19

### 1.2 行为概要

- **分类**：访存密集（L1 带宽测试）
- FADD 和 LDG.E.STRONG.SM 几乎 1:1，典型的 load-compute 交替模式
- LDG 使用 `.STRONG.SM` 修饰符——访存有意瞄准 L1/SM 级缓存
- 单 TB、单 SM 的隔离测试结构，明显是微基准设计而非追求 GPU 吞吐

### 1.3 异常发现

| # | 发现 | 严重程度 |
|---|------|---------|
| 1 | 仅 1 个 TB、grid=(1,1,1)，只使用 1 个 SM，GPU 整体利用率极低。符合 L1 带宽微基准的设计意图（隔离单 SM 的 L1 行为）。 | LOW |
| 2 | write_barriers(138) 远高于 read_barriers(19)，几乎每条 LDG 都设置了 write barrier，形成 load → wait → compute 的串行依赖链。 | MEDIUM |
| 3 | yields 占比 23%（100/440），流水线设计预期 warp 在 load 等待期间让出调度，依赖 32 warps 隐藏 L1 延迟。 | MEDIUM |
| 4 | 动态/静态指令比约 7.35x，存在明显的循环展开或迭代结构。 | LOW |

### 1.4 因果假设

| # | 假设 | 置信度 |
|---|------|--------|
| 1 | write barrier 密集反映的是 FADD 依赖链必须等 LDG 完成——这是 L1 带宽测试的核心模式：尽量密集发射 load 以饱和 L1 端口。 | HIGH |
| 2 | yield 高是因为 load 延迟不可避免，编译器/硬件预期通过 warp 切换隐藏延迟。 | MEDIUM |
| 3 | stall_count 均值 2 且 p75=3，说明大部分指令等待很短（L1 hit 延迟低），偶有较长等待（barrier 同步或 bank conflict）。 | MEDIUM |

### 1.5 建议探索方向

- 在模拟器中观察 **L1 bank conflict 率**——`.STRONG.SM` 访存下 128 条 load 的地址分布决定 bank conflict 程度
- 观察 **warp scheduler 策略** 对 32 warps 隐藏 L1 延迟的效率

---

## 2. mem_bw

### 2.1 关键输入特征

- 160 个 TB，每个 32 warps，1024 线程
- 40 寄存器/线程，0 shared memory
- 静态指令 72 条（紧凑），动态每 warp 约 450 条
- 跨 160 个 TB：`instructions_per_warp_mean`: mean=450.4, std=13.2, min=424, max=457 → TB 间行为高度一致
- Top opcodes:
  - `FADD` × 16, `NOP` × 10
  - `LDC.64` × 8, `IMAD.WIDE` × 8
  - `LDG.E.128.STRONG.GPU` × 5
  - `BRA` × 3, `BAR.SYNC.DEFER_BLOCKING` × 2
- Control bits summary:
  - `stall_count`: mean=2.56, std=2.85, p50=1, p75=4, max=13
  - barrier_waits=17, yields=23, write_barriers=19, read_barriers=1

### 2.2 行为概要

- **分类**：访存密集（HBM 带宽测试）
- 大规模并行（160 TB），意图饱和所有 SM 和 HBM 通道
- 关键 load 只有 5 条 `LDG.E.128.STRONG.GPU`——128 字节宽 load，绕过 L1 走 L2/HBM
- 紧凑循环结构（72 条静态 → 450 条动态/warp，循环迭代 ≈ 6x）

### 2.3 异常发现

| # | 发现 | 严重程度 |
|---|------|---------|
| 1 | 5 条静态 LDG.E.128 在循环中被反复执行，驱动全部 HBM 流量（每条 128B × 32 线程 × 多次迭代）。经典 bandwidth benchmark 模式。 | LOW |
| 2 | 存在 BAR.SYNC.DEFER_BLOCKING × 2，说明带宽测试中有阶段性同步点（可能是先 load 再 store，或多 pass 结构）。 | LOW |
| 3 | stall_count max=13 明显高于 l1_bw（max=9），对应 HBM 访存的更高延迟。 | MEDIUM |
| 4 | 160 TB 全部 32 warps，instructions_per_warp std=13.2（波动 3%），TB 间行为极其规则。 | LOW（对压缩有意义）|

### 2.4 因果假设

| # | 假设 | 置信度 |
|---|------|--------|
| 1 | stall_count max=13 对应 HBM 延迟（数百周期），被 stall count 饱和在硬件允许的最大值。 | HIGH |
| 2 | TB 间高度一致性是因为每个 TB 处理不同地址段但执行完全相同的代码路径——经典 data-parallel 模式。 | HIGH |
| 3 | `.STRONG.GPU` 修饰符绕过 L1，L1 miss rate 在此 workload 上不是瓶颈——性能取决于 L2/HBM 带宽。 | HIGH |

### 2.5 建议探索方向

- 模拟器验证 **HBM 带宽利用率是否接近理论峰值**
- 关注 **L2 cache 行为**——虽然 `.STRONG.GPU` 倾向绕过 L1，但 L2 的 hit/miss 取决于地址分布
- TB 间高一致性意味着 Squash 机制在此 workload 上压缩率应该极高，相变边界出现在 kernel 启动/结束处

---

## 3. MaxFlops

### 3.1 关键输入特征

- 单 TB，32 warps，1024 线程
- 16 寄存器/线程（非常少），0 shared memory
- 静态指令 4136 条，动态每 warp 4122 条（动态/静态 ≈ 1.0，无循环）
- Top opcodes:
  - `FFMA` × 4096（占 99%）
  - `NOP` × 13, `LDC.64` × 5, `IMAD.WIDE` × 5
  - `STG.E` × 3, `LDG.E` × 2, `BAR.SYNC.DEFER_BLOCKING` × 2
- Control bits summary:
  - `stall_count`: **mean=3.98, std=0.33, p25=p50=p75=4, max=8**
  - barrier_waits=8, yields=4114, write_barriers=11, read_barriers=0

### 3.2 行为概要

- **分类**：计算密集（峰值 FP32 吞吐测试）
- 几乎纯 FMA kernel（FFMA 占 99%）
- 代码完全展开（动态/静态 ≈ 1.0），无循环迭代
- 极少内存操作（2 条 LDG + 3 条 STG）

### 3.3 异常发现

| # | 发现 | 严重程度 |
|---|------|---------|
| 1 | yields 占比 99.5%（4114/4136），几乎每条 FFMA 都设置了 yield 位。 | LOW |
| 2 | **stall_count 分布极其均匀**（mean=3.98, std=0.33, p25=p50=p75=4），几乎所有 FFMA 的 stall count 都是 4。 | HIGH（诊断信息量大）|
| 3 | barrier_waits 极低（8）、write_barriers 极低（11），几乎没有内存依赖等待。 | LOW（验证性发现）|
| 4 | 动态/静态比 ≈ 1.0，代码是完全展开的 FFMA 序列，无循环结构。 | LOW |

### 3.4 因果假设

| # | 假设 | 置信度 |
|---|------|--------|
| 1 | **stall_count=4 直接揭示了 FFMA 在 Blackwell SM_120 上的 issue-to-issue latency**。编译器知道 FFMA 需要 4 个周期才能 issue 下一条同 warp 指令，因此把 stall count 统一设为 4。这是一个可以直接用于校准模拟器的架构常数。 | HIGH |
| 2 | 99.5% yield 意味着 warp scheduler 可以完美地在 4 个周期内轮转到其他 warp。32 warps 在 4 周期 issue latency 下能否充分隐藏延迟取决于 SM 的 scheduler 数量（通常 4 个）。 | HIGH |
| 3 | 16 寄存器/线程 × 1024 线程 = 16K 寄存器，远低于 SM 寄存器文件容量，不会成为 occupancy 瓶颈。 | HIGH |

### 3.5 建议探索方向

- **stall_count=4 是可以用于校准模拟器的架构常数**——如果模拟器配置的 FMA issue latency 不是 4，模拟结果会偏离
- **warp scheduler 吞吐极限**：32 warps × 1 FFMA/4 cycles = 8 FFMA/cycle。假设 SM 有 4 个 warp scheduler，每个每周期 dispatch 1 条 FFMA → 4 FFMA/cycle 需要至少 16 warps 隐藏延迟，32 warps 提供 2x 余量
- 对比 NCU 的 `sm__throughput.pct_of_peak` 验证是否接近峰值 FLOPS

---

## 评估表（待人工填写）

### l1_bw_32f

| 发现 # | 类别 | 备注 |
|--------|------|------|
| 1 | 正确且非平庸 / 正确但平庸 / 错误 / 盲区 | |
| 2 | | |
| 3 | | |
| 4 | | |

### mem_bw

| 发现 # | 类别 | 备注 |
|--------|------|------|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |

### MaxFlops

| 发现 # | 类别 | 备注 |
|--------|------|------|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |

### 盲区（AI 完全没发现但你知道的问题）

| # | 已知问题 | AI 为何遗漏 | 需要什么特征才能发现 |
|---|---------|------------|--------------------|
| 1 | | | |
| 2 | | | |

### 汇总

- 总发现数：12（每个微基准 4 条）
- 正确且非平庸：__
- 正确但平庸：__
- 错误：__
- 盲区数：__
- 诊断价值评分（1-5）：__

### 结论

是否值得进入 Step 2（Squash/Delta 语义增强）？

- [ ] 是，已识别明确的盲区，压缩机制可以填补
- [ ] 否，诊断价值不足，需要重新审视方案
- [ ] 部分是，盲区存在但可能需要 Squash/Delta 之外的机制

---

## 附录：值得特别关注的发现

### A.1 stall_count 作为架构常数的直接读取

MaxFlops 的 stall_count 分布（mean=3.98, std=0.33）几乎是一个单点分布，
直接暴露了 FFMA 在 Blackwell SM_120 上的 issue latency。

这是一个**非常具体的、可以跨 workload 对比的架构指纹**。如果在其他
不同 workload 上看到相同的 stall_count 分布模式，我们可以反推该 workload
也是受同一流水线资源约束。

### A.2 跨 TB 一致性作为压缩优势的预兆

mem_bw 的 160 个 TB instructions_per_warp 波动只有 3%（std=13.2/mean=450.4），
这意味着：

- 如果升级到 compressed_kernel_v8 格式，cross-TB delta 覆盖率会极高
- Squash 机制在这类 workload 上会把整个 kernel 压缩成少量分段
- 反之，如果某个 workload 的 TB 间波动大，就是异常信号

### A.3 LDG 修饰符作为缓存路径的直接指示

- `LDG.E.STRONG.SM`（l1_bw_32f）：瞄准 L1/SM 级缓存
- `LDG.E.128.STRONG.GPU`（mem_bw）：绕过 L1 走 L2/HBM

这些修饰符是**静态可读的架构路径信息**，不需要硬件计数器就能知道
load 走了哪条缓存路径。这对于模拟器配置验证非常有用——如果模拟器
没有正确区分这些修饰符，就会在这些 workload 上偏离真实行为。

### A.4 当前数据的局限

- 没有 NCU 硬件 stats 作交叉验证——诊断假设无法被真实测量数据证实
- 所有微基准使用原始 `compressed_threadblock` 格式，没有触发 v6/v7/v8
  的高级压缩特征——cross_tb_offset_coverage、RLE 等特征全部缺失
- 单一架构（Blackwell SM_120），跨架构泛化性无法验证
