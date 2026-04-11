这版设计更像“处方敏感性证明”，还不够像“闭环校准验证”。最大的问题不是流程顺序，而是目标定义、指标映射和参数隔离性还没有收紧；如果按现在的方式执行，最可能得到的是“这些参数会影响结果”，而不是“这些参数解释了 RTX 3080 Ti 上的真实偏差”。

**CORE_RISKS**
- `C-1` 当前不是严格意义上的校准，因为 baseline 已经使用了真实硬件值 `gpgpu_shader_registers 65536`，而且硬件摘要里 `gemm_tiled` 与 `attention_score` 的 `block_limit_registers` 都已经是 `6`，与设计假设一致；这更像一致性检查，不像待求解偏差源。[gpgpusim.config](/home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM86_RTX3080_TI/gpgpusim.config#L62) [mini_transformer_v4_hw.json](/home/dyf/modern-gpu-simulator-micro-2025/experiments/mini_transformer/mini_transformer_v4_hw.json#L37) [mini_transformer_v4_hw.json](/home/dyf/modern-gpu-simulator-micro-2025/experiments/mini_transformer/mini_transformer_v4_hw.json#L131)
- `C-2` 把 `gpgpu_n_mem` 当成“纯带宽旋钮”风险很高，因为该参数和 `gpgpu_n_sub_partition_per_mchannel=2`、`dl2` 切片数量、地址映射一起决定 memory hierarchy；把 `24` 改成 `12` 不只是减半带宽，也是在同时改 L2 slice 数量和分区拓扑，无法隔离 DRAM 根因。[gpgpusim.config](/home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM86_RTX3080_TI/gpgpusim.config#L54) [gpgpusim.config](/home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM86_RTX3080_TI/gpgpusim.config#L55) [gpgpusim.config](/home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM86_RTX3080_TI/gpgpusim.config#L144)
- `C-3` 的“L2 配大了所以 softmax DRAM 被低估”只是一个可疑假设，不是已证实根因；`softmax` 在硬件侧其实已有 `l2_hit_rate_pct=66.7%`，并不是简单的“完全溢出 L2”画像，工作集大于 L2 也不自动推出应降低 L2 容量。[mini_transformer_v4_hw.json](/home/dyf/modern-gpu-simulator-micro-2025/experiments/mini_transformer/mini_transformer_v4_hw.json#L248) [mini_transformer_v4_hw.json](/home/dyf/modern-gpu-simulator-micro-2025/experiments/mini_transformer/mini_transformer_v4_hw.json#L260)
- 设计文档把 RTX 3080 Ti 的内存校准称为 “HBM timing” 本身就是概念性风险。3080 Ti 是 GDDR6X，不是 HBM；如果参数范围和解释基于 HBM 心智模型，结论很容易偏掉。
- 当前成功标准默认“目标 kernel APE 下降且 control <2%”就算有效，但对全局结构参数这条过于乐观。`shader_registers`、`n_mem`、`dl2` 都不是局部旋钮，control kernel 完全可能合法地一起变化。
- 你现在的计划里实际上用的是“故意改错配置看 APE 是否变差”，例如 `65536→32768`、`24→12`、`S:64→256`；这能证明敏感性，但不能证明 baseline 真正正确，也不能证明 Delta 处方足够唯一。[2026-04-12-stageC-validation.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/superpowers/plans/2026-04-12-stageC-validation.md#L73) [2026-04-12-stageC-validation.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/superpowers/plans/2026-04-12-stageC-validation.md#L97) [2026-04-12-stageC-validation.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/superpowers/plans/2026-04-12-stageC-validation.md#L115)

**MISSING_REQUIREMENTS**
- 缺了“trace 与 NCU 数据必须来自同一 binary / 同一 launch 序列 / 同一输入规模”的可追溯要求。否则 Stage C 可能在对齐两个不同运行。
- 缺了“launch 级匹配规则”。设计写的是 `gemm_tiled_1`、`residual_add_9`、`layernorm_10` 这类代表 launch，但硬件摘要目前是按 kernel family 聚合后的 plain name，且 launch 次数是 `37/12/12`，不是文档里的 `7/2/2`。[2026-04-12-stageC-validation-design.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/superpowers/specs/2026-04-12-stageC-validation-design.md#L54) [mini_transformer_v4_hw.json](/home/dyf/modern-gpu-simulator-micro-2025/experiments/mini_transformer/mini_transformer_v4_hw.json#L142) [mini_transformer_v4_hw.json](/home/dyf/modern-gpu-simulator-micro-2025/experiments/mini_transformer/mini_transformer_v4_hw.json#L189) [mini_transformer_v4_hw.json](/home/dyf/modern-gpu-simulator-micro-2025/experiments/mini_transformer/mini_transformer_v4_hw.json#L236)
- 缺了硬件测量重复性要求。现在的 `APE delta > 5%` 被当成高于噪声，但没有 3 到 5 次重复运行、时钟锁定、方差估计，这个 5% 门槛没有证据支撑。
- 缺了 GPU clocks / power state 控制要求。你现在的 NCU 汇总里 `sm_freq_hz` 和 `dram_freq_hz` 在不同 kernel 间会漂移，若不锁频，throughput 百分比类指标的可重复性会受影响。[mini_transformer_v4_hw.json](/home/dyf/modern-gpu-simulator-micro-2025/experiments/mini_transformer/mini_transformer_v4_hw.json#L9) [mini_transformer_v4_hw.json](/home/dyf/modern-gpu-simulator-micro-2025/experiments/mini_transformer/mini_transformer_v4_hw.json#L10)
- 缺了“失败归因分流”要求。比如 baseline 若已经很好，处方应标记为“已满足，无需校准”，而不是硬要做 perturbation。
- 缺了“耦合参数联合校准”的允许条件。若 `n_mem` 改动后必须一起调整 queue / mapping / latency，当前设计没有规则说明什么时候允许从单因子切换到联合因子。
- 缺了缓存暖态语义要求。当前 config 明确 `-gpgpu_flush_l1_cache 1` 和 `-gpgpu_flush_l2_cache 1`，这会让每个 kernel 冷启动；如果硬件侧 kernel 之间存在 L2 残留，这个闭环天然不对齐。[gpgpusim.config](/home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM86_RTX3080_TI/gpgpusim.config#L127) [gpgpusim.config](/home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM86_RTX3080_TI/gpgpusim.config#L128)

**TECHNICAL_GAPS**
- `compute_throughput_pct -> gpu_ipc` 的映射没有物理闭环定义。Nsight 的 “Compute (SM) Throughput” 不是简单等价于 GPGPU-Sim 的 `gpu_ipc`，尤其在 tensor / memory stall / scheduler policy 混合存在时更不稳。
- `warp_cycles_per_issued_inst -> gpu_ipc 换算` 基本上是不充分的。这个指标是 warp 级 issue 效率，不是单个 IPC 倒数就能可靠还原；如果不写清公式，这个 APE 没有可解释性。[2026-04-12-stageC-validation-design.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/superpowers/specs/2026-04-12-stageC-validation-design.md#L73)
- `C-3` 选 `l1_hit_rate_pct` 作为关键指标不够合理。改的是 `dl2`，更直接的验证指标应优先看 `l2_hit_rate_pct`、`dram_throughput_pct`、`elapsed_cycles` 或 memory stall 类指标，而不是把 L1 命中率作为主证据。
- 当前硬件摘要里 `compute_throughput_pct`、`mem_pipes_busy_pct`，并且在 5 个 kernel 上还与 `max_bandwidth_pct` 数值完全一致，这非常可疑；在把它当主验证指标之前，我会先审一次 NCU 解析链路是否误映射了字段。[mini_transformer_v4_hw.json](/home/dyf/modern-gpu-simulator-micro-2025/experiments/mini_transformer/mini_transformer_v4_hw.json#L18) [mini_transformer_v4_hw.json](/home/dyf/modern-gpu-simulator-micro-2025/experiments/mini_transformer/mini_transformer_v4_hw.json#L23) [mini_transformer_v4_hw.json](/home/dyf/modern-gpu-simulator-micro-2025/experiments/mini_transformer/mini_transformer_v4_hw.json#L26)
- 设计说“提取 6 个代表 kernel 的 trace 文件”，但实施计划又是直接跑整份 `dynamic_trace.pb` 再按 short name 聚合。这样做并没有真正验证代表 launch，只是在 family 均值上比较。[2026-04-12-stageC-validation-design.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/superpowers/specs/2026-04-12-stageC-validation-design.md#L31) [2026-04-12-stageC-validation.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/superpowers/plans/2026-04-12-stageC-validation.md#L345)
- `gpgpu_n_mem` 的变化还会牵动总 L2 容量。如果按当前 config 粗算，`dl2 S:64:128:16` 每 slice 是 128 KB，`24 × 2` 个 slice 总计约 6 MB；改到 `n_mem=12` 会把总 L2 一起砍到约 3 MB，所以 `C-2` 和 `C-3` 在结构上并不独立。[gpgpusim.config](/home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM86_RTX3080_TI/gpgpusim.config#L54) [gpgpusim.config](/home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM86_RTX3080_TI/gpgpusim.config#L55) [gpgpusim.config](/home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM86_RTX3080_TI/gpgpusim.config#L144)

**ALTERNATIVE_DIRECTIONS**
- 把 Stage C 拆成两阶段。第一阶段只做 sensitivity，确认 Delta 处方方向是否能驱动目标指标单调变化；第二阶段再做 bounded calibration，在真实值附近小范围搜索最优配置。优点是结论更干净，代价是实验次数上升。
- 把主目标从“百分比指标 APE”改成“elapsed_cycles + occupancy + memory traffic 的加权目标”。优点是更贴近 GPGPU-Sim 能直接建模的量，代价是报告会少一点 Nsight 风格术语。
- 对 `C-2` 先用 microbench 做 memory subsystem 预校准，再回到 residual_add 做应用验证。优点是能隔离 `n_mem / timing / queue` 耦合，代价是多一套基准。
- 对 `C-3` 用 per-launch subtrace 而不是 family mean。优点是可以真正对应 `softmax_kernel` 的代表实例，代价是 trace slicing 和命名追踪要补工具。
- 对全局参数用 3 点 sweep 而不是 1 次单点 perturbation，例如 `shader_registers: 32K / 64K / 96K`、`n_mem: 20 / 24 / 28`、`dl2 sets: 48 / 64 / 80`。优点是能看单调性和局部最优，代价是总 runtime 增加。
- 如果目标是论文证据而不是工程调参，可以把 C-1/C-2/C-3 写成“causal falsification”框架：证明错误参数会系统性破坏对应 kernel，而正确邻域能恢复准确性。这样叙事上比“硬调一个值”更稳。

**QUESTIONS_FOR_USER**
- 你要验证的是“Delta 处方有因果解释力”，还是“生成一个新的 production config”？
- `C-2` 是否允许从 “HBM” 改写为 “GDDR6X/DRAM 子系统校准”，避免硬件术语本身出错？
- 你是否接受把 Stage C 的核心对象从 kernel family 均值改成具体 launch instance，使 `gemm_tiled_1` 这类标签真的可追溯？
- 你是否愿意为 NCU 数据增加重复采样，并锁定 GPU clocks？如果不愿意，`5%` 噪声门槛最好不要写成硬标准。
- 对 `C-2`/`C-3`，你是否允许联合调参？如果不允许，当前单因子假设需要明说只是一阶近似。
- 你是否接受把 `elapsed_cycles` 加入 primary metric 集合？如果不接受，Stage C 会过度依赖 proxy 指标。
- 对 cache 语义，你要的是“冷启动 kernel 对齐”，还是“真实应用连续 launch 对齐”？这会直接决定是否应保留 `flush_l2_cache=1`。

**CANDIDATE_CRITERIA**
- 建议把“处方成立”和“配置接受”拆开。处方成立只要求方向正确且可重复，配置接受则要求加权总误差改善且无明显副作用。
- `C-1` 可用更强标准：`gemm_tiled` 和 `attention_score` 必须同时满足 `block_limit_registers` 整数预测正确，且 `achieved_occupancy_pct` APE < 5%。这比只看 APE 是否下降更像真正的寄存器约束验证。
- `C-2` 应至少同时改善 `dram_throughput_pct` 和 `elapsed_cycles`，否则只靠带宽百分比下降不足以说明 memory subsystem 更准。
- `C-3` 不建议把 `l1_hit_rate_pct` 作为主判据。更合理的接受标准是 `softmax` 的 `l2_hit_rate_pct`、`dram_throughput_pct`、`elapsed_cycles` 至少两项改善，且方向一致。
- 对 control kernel，不建议固定写死 `<2%`。更稳的规则是 “control 的主要指标变化不超过 target 改善幅度的 1/3，且不把任何 control metric 推入 >30% APE 区间”。
- 建议增加单调性标准。若某处方在 3 点 sweep 中对目标 kernel 没有单调响应，就不应标记为 HIGH confidence。
- 建议增加总体验收标准。六个代表 kernel 的加权平均 APE 必须下降，并且不能出现任一 kernel 的 primary metric 从 `<30%` 恶化到 `>30%`。
- 建议增加数据质量门槛。只有在 NCU 指标映射被审过、trace/NCU/binary 对齐被证明一致之后，Stage C 结果才允许进入最终报告。

如果你愿意，我下一步可以把这份分析收敛成一版“更可执行的 Stage C revised spec”，直接给出一套更严谨的验证准则和参数 sweep 设计。
