# 代表压缩相关工作的维度总结

日期：2026-04-21

## 1. 文档目的

这份文档用于总结当前几篇关键相关工作各自引入了什么新的“压缩 / grouping 维度”，并回答一个对我们后续方法非常重要的问题：

**如果我们最终希望从 representative kernels 走到 family 与 importance ratio，那么前人已经证明了哪些维度是必要的，我们又还缺哪些维度。**

这里的“维度”不是数学维度，而是：

**一类被用来区分、组织或压缩 workload 对象的关键信息来源。**

---

## 2. 当前要对照的四篇工作

这份文档聚焦四篇最关键的前置工作：

1. `PKA`
2. `Sieve`
3. `Photon`
4. `STEM+ROOT`

它们虽然都属于 sampled GPU simulation / workload compression 大方向，但它们分别把 attention 放在不同的信息维度上。

---

## 3. 总览表

| 工作 | 它新增强调的核心维度 | 它主要解决的问题 | 它的输出对象 | 对我们的启发 |
|---|---|---|---|---|
| `PKA` | 行为特征空间（behavior feature space） | 如何从完整 workload 中挑出 representative kernels | representative kernels + clusters | representative compression 是必要前端，但 cluster 不等于 mechanism family |
| `Sieve` | 工作量尺度 / instruction-count stratification | 如何让 strata 内 execution time variance 更小 | strata + representative invocations | grouping 不能只看行为相似，还要显式控制 work-size 差异 |
| `Photon` | 在线执行路径结构（warp / basic-block / GPU BBV） | 如何在线判断应该在哪一层采样 | adaptive sampling decision | 在线执行结构本身可以成为有效特征，不必完全依赖离线 hand-crafted features |
| `STEM+ROOT` | invocation 级 runtime distribution heterogeneity | 如何处理同名 kernel 的时间分布异质性 | refined clusters + sample budgets | grouping 还必须显式考虑 runtime distribution，而不是只看结构相似 |

---

## 4. PKA：行为特征空间这一维

### 4.1 PKA 主要看什么

PKA 的核心输入是：

- global / local / shared memory 行为
- instruction count
- divergence efficiency
- thread block 数量

这些特征共同构成：

**kernel behavior feature space**

然后再通过：

- PCA
- K-means

把 kernel 压缩成少量 representative kernels。

### 4.2 它真正新增的维度是什么

PKA 真正新增强调的是：

**不能只按 kernel 名字分对象，而要在“行为特征空间”里找 representative objects。**

这一步的重要性在于：

- 它把 raw workload 变成了 behavior-organized workload
- 它让 representative compression 成为可能

### 4.3 它的边界在哪里

PKA 的 cluster 主要回答的是：

**谁能代表谁。**

它没有进一步回答：

- 谁和谁共享同一类架构机制
- 谁应该进入同一 simulator reasoning lane
- family / regime 应如何定义

### 4.4 对我们的启发

PKA 给我们的最重要启发是：

**workload 压缩必须建立在对象的行为表示之上，而不是建立在名字标签之上。**

但它同时也提醒我们：

**representative cluster 不等于 mechanism family。**

---

## 5. Sieve：工作量尺度这一维

### 5.1 Sieve 主要看什么

Sieve 重点看的是：

- instruction count

它不是把 instruction count 当作完整行为刻画，而是把它当作：

**work-size proxy**

### 5.2 它真正新增的维度是什么

Sieve 真正新增强调的是：

**grouping 不仅要看行为相似，还要看工作量尺度是否相近。**

也就是说，同一个 kernel 的 invocations 即使工作模式相似，如果：

- instruction count 差很多

那么它们被放进同一个 strata 后，execution time variance 仍然会很大。

### 5.3 它的边界在哪里

Sieve 主要回答的是：

**如何把工作量尺度差异显式切开，使 strata 内执行时间更稳定。**

它并不回答：

- 这些 strata 是否共享同一类机制
- 这些 strata 如何进入 family / regime

### 5.4 对我们的启发

Sieve 给我们的最重要启发是：

**“工作模式相近”还不够，“工作量尺度相近”同样是 grouping 的必要条件。**

这对我们后续构建 regime 非常关键，因为 regime 不能只看 mechanism，还要看：

- shape / size
- work-size

---

## 6. Photon：在线执行路径结构这一维

### 6.1 Photon 主要看什么

Photon 的核心不是离线 hand-crafted features，而是在线提取：

- warp 数量
- warp type 分布
- basic block 分布
- GPU BBV

这些信息共同构成：

**execution-path structure**

### 6.2 它真正新增的维度是什么

Photon 真正新增强调的是：

**执行路径结构本身可以成为 sampled simulation 的有效特征。**

也就是说，不必总是先做：

- silicon profiling
- hand-crafted feature extraction

也可以直接从在线执行过程中提取：

- warp / BB 结构签名

### 6.3 它的边界在哪里

Photon 主要回答的是：

**当前 kernel 应该在 kernel / warp / basic-block 哪一层做采样。**

它的输出更像：

- adaptive sampling decision

而不是：

- mechanism family
- simulator-side organization object

### 6.4 对我们的启发

Photon 给我们的最重要启发是：

**family 或 regime 的证据源不一定只能来自离线统计特征，还可以来自在线执行路径结构。**

也就是说，后续如果我们需要更强的 family 判据，也许可以考虑：

- warp type distribution
- BBV-like structure
- path regularity

---

## 7. STEM+ROOT：runtime distribution heterogeneity 这一维

### 7.1 STEM+ROOT 主要看什么

STEM+ROOT 的核心输入不是 static feature，也不是 instruction count，而是：

- invocation 级 execution time distribution
- CoV
- distribution shape / multi-modality

### 7.2 它真正新增的维度是什么

STEM+ROOT 真正新增强调的是：

**grouping 不仅有“行为结构空间”这一维，还有“runtime distribution”这一维。**

也就是说：

- 同名 kernel
- 甚至相似结构的 kernel

也可能在 invocation 级时间分布上高度异质。

### 7.3 它的边界在哪里

STEM+ROOT 主要回答的是：

- cluster 要不要继续拆
- 每个 refined cluster 该采多少 sample

它仍然服务于：

**更可信的 sampled simulation**

而不是：

- family-level tuning priority
- simulator-side mechanism interface

### 7.4 对我们的启发

STEM+ROOT 给我们的最重要启发是：

**grouping 不能只建立在“结构相似”上，还必须显式考虑 invocation 级的 runtime heterogeneity。**

这对我们非常关键，因为后续如果要推出 importance ratio，就不能只看：

- 工作模式

还必须看：

- runtime contribution
- runtime variability

---

## 8. 四篇工作合在一起意味着什么

如果把这四篇论文放在一起看，它们实际上把 sampled simulation 前端逐步补成了一个多维空间：

### 维度 1：行为特征空间

由 PKA 强调。

它告诉我们：

**对象不能只靠名字组织，要靠行为表示。**

### 维度 2：工作量尺度

由 Sieve 强调。

它告诉我们：

**对象不能只靠结构相似组织，还要控制 work-size 差异。**

### 维度 3：在线执行路径结构

由 Photon 强调。

它告诉我们：

**对象的行为表示不一定都要离线构造，执行路径本身也可成为 signature。**

### 维度 4：runtime distribution heterogeneity

由 STEM+ROOT 强调。

它告诉我们：

**对象的稳定性不能只靠静态结构判断，还要看 invocation 级运行时间分布。**

---

## 9. 这四个维度和我们工作的关系

如果用一句话总结，这四篇工作共同说明：

**representative compression 不是单维问题，而是一个多维 workload organization 问题。**

但即便如此，它们仍然主要停在：

- representative kernel selection
- sampled simulation budgeting
- adaptive sampling control

它们还没有继续统一回答：

- representative kernels 如何继续变成 family
- family 如何继续变成 representative execution regime
- family / regime 如何进一步推出 importance ratio
- importance ratio 如何服务 simulator tuning priority

所以这四篇工作的共同意义在于：

**它们帮我们确认了前端压缩必须考虑哪些维度，但后端的 family / importance 层仍未被显式建立。**

---

## 10. 对我们后续方法最直接的启发

如果要把这些工作转化成我们的前置认知，当前最合理的结论是：

### 10.1 family 不能只靠单一特征定义

至少不能只靠：

- kernel name
- 单一 similarity score

### 10.2 regime 必须同时考虑机制与尺度

这一步明显受 Sieve 启发。

### 10.3 importance ratio 不能只靠工作模式推出

它至少应结合：

- 行为结构
- 工作量尺度
- runtime distribution
- tuning sensitivity

### 10.4 后续方法应把前端多维 compression 结果统一变成 simulator-side structured objects

这就是我们真正补的那一层。

---

## 11. 当前阶段的简短结论

如果把这份文档压成最短形式，可以写成：

1. `PKA` 让我们看到行为特征空间是 representative compression 的基础。
2. `Sieve` 让我们看到工作量尺度同样是 grouping 的必要维度。
3. `Photon` 让我们看到在线执行路径结构可以直接成为 sampled simulation 的特征基础。
4. `STEM+ROOT` 让我们看到 runtime distribution heterogeneity 是 grouping 稳定性的关键维度。
5. 四篇工作共同说明：前端 compression 已经是多维的，而我们后续的 family / importance 层必须吸收这些维度，才能真正从 representative kernels 走到 simulator tuning priorities。
