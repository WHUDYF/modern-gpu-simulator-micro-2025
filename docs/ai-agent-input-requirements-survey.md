# AI Agent GPU Architecture Diagnostics: Input Requirements Survey

Date: 2026-04-06

## Purpose

This document surveys existing research on what information AI agents need to
perform GPU architecture diagnosis and optimization. It serves as the reference
baseline for our subsequent work: designing a trace compression pipeline that
simultaneously extracts semantic features for AI agent consumption.

Our core question: **what key data should we extract during trace compression
to serve as effective AI agent input?**

---

## Part 1: Landscape of Existing Systems

### 1.1 Opal (2025) — LLM + Profiling Analytics for Kernel Optimization

**Source:** arXiv:2510.00932

**What it does:** Opal connects hardware performance counter insights with
Roofline analysis, feeds them to an LLM, and generates kernel optimization
recommendations.

**What the AI consumes:**
- Hardware performance counters: FLOPS utilization, HBM/L2/L1/LDS bandwidth,
  cache hit rates
- Roofline classification: compute-bound vs memory-bound, distance to
  theoretical peak
- PC Sampling data: stall reasons (memory, execution, barrier)
- Source code of the target kernel

**Key finding:** Even a single source of performance information yields
speedups in 98.5% of cases (19-52% average). This proves that AI does not
need exhaustive data to produce value — focused, well-structured input is
sufficient.

**Limitation:** Opal operates on real hardware profiling data, not simulator
traces. It cannot see internal microarchitectural state (warp instruction
flows, per-TB memory addresses, control bits).

### 1.2 Chopper (2025) — Multi-Level GPU Characterization

**Source:** arXiv:2512.08242

**What it does:** Chopper collects GPU kernel traces and hardware performance
counters, then aligns them across multiple granularities: kernel, operation,
layer, phase, iteration, and GPU.

**What the AI consumes (multi-level hierarchy):**

```
Level 1: Kernel        — individual kernel execution metrics
Level 2: Operation     — one or more kernels forming a logical operation
Level 3: Layer         — neural network layer granularity
Level 4: Phase         — forward / backward / optimizer
Level 5: Iteration     — one training step
Level 6: GPU / Node    — system-wide view
```

At each level: compute utilization, memory bandwidth, cache behavior,
communication overhead, frequency (DVFS) effects.

**Key finding:** The largest performance loss in LLM training comes from
frequency overhead (DVFS) — a cross-level phenomenon invisible at single-kernel
granularity. **Multi-level alignment is essential for discovering system-level
bottlenecks.**

**Limitation:** Chopper requires serialized hardware counter collection (only
2-3 counters at a time), making profiling expensive. It does not perform
semantic compression or abstraction.

### 1.3 TAAF (ICSE 2026) — Trace Abstraction via Knowledge Graphs + LLM

**Source:** arXiv:2601.02632

**What it does:** TAAF transforms raw OS kernel traces (34 million events)
into a time-indexed knowledge graph. LLM interprets query-specific subgraphs
to answer natural-language diagnostic questions.

**Three-layer architecture:**
1. Temporal state system: summarizes millions of events into structured,
   time-indexed transitions
2. Query-specific knowledge graph: organizes entities (threads, CPUs,
   resources) and relationships
3. LLM interpretation layer: reads graph subsets, answers questions

**Key finding:** Accuracy improves by 31.2% compared to feeding raw trace
data, especially on multi-hop and causal reasoning tasks.

**Limitation:** TAAF operates on OS-level traces (Linux kernel), not GPU
traces. The knowledge graph construction is query-driven, which means the
abstraction changes depending on what you ask.

### 1.4 SwizzlePerf (2025) — Hardware-Aware LLM for Memory Optimization

**Source:** arXiv:2508.20258

**What the AI consumes:**
- Memory access patterns of the target kernel
- GPU architecture specifications (XCD topology, L2 partitioning)
- Filtered profiling logs (L2 hit rate, off-chip traffic)
- Historical performance reflections from previous iterations

**Key finding:** By giving LLMs explicit hardware-awareness (architecture
specs + profiling data), they can generate spatial optimizations (swizzling
formulas) that cut off-chip memory traffic.

### 1.5 AutoKernel (2025-2026) — Iterative Agent-Driven Kernel Search

**Source:** arXiv:2603.21331

**What the AI consumes:**
- Model-level profiling: which kernels dominate total runtime
- Per-kernel correctness and throughput measurements
- Iterative feedback: edit → benchmark → keep/revert

**Key finding:** 70% of total execution time typically concentrates in a
single kernel. Focusing optimization on hot kernels ensures efforts are
well-targeted.

### 1.6 Omniwise (2025) — LLM-Predicted Performance Counters

**Source:** arXiv:2506.20886

**What it does:** Uses LLMs to predict GPU performance counters from code,
without actually running the kernel. Identifies computation motifs as
intermediate representation.

**Key finding:** LLMs can recognize high-level computation patterns and map
them to expected hardware behavior, dramatically reducing profiling cost.

### 1.7 KPerfIR (OSDI 2025) — Compiler-Centric Semantic Profiling

**Source:** USENIX OSDI 2025

**What it does:** Integrates profiling at the MLIR/LLVM IR level, preserving
high-level program semantics during instrumentation.

**Key finding:** Profiling at a higher semantic level (MLIR dialects) retains
information that is lost when profiling at the SASS/PTX level alone.

---

## Part 2: What AI Agents Need — Consolidated View

Across all surveyed systems, the information AI agents need falls into four
categories:

### Category 1: Bottleneck Classification

**What:** Is this kernel compute-bound, memory-bound, or latency-bound?
How far from the theoretical peak?

**How it is obtained today:** Roofline model from hardware counters (Opal),
or LLM prediction from code (Omniwise).

**Our trace can provide:** Simulator stats directly give IPC, cache miss
rates, memory throughput, compute utilization. This is the easiest category
— we already have it.

### Category 2: Behavioral Segmentation

**What:** How does the workload divide into distinct behavioral phases?
Where do transitions happen? What characterizes each phase?

**How it is obtained today:** Multi-level trace alignment (Chopper). Requires
manual definition of granularity levels.

**Our trace can provide:** This is currently **missing** from GPU trace
output. This is exactly what Squash can fill — temporal merging naturally
produces behavioral segments, and merge break points mark phase transitions.

### Category 3: Anomaly Detection

**What:** Which units (warps, threadblocks, kernels) deviate from the
group pattern? By how much? In which dimensions?

**How it is obtained today:** Statistical analysis over hardware counters.
No existing system does this systematically on GPU traces.

**Our trace can provide:** Partially. Cross-TB delta in `compressed_kernel_v8`
reveals TB-level outliers (those requiring `address_override` or
`is_full_encoding`). Warp diff entries reveal warp-level divergence. But the
current representation only gives **counts**, not the **content** of
deviations. Delta mechanism can enhance this.

### Category 4: Causal Chain

**What:** A caused B, B caused C. Stall X was triggered by cache miss Y,
which was caused by access pattern Z.

**How it is obtained today:** Knowledge graph construction (TAAF) on OS
traces. **No existing system does this for GPU traces.**

**Our trace can provide:** Not directly. Current trace captures events but
not causality. This is the hardest category and likely requires a new
mechanism beyond Squash/Batch/Delta. However, control bits (stall count,
barrier wait) partially encode local causal information — "this instruction
waited because of that barrier."

---

## Part 3: Insights for Our Design — What to Extract During Compression

Based on the survey, here is our recommended feature extraction plan,
organized by compression mechanism and mapped to the four categories above.

### 3.1 Features from Squash (Temporal Merging)

Squash serves **Category 2: Behavioral Segmentation**.

| Feature to Extract | Description | AI Interpretation |
|--------------------|-------------|-------------------|
| `segment_boundaries` | List of time points where merge breaks | Phase transition points in the workload |
| `segment_lengths` | Duration (in kernels or TBs) of each merged segment | Stability of each behavioral phase |
| `segment_similarity_score` | How similar were the merged units within each segment | Uniformity within phases |
| `transition_delta` | What changed at each boundary | What triggers phase transitions |

**Why this matters:** Chopper showed that cross-level phenomena (like DVFS)
are the largest bottleneck source. Squash segments provide this multi-level
view without requiring manual granularity definition — the data tells us
where the natural boundaries are.

### 3.2 Features from Delta (Field-Level Incremental)

Delta serves **Category 3: Anomaly Detection** and partially **Category 4:
Causal Chain**.

| Feature to Extract | Description | AI Interpretation |
|--------------------|-------------|-------------------|
| `hot_fields` | Which fields change most frequently across units | What behavioral dimensions are most active |
| `cold_fields` | Which fields rarely change | Invariants of the workload |
| `outlier_units` | Units whose delta is significantly larger than average | Behavioral anomalies worth investigating |
| `delta_pattern_clusters` | Groups of units with similar delta patterns | Sub-populations within the workload |
| `field_correlation` | Which fields tend to change together | Potential causal relationships |

**Why this matters:** Opal proved that even one focused signal can drive
useful optimization. Delta hot/cold analysis tells the AI exactly where to
look, avoiding the "hundreds of megabytes of raw data" problem that Opal
explicitly calls out.

### 3.3 Features from Existing Compression Structure

These come from the current `compressed_kernel_v8` format and serve
**Category 1** and **Category 3**.

| Feature to Extract | Description | AI Interpretation |
|--------------------|-------------|-------------------|
| `rle_coverage` | Fraction of instructions compressible by RLE | Computational regularity / loop structure |
| `rle_length_distribution` | Histogram of RLE run lengths | Fine vs coarse loop granularity |
| `cross_tb_offset_coverage` | Fraction of TBs representable by address offset alone | Data parallelism regularity |
| `address_override_density` | Fraction of addresses needing per-TB override | Data-dependent memory divergence |
| `warp_diff_distribution` | Distribution of per-warp diff entry counts | Control flow divergence structure |
| `full_encoding_fallback_rate` | Fraction of TBs that cannot be delta-encoded | Workload irregularity at TB level |
| `shared_pc_sequence_length` | Length of the shared PC sequence | Degree of SIMT convergence |

### 3.4 Features from Simulator Statistics

These are already available and serve **Category 1**.

| Feature to Extract | Description | AI Interpretation |
|--------------------|-------------|-------------------|
| `ipc` | Instructions per cycle | Overall execution efficiency |
| `l1_miss_rate` | L1 cache miss rate | Data locality quality |
| `l2_miss_rate` | L2 cache miss rate | Working set vs cache capacity |
| `memory_throughput` | Achieved memory bandwidth | Distance from memory roof |
| `compute_utilization` | Achieved compute throughput | Distance from compute roof |
| `occupancy` | Active warps / max warps | Resource utilization |
| `shared_mem_bank_conflict` | Bank conflict rate | Shared memory layout quality |
| `warp_divergence_rate` | Fraction of divergent branches | Control flow efficiency |

### 3.5 Features from Control Bits (Architecture-Specific, Optional)

These serve **Category 4: Causal Chain** at the local level. They are
architecture-specific (NVIDIA SASS), so they should be treated as
supplementary, not foundational.

| Feature to Extract | Description | AI Interpretation |
|--------------------|-------------|-------------------|
| `stall_count_distribution` | Distribution of stall counts across instructions | Where the pipeline waits |
| `barrier_wait_frequency` | How often instructions wait on barriers | Synchronization pressure |
| `yield_frequency` | How often instructions yield | Warp scheduler pressure |

---

## Part 4: The Gap We Fill

| Aspect | Existing Work | Our Position |
|--------|--------------|-------------|
| **Data source** | Real hardware profiling (Opal, Chopper, zymtrace) | Simulator trace — sees internal state invisible to hardware profilers |
| **Trace abstraction** | OS kernel traces (TAAF) | GPU execution traces — no one has done semantic abstraction here |
| **Multi-level analysis** | Manual granularity definition (Chopper) | Data-driven segmentation via Squash — natural boundaries, not predefined |
| **Feature extraction** | Separate from compression (all existing work) | Unified with compression — compression IS feature extraction |
| **AI input** | Raw counters or manually curated reports | Structured semantic features produced as compression byproduct |

**Our unique contribution:** We are the first to unify GPU trace compression
with semantic feature extraction, producing structured AI-consumable features
as a natural byproduct of the compression process. Existing systems either
profile on real hardware (limited visibility), abstract OS traces (wrong
domain), or do multi-level analysis with manual granularity (not scalable).

---

## Part 5: Recommended Execution Order

### Step 1: Baseline Diagnosis (no compression changes)

- Run a familiar AI workload on 4090 via tracer + simulator
- Extract features from Section 3.3 (existing compression) + Section 3.4
  (simulator stats)
- Feed to AI Agent, evaluate diagnostic quality
- Identify blind spots

### Step 2: Squash Prototype (behavioral segmentation)

- Implement temporal merging on GPU trace (kernel or TB level)
- Extract features from Section 3.1
- Re-run AI diagnosis, compare with baseline
- Evaluate whether behavioral segmentation fills the blind spots

### Step 3: Delta Prototype (anomaly detection)

- Implement field-level incremental encoding for high-level fields
  (active mask distribution, address patterns, divergence metrics)
- Extract features from Section 3.2
- Re-run AI diagnosis, compare with Step 2

### Step 4: Unified Pipeline

- Integrate Squash + Delta into the trace compression pipeline
- Evaluate compression ratio vs semantic quality tradeoff
- Finalize the feature extraction schema for AI consumption

---

## Part 6: AI Agent for Architecture Evaluation — Related Work and Our Value

### 6.1 Current State of AI-Driven Architecture Evaluation

The question "can AI judge whether a hardware architecture decision is
reasonable?" is being actively explored but remains largely unsolved.

#### ArchAgent (2026) — AI Discovers Cache Replacement Policies

**Source:** arXiv:2602.22425

ArchAgent uses AlphaEvolve (Google's evolutionary coding agent) to
automatically discover last-level cache replacement policies in ChampSim.
It achieved 5.3% IPC speedup over prior SoTA on Google Workload Traces
in two days without human intervention.

**What it tells us:**
- AI CAN make architecture-level decisions (cache policy) that outperform
  human-designed ones
- The key enabler is a fast evaluation loop: ArchAgent writes C++ policy
  code, runs ChampSim simulation, evaluates IPC, iterates
- It also discovered "simulator escapes" — loopholes in the simulator that
  AI exploited, which humans missed

**Relevance to us:** ArchAgent operates in a closed loop (generate policy
→ simulate → evaluate → iterate). Our work provides the DIAGNOSTIC side:
instead of blindly searching the design space, our AI agent analyzes trace
semantics to EXPLAIN why a configuration performs the way it does, enabling
targeted exploration rather than brute-force search.

#### CacheC (Euro-Par 2025) — LLM-Based GPU Cache Management

**Source:** Springer LNCS vol 15901

CacheC uses LLMs to analyze cache affinity at load-instruction granularity
for concurrent GPU kernels. It scores kernel pairs for concurrency suitability
and formulates load-specific cache bypassing strategies.

**What it tells us:**
- LLMs can reason about cache behavior at fine granularity when given
  structured input (per-load features, cache level affinity scores)
- The key is not feeding raw data but structured, per-unit feature vectors

**Relevance to us:** CacheC operates on a narrow scope (cache policy for
kernel pairs). Our approach is broader — we aim to provide a general
diagnostic framework covering multiple architecture dimensions.

#### Architecture 2.0 (Harvard CS249r / IEEE Computer 2025)

**Source:** IEEE Computer Magazine, February 2025; Harvard CS249r Fall 2025

This is the academic framing of the entire field. Key positions:

- "AI agents could herald a new golden age of modern computer system design"
- The field needs: curated datasets, defined benchmarks, interpretability,
  balanced AI autonomy with human expertise
- GenAI for Systems publications expanded 20x between 2017-2025
- GPU optimization is described as "qualitatively different from CPU" —
  the optimization space is much larger and less understood

**What it tells us:**
- The field is real, growing fast, and recognized at top venues
- But most work focuses on CPU (cache policy, branch prediction, prefetching)
- **GPU microarchitecture is explicitly called out as underexplored**

#### ReaLLM (ASAP 2025) — Trace-Driven Simulation for LLM Hardware DSE

**Source:** BSG.ai / ASAP 2025

ReaLLM achieves 164x speedup in design space exploration by identifying
1,600 key kernels and precomputing their latencies for trace-driven replay.

**What it tells us:**
- Trace-driven simulation is the dominant methodology for hardware DSE
- Speed is critical — faster simulation enables larger design space coverage
- But ReaLLM does NOT use AI to interpret results, only to accelerate
  simulation

#### Omniwise (2025) — LLM Predicts Performance Counters Without Execution

**Source:** arXiv:2506.20886

Omniwise achieves 98.2% accuracy predicting arithmetic intensity and 87.3%
for cache hit rate from code alone, without running the kernel.

**What it tells us:**
- LLMs already have strong understanding of GPU compute patterns
- This can complement our approach: Omniwise predicts counters from code,
  our system extracts behavioral semantics from trace — together they
  provide both "what should happen" and "what actually happened"

### 6.2 The Gap Map — Where Existing Work Stops

```
                    Architecture Evaluation Capability
                    
   Code Analysis    Trace Analysis    Architecture DSE
   ~~~~~~~~~~~~    ~~~~~~~~~~~~~~    ~~~~~~~~~~~~~~~~
   Omniwise         (gap)            ArchAgent
   (predict          |               (cache policy
    counters         |                via evolution)
    from code)       |
                     |               ReaLLM
   Opal             |               (fast simulation
   (optimize         |                for DSE)
    kernel from      |
    HW counters)     |               CacheC
                     |               (cache bypass
   SwizzlePerf      |                via LLM)
   (memory           |
    swizzling)       |
                     |
   KPerfIR          TAAF (OS only)
   (compiler-level   |
    profiling)       |
                     |
                  >> OUR WORK <<
                  (GPU trace semantic
                   abstraction + AI
                   architecture diagnosis)
```

The central column — **AI-driven analysis of GPU execution traces for
architecture diagnosis** — is empty. Everyone is either:
- Analyzing code or hardware counters (left column)
- Searching design space via simulation (right column)
- Abstracting OS traces, not GPU traces (TAAF)

Nobody is extracting structured behavioral semantics from GPU simulator
traces and feeding them to AI for architecture-level diagnosis.

### 6.3 Our Work's Value Proposition

**Why this matters for the research community:**

1. **Closes the diagnostic gap:** ArchAgent can find good policies by
   brute-force search, but cannot explain WHY they work. Our trace
   semantics provide the explanatory layer that turns black-box search
   into interpretable architecture insights.

2. **Enables targeted DSE:** Instead of exploring the entire design space
   (10,000+ experiments per NSDI 2025), AI can use our diagnostic output
   to focus on the dimensions that actually matter for a given workload.

3. **Unique data source:** Simulator traces contain information invisible
   to hardware profilers (complete warp instruction flows, per-TB address
   streams, control bits). No existing system exploits this richness.

4. **Compression as feature extraction:** Unlike TAAF which builds a
   separate knowledge graph, our approach extracts features AS PART OF
   compression — zero additional cost for feature extraction.

5. **Cross-architecture applicability:** High-level behavioral features
   (access patterns, divergence, TB similarity) are architecture-agnostic,
   making the diagnostic framework reusable across GPU generations.

### 6.4 Confirmed Direction: What We Should Do Next

Based on this analysis, our work has clear value and a defensible position.
The next steps are:

**Immediate (validate AI diagnostic capability):**
- Run a familiar AI workload, collect trace + stats
- Build a feature extraction prototype using existing compression metadata
- Evaluate what AI can diagnose and where it fails

**Short-term (fill diagnostic blind spots):**
- Introduce Squash for behavioral segmentation (Category 2)
- Introduce Delta for anomaly detection (Category 3)
- Re-evaluate diagnostic quality

**Medium-term (architecture evaluation loop):**
- Connect diagnostic output to architecture parameter suggestions
- Build a closed loop: trace → semantic features → AI diagnosis →
  architecture parameter change → re-simulate → compare
- This is the "Architecture 2.0" vision applied to GPU microarchitecture
  via trace semantics

**Long-term (paper story):**
- Contribution 1: Unified GPU trace compression + semantic feature
  extraction framework (Squash/Delta adapted from difftest)
- Contribution 2: AI agent that diagnoses GPU architecture issues from
  trace semantics, outperforming raw-data approaches (like TAAF's 31.2%
  improvement, but for GPU domain)
- Contribution 3: Closed-loop architecture evaluation enabled by
  trace-semantic diagnostics

---

## Sources

- [Opal: A Modular Framework for Optimizing Performance using Analytics and LLMs](https://arxiv.org/pdf/2510.00932)
- [Chopper: A Multi-Level GPU Characterization Tool](https://arxiv.org/abs/2512.08242)
- [TAAF: A Trace Abstraction and Analysis Framework](https://arxiv.org/abs/2601.02632)
- [SwizzlePerf: Hardware-Aware LLMs for GPU Kernel Performance Optimization](https://www.arxiv.org/pdf/2508.20258)
- [AutoKernel: Autonomous GPU Kernel Optimization](https://arxiv.org/html/2603.21331v1)
- [Omniwise: Predicting GPU Kernels Performance with LLMs](https://arxiv.org/pdf/2506.20886)
- [KPerfIR — OSDI 2025](https://www.usenix.org/system/files/osdi25-guan.pdf)
- [CS249r: Can LLMs Optimize GPU Performance?](https://harvard-edge.github.io/cs249r_fall2025/blog/2024/10/01/gpu-performance-engineering/)
- [zymtrace GPU Profiler](https://zymtrace.com/article/zero-friction-gpu-profiler/)
- [Dissecting and Modeling the Architecture of Modern GPU Cores — MICRO 2025](https://dl.acm.org/doi/10.1145/3725843.3756041)
- [ArchAgent: Agentic AI-driven Computer Architecture Discovery](https://arxiv.org/abs/2602.22425)
- [CacheC: LLM-Based GPU Cache Management — Euro-Par 2025](https://link.springer.com/chapter/10.1007/978-3-031-99857-7_9)
- [Architecture 2.0: Foundations of AI Agents for Modern Computer System Design — IEEE Computer](https://ieeexplore.ieee.org/document/10857820/)
- [CS249r: Architecture 2.0 — Harvard Course](https://harvard-edge.github.io/cs249r_fall2025/)
- [GenAI for Systems: Recurring Challenges and Design Principles](https://arxiv.org/html/2602.15241)
- [ReaLLM: Trace-Driven Framework for Rapid LLM Simulation — ASAP 2025](https://www.bsg.ai/papers/Peng_ReaLLM_ASAP_2025.pdf)
- [Accelerating Design Space Exploration for LLM Training — NSDI 2025](https://www.usenix.org/system/files/nsdi25-gui.pdf)
- [GainSight with Accel-Sim Backend](https://gainsight.stanford.edu/backend/gpu-sim.html)
- [MAccel-Sim: Multi-GPU Simulator — IISWC 2024](https://engineering.purdue.edu/tgrogers/publication/bose-iiswc-poster-2024/bose-iiswc-poster-2024.pdf)
