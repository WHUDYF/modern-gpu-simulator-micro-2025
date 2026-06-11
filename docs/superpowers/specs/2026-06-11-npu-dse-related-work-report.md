# NPU / DNN Accelerator DSE Related Work Report

Date: 2026-06-11

## 1. Why This Direction Is More Natural Than GPU DSE

If the target platform is an internal NPU rather than a commercial GPU, the
research risk changes substantially.

For GPUs, most hardware details are externally inferred:

```text
workload trace / counters
  -> inferred bottleneck
  -> guessed architecture knob
  -> simulator validation
```

For an NPU designer, RTL and microarchitecture knowledge are available:

```text
operator profile / workload trace
  -> RTL module-level bottleneck model
  -> resource/knob attribution
  -> PPA-constrained architecture recommendation
```

This makes the work easier to defend. The contribution no longer depends on
guessing hidden GPU resources. It can be framed as an RTL-aware, workload-aware
NPU DSE workflow.

Recommended positioning:

```text
RTL-Aware Bottleneck-Guided Design Space Exploration for NPUs
```

or:

```text
Representative-Operator-Guided NPU Architecture Exploration
```

## 2. Publication Landscape

The related work shows that NPU / DNN accelerator DSE is a mature but still
active area. Strong versions appear in top architecture, EDA, and performance
analysis venues:

| Work | Main Idea | Venue / Status | Relevance |
|---|---|---|---|
| Timeloop | Systematic DNN accelerator architecture/mapping evaluation | ISPASS 2019 | Open baseline for accelerator evaluation |
| MAESTRO | Data-centric performance/cost model for DNN dataflows | MICRO 2019; IEEE Micro Top Picks | Key cost-model reference |
| ZigZag | Memory-centric DNN accelerator architecture-mapping DSE | IEEE TC / arXiv | Very close to NPU memory hierarchy exploration |
| SCALE-Sim | Systolic-array simulator and scalability evaluation | arXiv 2018; ISPASS 2020; ISPASS 2025 v3 | Good open simulator baseline |
| ConfuciuX | RL-based hardware resource assignment | MICRO 2020 | Shows automated resource assignment is publishable |
| GAMMA | GA-based mapping-space exploration | ICCAD 2020 | Mapping optimization baseline |
| DiGamma | HW-mapping co-optimization | DATE 2022 | Close to hardware + mapping joint search |
| Mind Mappings | Differentiable mapping-space search | ASPLOS 2021 | High-end mapping search reference |
| Demystifying MSE for NPUs | Analysis of NPU mapping-space exploration behavior | IISWC 2022 | Directly NPU-oriented |
| Explainable-DSE | Bottleneck-guided accelerator HW/SW codesign DSE | ASPLOS 2023 | Closest method-level inspiration |

Implication:

```text
Top-tier path:
  new modeling/search method + broad benchmarks + strong baselines.

Pragmatic journal/conference path:
  open workflow + RTL-aware bottleneck model + representative operator
  evaluation + internal NPU case study.
```

## 3. What Existing Work Already Covers

### 3.1 Timeloop

Timeloop is an evaluation framework for DNN accelerator architectures and
mapping choices. It helps describe architectures, map workloads, and estimate
performance/cost.

What it gives us:

```text
architecture/mapping specification
evaluation baseline
open reference for DNN accelerator modeling
```

What it does not fully solve for us:

```text
RTL-aware bottleneck attribution for a specific internal NPU
representative-operator compression for reducing DSE cost
module-level knob recommendation tied to internal RTL blocks
```

### 3.2 MAESTRO

MAESTRO models DNN dataflows from a data-centric perspective. It explains how
reuse, data movement, and hardware resources affect performance and cost.

What it gives us:

```text
reuse / data movement reasoning
dataflow-aware cost decomposition
strong conceptual support for bottleneck modeling
```

How it connects to our NPU direction:

```text
MAESTRO-style data-centric analysis can become the abstract layer,
while RTL module counters and internal simulator outputs become the concrete
NPU-specific evidence.
```

### 3.3 ZigZag

ZigZag is memory-centric DNN accelerator DSE. It focuses on mapping neural
network layers onto accelerator memory hierarchies efficiently.

What it gives us:

```text
memory hierarchy as a first-class DSE target
architecture-mapping co-exploration
good reference if our NPU bottleneck is SRAM / DRAM / data movement
```

Risk:

```text
If our work only says "optimize memory hierarchy for NPU",
ZigZag is already close. We need RTL-aware diagnosis or representative
operator compression as the differentiator.
```

### 3.4 SCALE-Sim

SCALE-Sim provides systolic-array simulation and accelerator scalability
analysis. The v3 line shows this remains active.

What it gives us:

```text
open systolic-array simulator baseline
low-barrier experiment platform
publication precedent for simulator/tool papers
```

How we can use it:

```text
If internal NPU simulator/RTL cannot be fully disclosed,
SCALE-Sim can be a public validation backend or artifact-compatible fallback.
```

### 3.5 ConfuciuX

ConfuciuX uses reinforcement learning to assign hardware resources for DNN
accelerators.

What it gives us:

```text
"hardware resource assignment" is a recognized DNN accelerator problem
automated NPU parameter selection can be publishable
```

Difference:

```text
ConfuciuX is search/learning-driven.
Our intended direction should be bottleneck-guided and RTL-aware,
so the system can explain why a resource should be changed.
```

### 3.6 GAMMA and DiGamma

GAMMA focuses on mapping-space exploration. DiGamma extends toward
hardware-mapping co-optimization.

What they give us:

```text
strong mapping-space and co-optimization baselines
evidence that DNN accelerator search can be improved with domain-aware search
```

Risk:

```text
If our work becomes only "another search algorithm",
these works become hard baselines.
```

Safer framing:

```text
our contribution is not a new generic optimizer,
but a bottleneck-explained representative-operator front-end for NPU DSE.
```

### 3.7 Mind Mappings

Mind Mappings uses differentiable search to improve accelerator mapping-space
exploration.

What it gives us:

```text
upper-bound reference for algorithmic search sophistication
evidence that mapping-space search is a hard and active problem
```

Practical advice:

```text
Do not compete directly with Mind Mappings in the first version.
Keep mapping optimization simple and focus on hardware bottleneck diagnosis.
```

### 3.8 Demystifying Map Space Exploration for NPUs

This is a directly NPU-oriented analysis of map-space exploration. It helps
explain why NPU mapping/search spaces are difficult and why naive exploration
can waste many trials.

What it gives us:

```text
NPU-specific motivation
evidence that exploration cost is real
terminology for NPU map-space behavior
```

How it supports our story:

```text
If map-space and design-space exploration are both expensive,
representative operators/layers become useful for reducing repeated evaluation.
```

### 3.9 Explainable-DSE

Explainable-DSE is the closest conceptual reference. It uses bottleneck models
to guide accelerator HW/SW codesign instead of black-box search.

Key lesson:

```text
explainability is useful because it improves search efficiency,
not only because it explains final results.
```

Our NPU extension opportunity:

```text
Explainable-DSE uses domain-specific bottleneck models for DNN accelerators.
We can make the bottleneck model RTL-aware and operator-representative for a
specific NPU design flow.
```

## 4. Where Our NPU Work Can Differ

A defensible first version should not claim to solve all NPU DSE. It should
focus on a narrower problem:

```text
Can representative operators preserve NPU architecture-DSE decisions
while reducing full-model evaluation cost?
```

The proposed pipeline:

```text
AI workload model
  -> operator/layer profiling
  -> representative operator selection
  -> RTL/module-level bottleneck diagnosis
  -> knob/resource sensitivity
  -> PPA-constrained candidate generation
  -> full-model validation
```

Candidate NPU knobs:

```text
MAC array dimensions
local SRAM capacity
global buffer capacity
SRAM bank count / bandwidth
NoC or crossbar bandwidth
DMA channel count / outstanding requests
activation / special-function unit count
tiling and buffer partitioning parameters
```

Candidate bottleneck categories:

```text
compute array under-utilization
DRAM bandwidth pressure
SRAM bandwidth pressure
SRAM capacity spill
NoC contention
DMA stall
pipeline imbalance
scheduler / dependency stall
special-function unit bottleneck
```

## 5. Recommended Baselines

Minimum baselines:

```text
full-model DSE
random operator selection
top-latency operator selection
top-memory-traffic operator selection
operator-type/manual-rule selection
our representative-operator selection
```

If time permits:

```text
Timeloop/MAESTRO-style abstract model baseline
SCALE-Sim baseline for systolic-array cases
black-box optimizer baseline, e.g. random / GA / BO
```

Most important comparison:

```text
Does our representative subset preserve the same top-k hardware configs
as full-model evaluation?
```

## 6. Suggested Metrics

Primary metrics:

```text
evaluation cost reduction
best speedup under fixed evaluation budget
top-k config overlap with full-model DSE
PPA-valid candidate rate
bottleneck agreement rate
```

Secondary metrics:

```text
latency improvement
energy improvement
area/power overhead
operator coverage by latency / traffic / type
search iterations to target quality
```

## 7. Practical Publication Strategy

Fast and realistic positioning:

```text
open / reproducible / RTL-aware NPU DSE prototype
with representative-operator evaluation
```

Potential paper title:

```text
Representative-Operator-Guided Bottleneck Analysis for Efficient NPU
Architecture Exploration
```

Conservative contribution list:

```text
1. An RTL-aware bottleneck model that maps operator execution symptoms to NPU
   modules and tunable resources.
2. A representative-operator selection workflow for reducing DSE evaluation
   cost.
3. A PPA-constrained candidate-generation loop validated against full-model
   NPU simulation.
4. An empirical comparison against random, top-latency, and top-traffic
   operator selection.
```

Avoid these claims in the first version:

```text
automatically designs the next-generation NPU
beats all DNN accelerator DSE methods
generalizes to every NPU architecture
LLM can infer all bottlenecks from RTL without validation
```

## 8. Recommended Reading Order

1. Explainable-DSE
   - Read first because it directly teaches bottleneck-guided acquisition.
2. MAESTRO
   - Read second because it gives data movement/reuse reasoning.
3. Timeloop
   - Read third because it defines a common open evaluation language.
4. ZigZag
   - Read when focusing on memory hierarchy and mapping.
5. Demystifying Map Space Exploration for NPUs
   - Read when designing representative operator selection and baselines.
6. SCALE-Sim
   - Read when building or validating a systolic-array simulator path.
7. GAMMA / DiGamma / Mind Mappings / ConfuciuX
   - Read as optimizer/search baselines after the core problem is fixed.

## 9. Immediate Next Step

The next concrete task should be a one-workload feasibility study:

```text
select one internal or public model
profile per-operator latency / traffic / module utilization
select 5-10 representative operators by simple rules
run a small NPU knob grid on representatives
validate top-k configs on the full model
compare with top-latency and random operator selection
```

Go/no-go threshold:

```text
Green:
  <=30% representative evaluation cost reaches >=85-90% of full-model best
  speedup, with meaningful top-k config overlap.

Yellow:
  beats random but not top-latency. Reframe as engineering workflow.

Red:
  no clear advantage over top-latency/random. Stop adding GNN/LLM complexity.
```

