# gcl_gpu_sim_acceleration_report - Design Spec

> Human-readable design narrative. Machine-readable execution contract: `spec_lock.md`.

## I. Project Information

| Item | Value |
| ---- | ----- |
| **Project Name** | gcl_gpu_sim_acceleration_report |
| **Canvas Format** | PPT 16:9 (1280x720) |
| **Page Count** | 12 |
| **Design Style** | General Consulting + 学术技术汇报 / 清爽科技报告风 |
| **Target Audience** | 老师、同学、课题组成员、项目评审 |
| **Use Case** | GCL GPU 仿真加速技术课程/组会报告 |
| **Created Date** | 2026-06-17 |

---

## II. Canvas Specification

| Property | Value |
| -------- | ----- |
| **Format** | PPT 16:9 |
| **Dimensions** | 1280x720 |
| **viewBox** | `0 0 1280 720` |
| **Margins** | left/right 56px, top 46px, bottom 38px |
| **Content Area** | 1168x604 |

---

## III. Visual Theme

### Theme Style

- **Style**: General Consulting
- **Theme**: Light theme
- **Tone**: professional, analytical, clean technology report

### Color Scheme

| Role | HEX | Purpose |
| ---- | --- | ------- |
| **Background** | `#F7F9FC` | Page background |
| **Secondary bg** | `#FFFFFF` | Cards and panels |
| **Primary** | `#155E75` | Title bars, process backbone, icons |
| **Accent** | `#2563EB` | Key metrics, selected nodes, highlights |
| **Secondary accent** | `#22C55E` | Positive compression / speedup signal |
| **Body text** | `#172033` | Main body text |
| **Secondary text** | `#5B6475` | Captions and annotations |
| **Tertiary text** | `#8792A2` | Footers and minor labels |
| **Border/divider** | `#D8DEE9` | Rules, card borders, axis lines |
| **Warning** | `#DC2626` | Limitations and unproven claims |

---

## IV. Typography System

### Font Plan

**Typography direction**: PPT-safe modern CJK sans with monospace for metrics and formulas.

| Role | Chinese | English | Fallback tail |
| ---- | ------- | ------- | ------------- |
| **Title** | `"Microsoft YaHei", "PingFang SC"` | Arial | sans-serif |
| **Body** | `"Microsoft YaHei", "PingFang SC"` | Arial | sans-serif |
| **Emphasis** | `"Microsoft YaHei", "PingFang SC"` | Arial | sans-serif |
| **Code** | — | `Consolas, "Courier New"` | monospace |

**Per-role font stacks**:

- Title: `"Microsoft YaHei", "PingFang SC", Arial, sans-serif`
- Body: `"Microsoft YaHei", "PingFang SC", Arial, sans-serif`
- Emphasis: `"Microsoft YaHei", "PingFang SC", Arial, sans-serif`
- Code: `Consolas, "Courier New", monospace`

### Font Size Hierarchy

**Baseline**: Body font size = 18px.

| Purpose | Size |
| ------- | ---- |
| Cover title | 58-64px |
| Section opener | 44-48px |
| Page title | 32-36px |
| Hero number | 40-48px |
| Subtitle | 22-26px |
| Body content | 18px |
| Annotation / caption | 13-15px |
| Footer | 11-12px |

---

## V. Layout Principles

### Page Structure

- **Header area**: 46-110px, page title and small section label.
- **Content area**: 540-600px, diagrams / cards / charts.
- **Footer area**: 28-38px, source note and page number.

### Layout Pattern Library

- Cover and ending: negative-space-driven title block, no heavy card grid.
- Method pages: process_flow / pipeline_with_stages with one dominant visual.
- Model pages: layered architecture and graph-node diagram.
- Result pages: KPI cards + scatter/cluster schematic + clear takeaway.
- Limitation pages: vertical list with warning color, not a dense table.

### Spacing Specification

| Element | Current Project |
| ------- | --------------- |
| Safe margin from canvas edge | 56px |
| Content block gap | 28-36px |
| Icon-text gap | 10-12px |
| Card gap | 22-28px |
| Card padding | 22-28px |
| Card border radius | 12px |

---

## VI. Icon Usage Specification

### Source

- **Built-in icon library**: `tabler-outline`
- **Stroke width**: 2
- **Usage method**: SVG placeholder `<use data-icon="tabler-outline/icon-name" .../>`

### Recommended Icon List

| Purpose | Icon Path | Page |
| ------- | --------- | ---- |
| Cover / presentation | `tabler-outline/presentation-analytics` | P01 |
| Motivation / acceleration | `tabler-outline/bolt` | P02, P12 |
| GPU / compute | `tabler-outline/cpu` | P02, P05 |
| Pipeline | `tabler-outline/route-2` | P03 |
| Graph structure | `tabler-outline/binary-tree` | P04 |
| RGCN relation | `tabler-outline/network` | P05 |
| Layered readout | `tabler-outline/layers-linked` | P06 |
| Clustering | `tabler-outline/chart-scatter` | P08, P10 |
| Metrics | `tabler-outline/chart-bar` | P09 |
| Target / representative | `tabler-outline/target-arrow` | P08 |
| Limitations | `tabler-outline/alert-triangle` | P11 |
| Conclusion | `tabler-outline/check` | P12 |
| Artifact / source | `tabler-outline/file-description` | footer |
| Settings / experiment | `tabler-outline/settings-code` | P09 |

---

## VII. Visualization Reference List

Catalog read: 71 templates

| Page | Template | Path | Summary-quote (verbatim from `charts_index.json`) | Usage |
| ---- | -------- | ---- | ------------------------------------------------- | ----- |
| P02 | kpi_cards | `templates/charts/kpi_cards.svg` | "Pick for 4-8 standalone numeric metrics shown as overview cards (2x2 or 1x4) — exec summary opener, dashboard headline, quarterly recap, results-at-a-glance. Skip if metrics have target baselines (use bullet_chart) or single hero number (use gauge_chart)." | Show trace scale and compression headline |
| P03 | pipeline_with_stages | `templates/charts/pipeline_with_stages.svg` | "Pick for 3-5 horizontal pipeline stages, each = title + 1-line description + output artifact, connected by arrows (data pipelines, ETL, build pipelines). Skip if any stage lacks an artifact (use process_flow or numbered_steps)." | Trace -> graph -> RGCN -> K-means -> representative |
| P05 | layered_architecture | `templates/charts/layered_architecture.svg` | "Pick for 3-4 horizontal architecture layers (presentation/service/data), 2-4 module cards per layer, each card = title + 1-line description (description required, even if source brief). Skip if no per-module descriptions (use icon_grid) or no horizontal layering (use module_composition)." | RGCN relation-aware encoder stack |
| P08 | process_flow | `templates/charts/process_flow.svg` | "Pick for 3-8 sequential steps connected by simple arrows — approval workflows, customer onboarding, request handling, lifecycle stages. Skip if cyclical (use circular_stages) or stages produce named outputs (use pipeline_with_stages)." | K-means iteration and representative selection |
| P09 | kpi_cards | `templates/charts/kpi_cards.svg` | "Pick for 4-8 standalone numeric metrics shown as overview cards (2x2 or 1x4) — exec summary opener, dashboard headline, quarterly recap, results-at-a-glance. Skip if metrics have target baselines (use bullet_chart) or single hero number (use gauge_chart)." | Experiment results dashboard |
| P10 | scatter_chart | `templates/charts/scatter_chart.svg` | "Pick for x-y correlation, cluster, or outlier scan. Skip if a size dimension also matters (use bubble_chart)." | Cluster 263/2 schematic |
| P11 | vertical_list | `templates/charts/vertical_list.svg` | "Pick for 3-6 numbered key points each with a short description — design principles, core tenets, action items, key takeaways, recommendations, executive summary points. Skip for icon-style cards (use icon_grid) or sequential steps (use numbered_steps)." | Limitations and next work |

**Runners-up considered**:

- `numbered_steps` | rejected for P03/P08: pipeline pages have explicit intermediate artifacts and outputs, so `pipeline_with_stages` / `process_flow` fit better.
- `layered_architecture` | rejected for P03: the overall method is sequential, not layered.
- `basic_table` | rejected for P09: results are headline metrics, not a dense grid requiring table precision.

---

## VIII. Image Resource List

No external images. All visuals are SVG-native diagrams, charts, icon placeholders, and text.

---

## IX. Content Outline

### Part 1: Why This Matters

#### Slide 01 - Cover

- **Layout**: Negative-space cover with thin technical grid and one pipeline line
- **Title**: 基于图对比学习的 GPU 仿真加速技术
- **Subtitle**: GCL + RGCN + K-means 的 ResNet-50 kernel 表示与聚类复现
- **Info**: 丁逸夫 / 吴弘煜 / 付嘉琪；2026

#### Slide 02 - Motivation

- **Layout**: KPI cards + right-side takeaway
- **Visualization**: `kpi_cards`
- **Content**:
  - GPU kernel 是模拟与性能分析基本单元
  - 完整 trace 成本高，逐 kernel 仿真开销大
  - 目标：用代表 kernel 减少模拟对象
  - 本实验：265 kernel -> K=2 聚类结构

#### Slide 03 - Overall Pipeline

- **Layout**: Horizontal pipeline with staged outputs
- **Visualization**: `pipeline_with_stages`
- **Content**:
  - Trace acquisition
  - Kernel graph construction
  - RGCN encoding + GCL training
  - K-means clustering
  - Representative kernel selection

### Part 2: Method

#### Slide 04 - Kernel as Graph

- **Layout**: Graph diagram + explanation callouts
- **Title**: 将 kernel 执行轨迹转化为异构关系图
- **Content**:
  - 节点：指令 / 访存 pseudo node
  - 边：control_flow / data_source / data_destination
  - 图表示比手工统计特征保留更多结构信息

#### Slide 05 - RGCN Encoder

- **Layout**: Layered model architecture
- **Visualization**: `layered_architecture`
- **Content**:
  - 3-layer RGCN
  - input_dim=64, hidden_dim=128
  - relation_count=3
  - relation-aware message passing

#### Slide 06 - Readout Hierarchy

- **Layout**: Vertical hierarchy from node to kernel
- **Title**: 从节点表示读出 kernel embedding
- **Content**:
  - node -> warp -> CTA -> selected SM -> kernel
  - kernel_embedding_dim=256
  - embedding 同时包含节点特征和结构关系

#### Slide 07 - Graph Contrastive Learning

- **Layout**: Two-view contrastive diagram
- **Title**: 无标签条件下训练图编码器
- **Content**:
  - 同一 kernel 的两个增强视图构成正样本
  - 不同 kernel 视图构成负样本
  - InfoNCE 让结构相近 kernel 在向量空间更接近

#### Slide 08 - K-means & Representative

- **Layout**: Process flow + representative target
- **Visualization**: `process_flow`
- **Content**:
  - 输入：265 个 256 维 kernel embedding
  - 用轮廓系数选择 K
  - 每个簇选择离中心最近的样本作为代表 kernel

### Part 3: Experiment and Findings

#### Slide 09 - Experimental Setup and Results

- **Layout**: KPI dashboard
- **Visualization**: `kpi_cards`
- **Content**:
  - ResNet-50 full trace
  - 265 kernel invocation
  - 124876 CTA records
  - selected K=2
  - silhouette=0.481866
  - inter/intra ratio=2.016339

#### Slide 10 - Cluster Interpretation

- **Layout**: Scatter-style cluster schematic + side explanation
- **Visualization**: `scatter_chart`
- **Content**:
  - cluster 0: 263 samples
  - cluster 1: 2 samples
  - 当前结果更像主体样本与少量特殊 kernel 分离
  - 不宜直接解释为明确功能类别

#### Slide 11 - Course Report Conclusion

- **Layout**: KPI conclusion page with compression result and three evidence points
- **Visualization**: conclusion cards
- **Content**:
  - 真实 ResNet-50 trace 共 265 个 kernel invocation
  - K-means 选出 K=2，最终只需仿真 2 个代表 kernel
  - 模拟对象从 265 压缩到 2，理论模拟加速约 132.5 倍
  - 该结果说明 GCL 可以显著减少课程实验中的模拟对象规模

#### Slide 12 - Final Conclusion

- **Layout**: Ending page with method chain and final takeaway
- **Title**: GCL 完成 kernel 表示学习与代表性压缩
- **Content**:
  - GNN 提供 kernel 结构编码能力
  - GCL 提供无监督表示学习目标
  - K-means 提供聚类和代表 kernel 选择
  - 本课程报告完成从 trace 到代表 kernel 的端到端复现，并得到约 132.5 倍理论模拟加速

---

## X. Speaker Notes Requirements

- **Total duration**: 10-12 minutes
- **Notes style**: Formal but conversational
- **Purpose**: Explain method, report reproduction evidence, state limitations honestly
- **Files**: one note per page after Executor phase; `notes/total.md` master uses `# Slide NN` headings

---

## XI. Technical Constraints Reminder

1. viewBox: `0 0 1280 720`
2. Background uses `<rect>` elements
3. Text wrapping uses `<tspan>`; `<foreignObject>` forbidden
4. Transparency uses `fill-opacity` / `stroke-opacity`; `rgba()` forbidden
5. Forbidden: `<style>`, `class`, `<foreignObject>`, `textPath`, `animate*`, `script`
6. Group opacity forbidden; set opacity on each child element
7. Icon placeholders must use approved `tabler-outline` inventory only
