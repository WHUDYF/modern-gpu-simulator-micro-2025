#!/usr/bin/env python3
"""Generate the GCL ResNet-50 reproduction paper as a Word .docx file."""

from __future__ import annotations

import json
import textwrap
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "gcl_resnet50_full_trace_reproduction"
OUT = ROOT / "docs" / "reports" / "gcl-resnet50-reproduction-paper.docx"


def read_json(name: str) -> dict:
    path = ARTIFACT_DIR / name
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def text(value: object, default: str = "未提供") -> str:
    if value is None:
        return default
    return str(value)


def fmt_float(value: object, digits: int = 6) -> str:
    if value is None:
        return "未提供"
    try:
        return f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(value)


def p(content: str = "", style: str | None = None) -> str:
    style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    runs = []
    for i, part in enumerate(content.split("\n")):
        if i:
            runs.append("<w:r><w:br/></w:r>")
        runs.append(f"<w:r><w:t xml:space=\"preserve\">{escape(part)}</w:t></w:r>")
    return f"<w:p>{style_xml}{''.join(runs)}</w:p>"


def table(rows: list[list[str]]) -> str:
    cells_xml = []
    for row in rows:
        cell_xml = "".join(
            "<w:tc><w:tcPr><w:tcW w:w=\"0\" w:type=\"auto\"/></w:tcPr>"
            + p(str(cell))
            + "</w:tc>"
            for cell in row
        )
        cells_xml.append(f"<w:tr>{cell_xml}</w:tr>")
    return (
        "<w:tbl>"
        "<w:tblPr><w:tblStyle w:val=\"TableGrid\"/>"
        "<w:tblW w:w=\"0\" w:type=\"auto\"/>"
        "<w:tblBorders>"
        "<w:top w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:left w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:bottom w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:right w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:insideH w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:insideV w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>"
        "</w:tblBorders></w:tblPr>"
        + "".join(cells_xml)
        + "</w:tbl>"
    )


def section(title: str, paragraphs: list[str]) -> list[str]:
    return [p(title, "Heading1"), *[p(x) for x in paragraphs]]


def subsection(title: str, paragraphs: list[str]) -> list[str]:
    return [p(title, "Heading2"), *[p(x) for x in paragraphs]]


def build_document_xml() -> str:
    reproduction = read_json("resnet50_full_trace_reproduction_manifest.json")
    training = read_json("rgcn_training_run_manifest.json")
    selector = read_json("selector_artifacts.json")
    gate7 = read_json("gate7_cluster_correctness_manifest.json")
    acceptance = read_json("gnn_acceptance_summary.json")

    k_report = selector.get("k_selection_report", {})
    arch = training.get("model_architecture", {})
    opt = training.get("optimizer_config", {})
    geom = gate7.get("embedding_geometry_metrics", {})
    family = gate7.get("family_alignment_metrics", {})
    metric_report = (
        gate7.get("gate7_report_artifacts", {})
        .get("metric_error_report", {})
        .get("report_payload", {})
    )
    representative_reports = (
        gate7.get("gate7_report_artifacts", {})
        .get("representative_quality_report", {})
        .get("report_payload", {})
        .get("cluster_reports", [])
    )
    cluster_evidence = selector.get("cluster_family_evidence_report", {}).get("clusters", [])

    cluster_rows = [["簇编号", "样本数", "权重占比", "解释"]]
    for item in cluster_evidence:
        cluster_rows.append(
            [
                text(item.get("cluster_id")),
                text(item.get("member_count")),
                fmt_float(item.get("weight"), 8),
                "多数标签为 resnet50_real_trace；该标签粒度较粗，不能直接证明语义 family 正确。",
            ]
        )

    rep_rows = [["簇编号", "代表记录", "成员数", "平均距离", "P95 距离", "最大距离"]]
    for item in representative_reports:
        rep_rows.append(
            [
                text(item.get("cluster_id")),
                text(item.get("representative_record_id")),
                text(item.get("member_count")),
                fmt_float(item.get("mean_distance_to_representative"), 6),
                fmt_float(item.get("p95_distance_to_representative"), 6),
                fmt_float(item.get("max_distance_to_representative"), 6),
            ]
        )

    k_rows = [["候选 K", "轮廓系数"]]
    for item in k_report.get("candidates", []):
        k_rows.append([text(item.get("k")), fmt_float(item.get("score"), 8)])

    acceptance_rows = [
        ["验收项", "当前状态", "解释"],
        ["输入来源", "通过", "formal_full_trace_run 为 true，输入为真实 ResNet-50 full trace。"],
        ["RGCN 结构", "通过", "三层 RGCN、三类边关系和层级 readout 均记录在 manifest 中。"],
        ["训练充分性", "未通过", "训练图数量为 4，optimizer step 为 1，只能支持 smoke-level 结论。"],
        ["Embedding 几何信号", "弱通过", "silhouette 与 inter/intra 指标显示向量空间存在分离信号。"],
        ["聚类结果", "弱通过", "K=2 且 263/2 分布更像特殊 kernel 或离群点识别。"],
        ["baseline ablation", "缺失", "尚无 random、histogram、no-edge 或 edge-ablation 对照。"],
        ["multi-seed 稳定性", "缺失", "当前为单次运行，不能证明 K、assignment 和 representative 稳定。"],
        ["语义正确性", "未证明", "粗粒度 family label 不足以证明语义类别正确。"],
        ["下游代表性", "缺失", "metric_claim_status 为 unavailable，缺少 simulator/measured metric 验证。"],
    ]

    body: list[str] = []
    body.append(p("基于图对比学习的 ResNet-50 GPU 执行轨迹复现与 Kernel 聚类分析", "Title"))
    body.append(p("作者：复现实验小组"))
    body.append(p("单位：GPU 执行轨迹压缩与模拟复现实验项目组"))
    body.append(p("日期：2026 年 6 月 15 日"))
    body.append(
        p(
            "摘  要：GPU 体系结构模拟依赖细粒度执行轨迹，但完整 trace 的采集、存储和仿真成本较高。"
            "GCL-Sampler 思路尝试将 kernel 执行轨迹构造成图结构，通过图神经网络学习 kernel-level 表示，"
            "再利用聚类方法选择代表样本，从而为后续采样式模拟与调参提供依据。本文按照真实 ResNet-50 "
            "full trace 复现路径，构建从 NVBit trace、代表 SM 选择、canonical graph、图张量化、RGCN "
            "对比学习 embedding 到 K-means 聚类选择的端到端流程。实验输入包含 "
            f"{text(reproduction.get('input_kernel_invocation_count'))} 个 kernel invocation 和 "
            f"{text(reproduction.get('input_cta_record_count'))} 条 CTA 记录；RGCN 导出 "
            f"{text(training.get('export_graph_count'))} 个 {text(arch.get('kernel_embedding_dim'))} 维 kernel embedding。"
            f"基于轮廓系数的 K-means 选择 K={text(k_report.get('selected_k'))}，形成 263/2 的聚类分布，"
            f"embedding silhouette 为 {fmt_float(geom.get('silhouette'), 6)}，簇间/簇内平均距离比为 "
            f"{fmt_float(geom.get('inter_intra_ratio'), 6)}。结果表明当前工程通路已经闭合，embedding 空间存在弱正向分离信号；"
            "但训练充分性、baseline ablation、多随机种子稳定性、语义类别正确性和下游 representative 有效性仍未充分证明。"
            "因此本文将当前状态界定为结构有效但正确性未证明，为后续可信性评估和采样模拟实验提供基线。"
        )
    )
    body.append(p("关键词：GPU 执行轨迹；图对比学习；图神经网络；RGCN；K-means；ResNet-50；kernel 聚类"))
    body.append(p("中图分类号：TP391    文献标识码：A    DOI：待定"))
    body.append(p("Reproduction of GCL-based ResNet-50 GPU Trace Modeling and Kernel Clustering", "Heading1"))
    body.append(
        p(
            "Abstract: Fine-grained GPU simulation often requires complete execution traces, which are expensive to collect, store and replay. "
            "This work reproduces a GCL-style pipeline for ResNet-50 GPU traces. The pipeline transforms real NVBit traces into canonical typed graphs, "
            "exports graph tensors, trains a relation-aware graph neural encoder, produces kernel-level embeddings, and applies silhouette-guided K-means "
            "to select representative kernel groups. The reproduced full-trace run contains "
            f"{text(reproduction.get('input_kernel_invocation_count'))} kernel invocations and "
            f"{text(reproduction.get('input_cta_record_count'))} CTA records. The RGCN exports "
            f"{text(training.get('export_graph_count'))} embeddings with {text(arch.get('kernel_embedding_dim'))} dimensions. "
            f"The selected K is {text(k_report.get('selected_k'))}, yielding a 263/2 cluster distribution. "
            f"The silhouette score is {fmt_float(geom.get('silhouette'), 6)}, and the inter/intra distance ratio is "
            f"{fmt_float(geom.get('inter_intra_ratio'), 6)}. The current result validates the engineering path and shows a weak geometric separation signal, "
            "but it does not prove semantic cluster correctness or downstream representative usefulness."
        )
    )
    body.append(p("Key words: GPU trace; graph contrastive learning; graph neural network; RGCN; K-means; ResNet-50; kernel clustering"))

    body.extend(
        section(
            "1 引言",
            [
                "现代 GPU 程序包含大量 kernel 调用和线程块级并行执行行为。对于体系结构研究者而言，完整 trace 可以提供指令、访存、warp、CTA 与 SM 调度等细节，是评估模拟器、定位性能瓶颈和研究 trace 压缩的重要基础。然而，真实深度学习模型的 trace 规模通常很大，直接对所有 kernel 和所有线程块进行仿真会带来明显的时间和存储压力。",
                "GCL-Sampler 一类方法的核心动机是把执行轨迹转化为图结构表示，再利用图神经网络提取 kernel embedding，并在 embedding 空间中寻找代表性样本。与只依赖 opcode 直方图或手工统计特征的方法相比，图建模可以显式表达控制流、数据来源和数据目的等关系；与全量仿真相比，代表样本选择可以为后续采样式评估提供候选压缩路径。",
                "本文的目标不是直接宣称 GCL 已经在 ResNet-50 上获得最终正确的 kernel family 划分，而是按照复现工程证据，完整记录从真实 ResNet-50 NVBit trace 到 RGCN embedding 与 K-means selector 的流程、结果和局限。全文采用期刊论文式组织，重点回答三个问题：工程复现链路是否闭合，当前 embedding 空间是否出现可观察结构，以及现有证据距离可信语义分类和下游代表性还差什么。",
            ],
        )
    )

    body.extend(
        section(
            "2 GCL-Sampler 复现任务与总体流程",
            [
                "本文复现路径以真实 ResNet-50 inference trace 为输入，要求 formal path 不使用 synthetic trace、ResNet-like fixture、手写 opcode 序列或人工截断 kernel-only trace。当前 manifest 记录 run_scope 为 real_resnet50_full_trace，formal_full_trace_run 为 true，输入根目录为 artifacts/gcl_resnet50_gate0_formal_trace/traces。",
                "整体流程可概括为：真实 ResNet-50 NVBit trace 采集，trace adapter 规范化，代表 SM 选择，canonical graph 构建，graph tensorization，RGCN 对比学习训练与 embedding 导出，silhouette-K/K-means selector，cluster correctness evaluation，以及 Gate9 的 sampled-vs-full simulator evaluation 占位报告。当前 final_gate 为 gate9_report_only，说明 pipeline 已执行到报告阶段，但 Gate9 的有效 metric 仍未提供。",
                f"本次 full trace 输入规模为 {text(reproduction.get('input_kernel_invocation_count'))} 个 kernel invocation 和 {text(reproduction.get('input_cta_record_count'))} 条 CTA 记录。所有关键中间产物都记录了 hash，包括 adapter bundle、canonical graph bundle、graph tensor bundle、embedding table、selector manifest、Gate7 correctness manifest、Gate8 tuning vector proposal 和 Gate9 sampled-vs-full evaluation。",
            ],
        )
    )

    body.extend(
        section(
            "3 GPU 执行轨迹的图结构建模",
            [
                "图结构建模的基本思想是把 kernel 执行中的指令、访存引用和动态执行上下文抽象为节点，把节点之间的控制和数据关系抽象为类型化边。与规则网格数据不同，执行轨迹图具有节点数量可变、连接结构不规则、层级信息明显等特点，因此适合采用图神经网络进行表示学习。",
                "本次复现采用 strict path 的 mem_ref pseudo node 模式，边关系类型限定为 control_flow、data_source 和 data_destination。canonical graph 先在 warp 层级构建局部结构，再保留 warp、CTA、selected SM 与 kernel 的层级边界，后续 readout 使用 node_to_warp_to_cta_to_selected_sm_to_kernel 的层级聚合方式。",
                "这种建模方式的优点在于：第一，控制流边保留指令顺序和局部执行路径；第二，data_source 与 data_destination 边把访存引用作为显式关系纳入图结构；第三，层级 readout 避免把 kernel 简化为单一统计向量，使模型可以在不同执行层级逐步聚合信息。其局限也很明确：如果只使用 selected SM，则该图表示首先代表被选中 SM 的执行子结构，不能自动等价于 full GPU 的全部动态行为。",
            ],
        )
    )

    body.extend(
        section(
            "4 RGCN 图表示学习方法",
            [
                "图神经网络通过消息传递机制学习节点和图的向量表示。对于第 l 层节点表示 h_i^(l)，模型从邻居节点收集消息并结合节点自身状态进行更新。关系图卷积网络进一步区分不同类型的边，使不同关系可以拥有独立的变换参数或聚合权重，适合处理本文中的 control_flow、data_source 和 data_destination 三类边。",
                f"当前 RGCN manifest 记录模型为 {text(arch.get('layers'))} 层，输入维度为 {text(arch.get('input_dim'))}，隐藏维度为 {text(arch.get('hidden_dim'))}，kernel embedding 维度为 {text(arch.get('kernel_embedding_dim'))}，projection hidden/output 维度分别为 {text(arch.get('projection_hidden_dim'))}/{text(arch.get('projection_output_dim'))}，relation_count 为 {text(arch.get('relation_count'))}。训练使用 InfoNCE 对比学习目标，temperature 为 {text(training.get('contrastive_loss_config', {}).get('temperature'))}，优化器为 {text(opt.get('optimizer'))}，学习率为 {text(opt.get('learning_rate'))}。",
                f"需要注意的是，当前训练规模仍然较弱。manifest 显示 train_graph_count 为 {text(training.get('train_graph_count'))}，export_graph_count 为 {text(training.get('export_graph_count'))}，optimizer_step_count 为 {text(opt.get('optimizer_step_count'))}，checkpoint_reuse 为 {text(training.get('checkpoint_reuse'))}。这说明当前结果足以证明 RGCN path、checkpoint、embedding export 和 artifact lineage 可以跑通，但不足以证明模型已经充分收敛或学到了稳定语义表示。",
            ],
        )
    )

    body.extend(
        section(
            "5 K-means 聚类与代表 Kernel 选择",
            [
                "在获得 kernel-level embedding 后，本文使用 K-means 对嵌入向量进行无监督聚类。设全部 kernel embedding 为 Z={z1,z2,...,zn}，其中 zi 为 d 维向量。K-means 的目标是最小化每个样本到其所属簇中心的平方距离和，从而使同一簇内部样本尽可能接近，不同簇之间尽可能分离。",
                "标准 K-means 需要预先指定 K。本文使用轮廓系数在候选 K 中选择聚类数。轮廓系数同时考虑样本与同簇样本的平均距离和与最近异簇样本的平均距离，取值越接近 1 表示簇内越紧密、簇间越分离。完成聚类后，从每个簇中选择距离簇中心最近的 embedding 记录作为 representative anchor。",
                f"当前 selector 使用 mode={text(k_report.get('mode'))}，候选 K 的评分见表 1。最终 selected_k 为 {text(k_report.get('selected_k'))}，selected_score 为 {fmt_float(k_report.get('selected_score'), 8)}。聚类结果形成两个簇，其中 cluster 0 包含 263 个样本，cluster 1 包含 2 个样本。该分布说明 embedding 空间存在明显不均衡结构，更适合解释为少数特殊 kernel 或 outlier group 的识别，而不能直接解释为两个稳定的语义 kernel family。",
            ],
        )
    )
    body.append(p("表 1 K-means 候选 K 的轮廓系数"))
    body.append(table(k_rows))
    body.append(p("表 2 聚类分布与 family evidence"))
    body.append(table(cluster_rows))
    body.append(p("表 3 Representative anchor 质量摘要"))
    body.append(table(rep_rows))

    body.extend(
        section(
            "6 实验结果与有效性分析",
            [
                f"从 embedding geometry 看，当前 silhouette 为 {fmt_float(geom.get('silhouette'), 8)}，Davies-Bouldin 指数为 {fmt_float(geom.get('davies_bouldin'), 8)}，Calinski-Harabasz 指数为 {fmt_float(geom.get('calinski_harabasz'), 8)}。簇内平均距离为 {fmt_float(geom.get('intra_distance_mean'), 8)}，簇间平均距离为 {fmt_float(geom.get('inter_distance_mean'), 8)}，簇间/簇内距离比为 {fmt_float(geom.get('inter_intra_ratio'), 8)}。这些指标说明 embedding space 并非完全退化，聚类之间存在可观察的几何分离信号。",
                f"从 family alignment 看，cluster_purity 为 {fmt_float(family.get('cluster_purity'), 6)}，weighted_purity 为 {fmt_float(family.get('weighted_purity'), 6)}，但 ARI 为 {fmt_float(family.get('ari'), 6)}，NMI 为 {fmt_float(family.get('nmi'), 6)}，completeness 为 {fmt_float(family.get('completeness'), 6)}。这里的 purity 不能单独作为正确性证据，因为当前 family label 主要是 resnet50_real_trace 这样的粗粒度来源标签，而不是卷积、归一化、激活、残差连接等细粒度 operator family 标签。",
                f"从下游有效性看，cluster metric error report 的 metric_claim_status 为 {text(metric_report.get('metric_claim_status'))}，complete_row_count 为 {text(metric_report.get('complete_row_count'))}，global_weighted_mape 为 {text(metric_report.get('global_weighted_mape'))}。这说明当前没有足够 measured 或 simulator metric 来证明 representative 可以近似 cluster members 的运行时间、SM cycles、memory throughput 或仿真误差。",
                f"综合验收状态为 {text(acceptance.get('gnn_acceptance_status'))}，claim_status 为 {text(acceptance.get('claim_status'))}。因此，本文对当前 GCL 复现结果给出保守结论：工程通路已闭合，RGCN 结构与 embedding export 成立，embedding 空间出现弱正向分离信号；但不能声明 semantic cluster correctness，也不能声明 sampled representative 已经能够替代 full trace simulator evaluation。",
            ],
        )
    )
    body.append(p("表 4 GNN 验收状态摘要"))
    body.append(table(acceptance_rows))

    body.extend(
        section(
            "7 当前局限与后续改进",
            [
                "第一，训练充分性不足。当前只使用 4 个 graph 训练并执行 1 次 optimizer step，无法支撑深层语义表示已经稳定收敛的结论。后续应扩大训练图数量，记录 multi-epoch loss curve，并比较不同训练轮数下 embedding geometry 与 representative 质量的变化。",
                "第二，缺少 baseline ablation。当前不能排除聚类信号主要来自图规模、节点数量、opcode 统计或权重分布，而不是来自 RGCN 的关系感知消息传递。后续应增加 random embedding、opcode histogram、node feature pooling without edge、control-flow-only、data-flow-only 等对照。",
                "第三，缺少多随机种子稳定性。K-means 结果和 RGCN 训练都可能受到初始化影响。后续需要在多个 seed 下重复训练和聚类，报告 selected K 稳定性、assignment ARI、centroid drift 和 representative stability rate。",
                "第四，语义标签粒度不足。当前 family evidence 更多反映同一工作负载来源，而非真实 kernel 功能类别。后续应结合 kernel name、operator metadata、节点数、边数、指令类型、访存行为和 runtime metric 构建更细粒度的语义验证集。",
                "第五，下游 representative 价值尚未验证。最终目标应是用代表样本减少 full trace simulator evaluation 的成本，同时控制误差。因此后续必须补充 measured/simulator metric rows，计算 cluster 级 weighted MAPE、P95 relative error、rank correlation 和采样加速比。",
            ],
        )
    )

    body.extend(
        section(
            "8 结论",
            [
                "本文按照中文期刊论文的组织形式，总结了 ResNet-50 真实 GPU 执行轨迹上的 GCL 复现过程。复现链路从 NVBit full trace 出发，经过 trace adapter、代表 SM 选择、canonical graph 构建、图张量化、RGCN 对比学习、kernel embedding 导出、K-means 聚类和 Gate7 正确性评价，形成了可追踪的端到端 artifact。",
                f"实验结果显示，当前输入包含 {text(reproduction.get('input_kernel_invocation_count'))} 个 kernel invocation 和 {text(reproduction.get('input_cta_record_count'))} 条 CTA 记录，模型导出 {text(training.get('export_graph_count'))} 个 {text(arch.get('kernel_embedding_dim'))} 维 embedding。K-means 选择 K={text(k_report.get('selected_k'))}，得到 263/2 的聚类分布，embedding silhouette 为 {fmt_float(geom.get('silhouette'), 6)}，簇间/簇内平均距离比为 {fmt_float(geom.get('inter_intra_ratio'), 6)}。",
                "这些结果支持“结构有效、工程闭合、存在弱分离信号”的判断，但不支持“GNN 分类已经正确”或“representative 已经可替代 full trace 仿真”的强结论。后续工作应围绕训练充分性、baseline ablation、多 seed 稳定性、细粒度语义标签和下游仿真误差验证展开，逐步把当前 weak acceptance 推进到可支撑性能建模和采样模拟结论的 strong acceptance。",
            ],
        )
    )

    body.extend(
        section(
            "参考文献",
            [
                "[1] Kipf T N, Welling M. Semi-Supervised Classification with Graph Convolutional Networks[C]//International Conference on Learning Representations, 2017.",
                "[2] Schlichtkrull M, Kipf T N, Bloem P, et al. Modeling Relational Data with Graph Convolutional Networks[C]//European Semantic Web Conference, 2018.",
                "[3] Veličković P, Cucurull G, Casanova A, et al. Graph Attention Networks[C]//International Conference on Learning Representations, 2018.",
                "[4] Chen T, Kornblith S, Norouzi M, Hinton G. A Simple Framework for Contrastive Learning of Visual Representations[C]//International Conference on Machine Learning, 2020.",
                "[5] Lloyd S. Least Squares Quantization in PCM[J]. IEEE Transactions on Information Theory, 1982, 28(2):129-137.",
                "[6] Arthur D, Vassilvitskii S. k-means++: The Advantages of Careful Seeding[C]//ACM-SIAM Symposium on Discrete Algorithms, 2007.",
                "[7] Rousseeuw P J. Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis[J]. Journal of Computational and Applied Mathematics, 1987, 20:53-65.",
                "[8] NVIDIA. NVBit: A Dynamic Binary Instrumentation Framework for NVIDIA GPUs[EB/OL].",
                "[9] GCL ResNet-50 GNN Acceptance Report. docs/reports/gcl-resnet50-gnn-acceptance-report-2026-06-11.md.",
                "[10] A 线 GCL ResNet-50 GNN 有效性当前状态报告. docs/superpowers/specs/2026-06-13-a-line-gcl-resnet50-gnn-effectiveness-current-status.md.",
            ],
        )
    )

    sect_pr = (
        "<w:sectPr>"
        "<w:pgSz w:w=\"11906\" w:h=\"16838\"/>"
        "<w:pgMar w:top=\"1440\" w:right=\"1440\" w:bottom=\"1440\" w:left=\"1440\" w:header=\"720\" w:footer=\"720\" w:gutter=\"0\"/>"
        "</w:sectPr>"
    )
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<w:document xmlns:wpc=\"http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas\" "
        "xmlns:mc=\"http://schemas.openxmlformats.org/markup-compatibility/2006\" "
        "xmlns:o=\"urn:schemas-microsoft-com:office:office\" "
        "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" "
        "xmlns:m=\"http://schemas.openxmlformats.org/officeDocument/2006/math\" "
        "xmlns:v=\"urn:schemas-microsoft-com:vml\" "
        "xmlns:wp14=\"http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing\" "
        "xmlns:wp=\"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing\" "
        "xmlns:w10=\"urn:schemas-microsoft-com:office:word\" "
        "xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\" "
        "xmlns:w14=\"http://schemas.microsoft.com/office/word/2010/wordml\" "
        "xmlns:wpg=\"http://schemas.microsoft.com/office/word/2010/wordprocessingGroup\" "
        "xmlns:wpi=\"http://schemas.microsoft.com/office/word/2010/wordprocessingInk\" "
        "xmlns:wne=\"http://schemas.microsoft.com/office/word/2006/wordml\" "
        "xmlns:wps=\"http://schemas.microsoft.com/office/word/2010/wordprocessingShape\" "
        "mc:Ignorable=\"w14 wp14\"><w:body>"
        + "".join(body)
        + sect_pr
        + "</w:body></w:document>"
    )


def content_types_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""


def root_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""


def document_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>
"""


def styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:pPr><w:spacing w:after="120" w:line="360" w:lineRule="auto"/><w:jc w:val="both"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Times New Roman" w:eastAsia="宋体" w:hAnsi="Times New Roman"/><w:sz w:val="21"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr><w:jc w:val="center"/><w:spacing w:after="240"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Times New Roman" w:eastAsia="黑体" w:hAnsi="Times New Roman"/><w:b/><w:sz w:val="32"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:pPr><w:keepNext/><w:spacing w:before="240" w:after="120"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Times New Roman" w:eastAsia="黑体" w:hAnsi="Times New Roman"/><w:b/><w:sz w:val="26"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:pPr><w:keepNext/><w:spacing w:before="180" w:after="80"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Times New Roman" w:eastAsia="黑体" w:hAnsi="Times New Roman"/><w:b/><w:sz w:val="23"/></w:rPr>
  </w:style>
  <w:style w:type="table" w:styleId="TableGrid">
    <w:name w:val="Table Grid"/>
    <w:tblPr>
      <w:tblBorders>
        <w:top w:val="single" w:sz="4" w:space="0" w:color="auto"/>
        <w:left w:val="single" w:sz="4" w:space="0" w:color="auto"/>
        <w:bottom w:val="single" w:sz="4" w:space="0" w:color="auto"/>
        <w:right w:val="single" w:sz="4" w:space="0" w:color="auto"/>
        <w:insideH w:val="single" w:sz="4" w:space="0" w:color="auto"/>
        <w:insideV w:val="single" w:sz="4" w:space="0" w:color="auto"/>
      </w:tblBorders>
    </w:tblPr>
  </w:style>
</w:styles>
"""


def core_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>基于图对比学习的 ResNet-50 GPU 执行轨迹复现与 Kernel 聚类分析</dc:title>
  <dc:creator>GPU 执行轨迹压缩与模拟复现实验项目组</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">2026-06-15T00:00:00Z</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">2026-06-15T00:00:00Z</dcterms:modified>
</cp:coreProperties>
"""


def app_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex OpenXML generator</Application>
  <DocSecurity>0</DocSecurity>
  <ScaleCrop>false</ScaleCrop>
</Properties>
"""


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    xml = build_document_xml()
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types_xml())
        z.writestr("_rels/.rels", root_rels_xml())
        z.writestr("word/_rels/document.xml.rels", document_rels_xml())
        z.writestr("word/document.xml", xml)
        z.writestr("word/styles.xml", styles_xml())
        z.writestr("docProps/core.xml", core_xml())
        z.writestr("docProps/app.xml", app_xml())
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
