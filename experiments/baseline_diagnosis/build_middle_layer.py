from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "middle_layer" / "mini_transformer_v4"
DEFAULT_RULE_CONFIG_RELATIVE = Path(
    "docs/family_criteria/mini_transformer_v4/mini_transformer_middle_layer_rules_v1_2026-04-22.yaml"
)
DEFAULT_RULE_CONFIG = REPO_ROOT / DEFAULT_RULE_CONFIG_RELATIVE

LABEL_SCORES = {
    "Low": 0.30,
    "Low-Medium": 0.45,
    "Medium": 0.60,
    "Medium-High": 0.75,
    "High": 0.90,
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _round4(value: float) -> float:
    return round(value, 4)


def _label_score(label: str) -> float:
    return LABEL_SCORES[label]


def _display_rule_config_path(rule_config_path: Path, repo_root: Path) -> str:
    resolved_config = rule_config_path.resolve()
    resolved_repo_root = repo_root.resolve()
    try:
        return str(resolved_config.relative_to(resolved_repo_root))
    except ValueError:
        return str(resolved_config)


def _resolve_rule_config_path(repo_root: Path, rule_config_path: Path | None) -> Path:
    if rule_config_path is None:
        return repo_root.resolve() / DEFAULT_RULE_CONFIG_RELATIVE
    if rule_config_path.is_absolute():
        return rule_config_path.resolve()
    return (repo_root.resolve() / rule_config_path).resolve()


def _ape_key(short_name: str, grid_dim: str, block_dim: str) -> str:
    grid = f"({grid_dim.replace('x', ', ')})"
    block = f"({block_dim.replace('x', ', ')})"
    return f"{short_name}|{grid}|{block}"


def _per_kernel_by_id(full_json: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for kernel in full_json["per_kernel"].values():
        result[int(kernel["kernel_id"])] = kernel
    return result


def load_middle_layer_sources(repo_root: Path) -> dict[str, Any]:
    experiment_dir = repo_root / "experiments" / "mini_transformer"
    result_dir = repo_root / "experiments" / "baseline_diagnosis" / "results" / "mini_transformer_v4"
    required = {
        "full": experiment_dir / "mini_transformer_v4_full.json",
        "squash": experiment_dir / "mechanisms" / "squash.json",
        "batch": experiment_dir / "mechanisms" / "batch.json",
        "baseline_ape": result_dir / "baseline_ape.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        missing_list = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing middle-layer evidence files:\n{missing_list}")
    return {name: _load_json(path) if path.suffix == ".json" else path.read_text() for name, path in required.items()}


def load_middle_layer_rules(repo_root: Path, rule_config_path: Path | None = None) -> tuple[dict[str, Any], Path]:
    resolved_rule_config_path = _resolve_rule_config_path(repo_root, rule_config_path)
    if not resolved_rule_config_path.exists():
        raise FileNotFoundError(f"Missing middle-layer rule config: {resolved_rule_config_path}")
    rules = _load_yaml(resolved_rule_config_path)
    if not isinstance(rules, dict) or "families" not in rules:
        raise ValueError(f"Invalid middle-layer rule config: {resolved_rule_config_path}")
    return rules, resolved_rule_config_path


def _iter_anchor_specs(rules: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for family in rules["families"]:
        for anchor in family["anchors"]:
            merged = dict(anchor)
            merged["family_id"] = family["family_id"]
            specs.append(merged)
    return specs


def _validate_rule_config_kernel_coverage(
    per_kernel: dict[int, dict[str, Any]],
    anchor_specs: list[dict[str, Any]],
) -> None:
    configured_kernel_ids = [kernel_id for spec in anchor_specs for kernel_id in spec["kernel_ids"]]
    expected_kernel_ids = set(per_kernel)
    configured_kernel_id_set = set(configured_kernel_ids)

    duplicate_kernel_ids = sorted(
        kernel_id for kernel_id in configured_kernel_id_set if configured_kernel_ids.count(kernel_id) > 1
    )
    unknown_kernel_ids = sorted(configured_kernel_id_set - expected_kernel_ids)
    missing_kernel_ids = sorted(expected_kernel_ids - configured_kernel_id_set)

    errors: list[str] = []
    if duplicate_kernel_ids:
        errors.append(f"duplicate kernel ids: {duplicate_kernel_ids}")
    if unknown_kernel_ids:
        errors.append(f"unknown kernel ids: {unknown_kernel_ids}")
    if missing_kernel_ids:
        errors.append(f"missing kernel ids: {missing_kernel_ids}")
    if errors:
        raise ValueError("Invalid middle-layer rule coverage: " + "; ".join(errors))


def _validate_anchor_mechanism_evidence(
    sources: dict[str, Any],
    anchor_specs: list[dict[str, Any]],
) -> None:
    squash_segments = sources["squash"]["kernel_level"]["squash_segments"]
    kernel_to_squash_segments: dict[int, list[int]] = {}
    for segment in squash_segments:
        start, end = segment["kernel_range"]
        for kernel_id in range(start + 1, end + 2):
            kernel_to_squash_segments.setdefault(kernel_id, []).append(segment["segment_id"])

    batch_clusters = sources["batch"]["kernel_level"]["batch_clusters"]
    kernel_to_batch_cluster: dict[int, int] = {}
    for cluster in batch_clusters:
        for kernel_id in cluster["kernel_ids"]:
            kernel_to_batch_cluster[int(kernel_id)] = int(cluster["cluster_id"])
    batch_outliers = {int(kernel_id) for kernel_id in sources["batch"]["kernel_level"]["outlier_kernels"]}

    errors: list[str] = []
    for spec in anchor_specs:
        expected_segments = sorted(spec.get("expected_squash_segments", []))
        actual_segments = sorted(
            {segment_id for kernel_id in spec["kernel_ids"] for segment_id in kernel_to_squash_segments.get(kernel_id, [])}
        )
        if expected_segments != actual_segments:
            errors.append(
                f"{spec['anchor_id']} squash segments mismatch: expected {expected_segments}, got {actual_segments}"
            )

        expected_batch_cluster_id = spec.get("expected_batch_cluster_id")
        expected_batch_outlier = bool(spec.get("expected_batch_outlier", False))
        actual_clusters = sorted(
            {kernel_to_batch_cluster[kernel_id] for kernel_id in spec["kernel_ids"] if kernel_id in kernel_to_batch_cluster}
        )
        outlier_members = [kernel_id for kernel_id in spec["kernel_ids"] if kernel_id in batch_outliers]
        actual_outlier = all(kernel_id in batch_outliers for kernel_id in spec["kernel_ids"])
        if outlier_members and not actual_outlier:
            errors.append(
                f"{spec['anchor_id']} mixes batch outlier kernels {sorted(outlier_members)} with clustered kernels {actual_clusters}"
            )
        if expected_batch_outlier:
            if not actual_outlier:
                errors.append(f"{spec['anchor_id']} expected batch outlier membership, got clusters {actual_clusters}")
        else:
            if actual_outlier:
                errors.append(f"{spec['anchor_id']} expected batch cluster {expected_batch_cluster_id}, got outlier")
            elif actual_clusters != [expected_batch_cluster_id]:
                errors.append(
                    f"{spec['anchor_id']} batch cluster mismatch: expected {[expected_batch_cluster_id]}, got {actual_clusters}"
                )

    if errors:
        raise ValueError("Invalid middle-layer mechanism evidence: " + "; ".join(errors))


def _validate_anchor_member_shapes(members: list[dict[str, Any]], anchor_id: str) -> None:
    grid_dims = {member["dynamic_stats"]["grid_dim"] for member in members}
    block_dims = {member["dynamic_stats"]["block_dim"] for member in members}
    raw_names = {member["kernel_name"] for member in members}
    if len(grid_dims) != 1 or len(block_dims) != 1 or len(raw_names) != 1:
        raise ValueError(
            "Invalid middle-layer anchor membership: "
            f"{anchor_id} mixes heterogeneous launch shapes or raw kernels "
            f"(grid={sorted(grid_dims)}, block={sorted(block_dims)}, raw={sorted(raw_names)})"
        )
    return None


def _validate_unique_regime_ids(rules: dict[str, Any]) -> None:
    regime_ids: list[str] = []
    for family in rules["families"]:
        for regime in family["regimes"]:
            regime_ids.append(regime["regime_id"])
    duplicate_regime_ids = sorted({regime_id for regime_id in regime_ids if regime_ids.count(regime_id) > 1})
    if duplicate_regime_ids:
        raise ValueError(f"Invalid middle-layer rule config: duplicate regime ids: {duplicate_regime_ids}")


def build_anchor_records(sources: dict[str, Any], rules: dict[str, Any]) -> list[dict[str, Any]]:
    per_kernel = _per_kernel_by_id(sources["full"])
    ape_table = sources["baseline_ape"]["ape_table"]
    anchor_specs = _iter_anchor_specs(rules)
    _validate_rule_config_kernel_coverage(per_kernel, anchor_specs)
    _validate_anchor_mechanism_evidence(sources, anchor_specs)

    anchor_ape_keys: dict[str, str] = {}
    ape_key_to_anchor_ids: dict[str, list[str]] = {}
    for spec in anchor_specs:
        primary = per_kernel[spec["kernel_ids"][0]]
        grid_dim = primary["dynamic_stats"]["grid_dim"]
        block_dim = primary["dynamic_stats"]["block_dim"]
        ape_lookup_key = spec.get("ape_lookup_key_override") or _ape_key(
            spec["canonical_kernel_name"],
            grid_dim,
            block_dim,
        )
        anchor_ape_keys[spec["anchor_id"]] = ape_lookup_key
        ape_key_to_anchor_ids.setdefault(ape_lookup_key, []).append(spec["anchor_id"])

    total_invocations = sum(len(spec["kernel_ids"]) for spec in anchor_specs)
    total_cycles = sum(
        per_kernel[kernel_id]["hardware_metrics"]["elapsed_cycles"]
        for spec in anchor_specs
        for kernel_id in spec["kernel_ids"]
    )

    anchors: list[dict[str, Any]] = []
    for spec in anchor_specs:
        kernel_ids = spec["kernel_ids"]
        members = [per_kernel[kernel_id] for kernel_id in kernel_ids]
        _validate_anchor_member_shapes(members, spec["anchor_id"])
        primary = members[0]
        grid_dim = primary["dynamic_stats"]["grid_dim"]
        block_dim = primary["dynamic_stats"]["block_dim"]
        canonical_kernel_name = spec["canonical_kernel_name"]
        ape_lookup = anchor_ape_keys[spec["anchor_id"]]
        ape_evidence_unique = len(ape_key_to_anchor_ids[ape_lookup]) == 1
        ape_lookup_key = ape_lookup if ape_lookup in ape_table and ape_evidence_unique else None
        ape_evidence_status = (
            "unique"
            if ape_lookup in ape_table and ape_evidence_unique
            else "shared_across_anchors"
            if ape_lookup in ape_table
            else "missing_in_ape_table"
        )
        elapsed_cycles = sum(member["hardware_metrics"]["elapsed_cycles"] for member in members)
        coverage_ratio = len(kernel_ids) / total_invocations
        time_ratio = elapsed_cycles / total_cycles

        anchors.append(
            {
                "anchor_id": spec["anchor_id"],
                "family_id": spec["family_id"],
                "kernel_name": canonical_kernel_name,
                "kernel_name_raw": primary["kernel_name"],
                "phase_id": spec["phase_id"],
                "context_scope": spec["context_scope"],
                "semantic_route": spec["semantic_route"],
                "cluster_id": spec["cluster_id"],
                "member_invocations": kernel_ids,
                "coverage_count": len(kernel_ids),
                "observed_coverage_ratio": _round4(coverage_ratio),
                "observed_time_ratio": _round4(time_ratio),
                "coverage_weight": spec["coverage_label"],
                "coverage_weight_score": _round4(_label_score(spec["coverage_label"])),
                "time_weight": spec["time_label"],
                "time_weight_score": _round4(_label_score(spec["time_label"])),
                "coverage_label": spec["coverage_label"],
                "time_label": spec["time_label"],
                "decision_label": spec["decision_label"],
                "weight_source": {
                    "coverage": "derived",
                    "time": "derived",
                },
                "trace_order_summary": spec["trace_order_summary"],
                "grid_dim_summary": grid_dim,
                "block_dim_summary": block_dim,
                "shape_hint_summary": spec["shape_hint_summary"],
                "route_hint": spec["route_hint"],
                "template_hint": spec["template_hint"],
                "weighted_elapsed_cycles": _round4(elapsed_cycles),
                "ape_lookup_key": ape_lookup_key,
                "ape_lookup_key_candidate": ape_lookup,
                "ape_evidence_status": ape_evidence_status,
                "ape_elapsed_cycles_ape": (
                    ape_table[ape_lookup]["metrics"]["elapsed_cycles"]["ape"]
                    if ape_lookup in ape_table and ape_evidence_unique
                    else None
                ),
                "notes": spec["notes"],
            }
        )
    return anchors


def build_family_records(anchors: list[dict[str, Any]], rules: dict[str, Any]) -> list[dict[str, Any]]:
    anchor_by_id = {anchor["anchor_id"]: anchor for anchor in anchors}
    total_coverage_count = sum(anchor["coverage_count"] for anchor in anchors)
    total_weighted_cycles = sum(anchor["weighted_elapsed_cycles"] for anchor in anchors)

    families: list[dict[str, Any]] = []
    for family in rules["families"]:
        selected = [anchor_by_id[anchor["anchor_id"]] for anchor in family["anchors"]]
        coverage_ratio = sum(anchor["coverage_count"] for anchor in selected) / total_coverage_count
        time_ratio = sum(anchor["weighted_elapsed_cycles"] for anchor in selected) / total_weighted_cycles
        importance = (
            0.3 * _label_score(family["coverage_label"])
            + 0.4 * _label_score(family["time_label"])
            + 0.3 * _label_score(family["decision_label"])
        )
        families.append(
            {
                "family_id": family["family_id"],
                "input_anchor_ids": [anchor["anchor_id"] for anchor in family["anchors"]],
                "phase_scope": family["phase_scope"],
                "route_primitive": family["route_primitive"],
                "hardware_template": family["hardware_template"],
                "boundary_status": family["boundary_status"],
                "boundary_notes": family["boundary_notes"],
                "shape_regime_summary": family["shape_regime_summary"],
                "resource_signature_summary": family["resource_signature_summary"],
                "observed_coverage_ratio": _round4(coverage_ratio),
                "observed_time_ratio": _round4(time_ratio),
                "coverage_weight": family["coverage_label"],
                "coverage_weight_score": _round4(_label_score(family["coverage_label"])),
                "time_weight": family["time_label"],
                "time_weight_score": _round4(_label_score(family["time_label"])),
                "decision_weight": family["decision_label"],
                "decision_weight_score": _round4(_label_score(family["decision_label"])),
                "coverage_label": family["coverage_label"],
                "time_label": family["time_label"],
                "decision_label": family["decision_label"],
                "decision_weight_factors": family["decision_weight_factors"],
                "decision_weight_note": family["decision_weight_note"],
                "weight_source": family["weight_source_status"],
                "importance_score": _round4(importance),
                "priority_class": family["priority_class"],
                "recommended_tuning_target": family["recommended_tuning_target"],
                "notes": family["notes"],
            }
        )
    return families


def build_regime_records(
    anchors: list[dict[str, Any]],
    families: list[dict[str, Any]],
    rules: dict[str, Any],
) -> list[dict[str, Any]]:
    anchor_by_id = {anchor["anchor_id"]: anchor for anchor in anchors}
    family_by_id = {family["family_id"]: family for family in families}
    total_coverage_count = sum(anchor["coverage_count"] for anchor in anchors)
    total_weighted_cycles = sum(anchor["weighted_elapsed_cycles"] for anchor in anchors)
    _validate_unique_regime_ids(rules)

    regimes: list[dict[str, Any]] = []
    for family in rules["families"]:
        for regime in family["regimes"]:
            selected = [anchor_by_id[anchor_id] for anchor_id in regime["source_anchor_ids"]]
            coverage_ratio = sum(anchor["coverage_count"] for anchor in selected) / total_coverage_count
            time_ratio = sum(anchor["weighted_elapsed_cycles"] for anchor in selected) / total_weighted_cycles
            family_importance = family_by_id[family["family_id"]]["importance_score"]
            regime_priority = (
                0.35 * family_importance
                + 0.25 * _label_score(regime["coverage_label"])
                + 0.25 * _label_score(regime["time_label"])
                + 0.15 * _label_score(regime["local_decision_label"])
            )
            regimes.append(
                {
                    "regime_id": regime["regime_id"],
                    "family_id": family["family_id"],
                    "source_anchor_ids": regime["source_anchor_ids"],
                    "phase_id": regime["phase_id"],
                    "route_primitive": regime["route_primitive"],
                    "hardware_template": regime["hardware_template"],
                    "shape_regime": regime["shape_regime"],
                    "context_scope": regime["context_scope"],
                    "resource_signature": regime["resource_signature"],
                    "observed_coverage_ratio": _round4(coverage_ratio),
                    "observed_time_ratio": _round4(time_ratio),
                    "coverage_weight": regime["coverage_label"],
                    "coverage_weight_score": _round4(_label_score(regime["coverage_label"])),
                    "time_weight": regime["time_label"],
                    "time_weight_score": _round4(_label_score(regime["time_label"])),
                    "local_decision_weight": regime["local_decision_label"],
                    "local_decision_weight_score": _round4(_label_score(regime["local_decision_label"])),
                    "coverage_label": regime["coverage_label"],
                    "time_label": regime["time_label"],
                    "family_importance_score": family_importance,
                    "local_decision_label": regime["local_decision_label"],
                    "decision_weight_factors": regime["decision_weight_factors"],
                    "decision_weight_note": regime["decision_weight_note"],
                    "weight_source": regime["weight_source_status"],
                    "regime_priority_score": _round4(regime_priority),
                    "simulator_lane_id": regime["lane"]["lane_id"],
                    "validation_status": regime["validation_status"],
                    "notes": regime["notes"],
                }
            )
    return regimes


def build_lane_records(rules: dict[str, Any]) -> list[dict[str, Any]]:
    lanes: list[dict[str, Any]] = []
    for family in rules["families"]:
        for regime in family["regimes"]:
            lane = regime["lane"]
            lanes.append(
                {
                    "lane_id": lane["lane_id"],
                    "target_regime_id": regime["regime_id"],
                    "target_family_id": family["family_id"],
                    "lane_goal": lane["lane_goal"],
                    "parameter_direction": lane["parameter_direction"],
                    "baseline_type": lane["baseline_type"],
                    "validation_metric": lane["validation_metric"],
                    "writeback_target": lane["writeback_target"],
                    "notes": lane["notes"],
                }
            )
    return lanes


def build_importance_scoring_sheet(
    families: list[dict[str, Any]],
    regimes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in families:
        rows.append(
            {
                "object_level": "family",
                "object_id": family["family_id"],
                "parent_family_id": family["family_id"],
                "coverage_weight": family["coverage_weight"],
                "coverage_weight_score": family["coverage_weight_score"],
                "time_weight": family["time_weight"],
                "time_weight_score": family["time_weight_score"],
                "decision_weight": family["decision_weight"],
                "decision_weight_score": family["decision_weight_score"],
                "family_importance_score": family["importance_score"],
                "local_decision_weight": None,
                "regime_priority_score": None,
                "importance_score": family["importance_score"],
                "priority_class": family["priority_class"],
                "weight_source": family["weight_source"],
                "decision_weight_note": family["decision_weight_note"],
            }
        )
    for regime in regimes:
        rows.append(
            {
                "object_level": "regime",
                "object_id": regime["regime_id"],
                "parent_family_id": regime["family_id"],
                "coverage_weight": regime["coverage_weight"],
                "coverage_weight_score": regime["coverage_weight_score"],
                "time_weight": regime["time_weight"],
                "time_weight_score": regime["time_weight_score"],
                "decision_weight": regime["local_decision_weight"],
                "decision_weight_score": regime["local_decision_weight_score"],
                "family_importance_score": regime["family_importance_score"],
                "local_decision_weight": regime["local_decision_weight"],
                "regime_priority_score": regime["regime_priority_score"],
                "importance_score": regime["regime_priority_score"],
                "priority_class": None,
                "weight_source": regime["weight_source"],
                "decision_weight_note": regime["decision_weight_note"],
            }
        )
    return rows


def build_writeback_lane_to_regime(
    lanes: list[dict[str, Any]],
    regimes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    regime_by_id = {regime["regime_id"]: regime for regime in regimes}
    rows: list[dict[str, Any]] = []
    for lane in lanes:
        regime = regime_by_id[lane["target_regime_id"]]
        rows.append(
            {
                "lane_id": lane["lane_id"],
                "target_regime_id": lane["target_regime_id"],
                "target_family_id": lane["target_family_id"],
                "writeback_target": lane["writeback_target"],
                "writeback_chain": {
                    "lane_to_regime": lane["target_regime_id"],
                    "regime_to_family": regime["family_id"],
                    "family_to_workload_explanation": lane["writeback_target"],
                },
                "parameter_direction": lane["parameter_direction"],
                "validation_metric": lane["validation_metric"],
            }
        )
    return rows


def build_middle_layer_artifacts(
    repo_root: Path | None = None,
    rule_config_path: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root or REPO_ROOT
    sources = load_middle_layer_sources(repo_root)
    rules, resolved_rule_config_path = load_middle_layer_rules(repo_root, rule_config_path)
    anchors = build_anchor_records(sources, rules)
    families = build_family_records(anchors, rules)
    regimes = build_regime_records(anchors, families, rules)
    lanes = build_lane_records(rules)
    importance_scoring_sheet = build_importance_scoring_sheet(families, regimes)
    writeback_lane_to_regime = build_writeback_lane_to_regime(lanes, regimes)
    return {
        "metadata": {
            "workload": rules["workload"],
            "builder": "experiments/baseline_diagnosis/build_middle_layer.py",
            "rule_config_path": _display_rule_config_path(resolved_rule_config_path, repo_root),
            "rule_config_version": rules["rule_config_version"],
            "importance_formula": "0.3*coverage_label + 0.4*time_label + 0.3*decision_label",
            "regime_priority_formula": "0.35*family_importance + 0.25*coverage_label + 0.25*time_label + 0.15*local_decision_label",
            "notes": "Observed coverage/time ratios come from kernel invocation membership and weighted elapsed cycles; qualitative labels remain provisional and are backed by rule notes in the YAML config.",
        },
        "anchors": anchors,
        "families": families,
        "regimes": regimes,
        "lanes": lanes,
        "importance_scoring_sheet": importance_scoring_sheet,
        "writeback_lane_to_regime": writeback_lane_to_regime,
    }


def _markdown_table(records: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "|---" * len(columns) + "|"
    rows = []
    for record in records:
        rows.append("| " + " | ".join(str(record.get(column, "")) for column in columns) + " |")
    return "\n".join([header, separator, *rows])


def render_markdown_snapshots(bundle: dict[str, Any]) -> dict[str, str]:
    metadata = bundle["metadata"]
    header = [
        "# mini_transformer_v4 Middle Layer Artifacts",
        "",
        f"- workload: `{metadata['workload']}`",
        f"- builder: `{metadata['builder']}`",
        f"- rule config: `{metadata['rule_config_path']}`",
        f"- rule config version: `{metadata['rule_config_version']}`",
        f"- importance formula: `{metadata['importance_formula']}`",
        f"- regime priority formula: `{metadata['regime_priority_formula']}`",
        "",
    ]
    anchors_md = "\n".join(
        header
        + [
            "## Anchors",
            "",
            _markdown_table(
                bundle["anchors"],
                [
                    "anchor_id",
                    "family_id",
                    "kernel_name",
                    "phase_id",
                    "context_scope",
                    "member_invocations",
                    "observed_coverage_ratio",
                    "observed_time_ratio",
                    "coverage_label",
                    "time_label",
                ],
            ),
            "",
        ]
    )
    families_md = "\n".join(
        header
        + [
            "## Families",
            "",
            _markdown_table(
                bundle["families"],
                [
                    "family_id",
                    "input_anchor_ids",
                    "observed_coverage_ratio",
                    "observed_time_ratio",
                    "coverage_label",
                    "time_label",
                    "decision_label",
                    "importance_score",
                    "priority_class",
                ],
            ),
            "",
        ]
    )
    regimes_md = "\n".join(
        header
        + [
            "## Regimes",
            "",
            _markdown_table(
                bundle["regimes"],
                [
                    "regime_id",
                    "family_id",
                    "source_anchor_ids",
                    "observed_coverage_ratio",
                    "observed_time_ratio",
                    "local_decision_label",
                    "regime_priority_score",
                    "simulator_lane_id",
                    "validation_status",
                ],
            ),
            "",
        ]
    )
    lanes_md = "\n".join(
        header
        + [
            "## Lanes",
            "",
            _markdown_table(
                bundle["lanes"],
                [
                    "lane_id",
                    "target_regime_id",
                    "target_family_id",
                    "parameter_direction",
                    "baseline_type",
                    "validation_metric",
                ],
            ),
            "",
        ]
    )
    scoring_md = "\n".join(
        header
        + [
            "## Importance Scoring Sheet",
            "",
            _markdown_table(
                bundle["importance_scoring_sheet"],
                [
                    "object_level",
                    "object_id",
                    "parent_family_id",
                    "coverage_weight",
                    "time_weight",
                    "decision_weight",
                    "family_importance_score",
                    "local_decision_weight",
                    "regime_priority_score",
                    "importance_score",
                ],
            ),
            "",
        ]
    )
    writeback_md = "\n".join(
        header
        + [
            "## Writeback Lane To Regime",
            "",
            _markdown_table(
                bundle["writeback_lane_to_regime"],
                [
                    "lane_id",
                    "target_regime_id",
                    "target_family_id",
                    "writeback_target",
                    "parameter_direction",
                ],
            ),
            "",
        ]
    )
    return {
        "anchors.md": anchors_md,
        "families.md": families_md,
        "regimes.md": regimes_md,
        "lanes.md": lanes_md,
        "importance_scoring_sheet.md": scoring_md,
        "writeback_lane_to_regime.md": writeback_md,
    }


def write_middle_layer_artifacts(bundle: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "anchors",
        "families",
        "regimes",
        "lanes",
        "importance_scoring_sheet",
        "writeback_lane_to_regime",
    ):
        (output_dir / f"{name}.json").write_text(json.dumps(bundle[name], indent=2, ensure_ascii=True) + "\n")
    (output_dir / "bundle.json").write_text(json.dumps(bundle, indent=2, ensure_ascii=True) + "\n")
    for name, text in render_markdown_snapshots(bundle).items():
        (output_dir / name).write_text(text + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build middle-layer artifacts for mini_transformer_v4")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated middle-layer artifacts.",
    )
    parser.add_argument(
        "--rule-config",
        type=Path,
        default=DEFAULT_RULE_CONFIG,
        help="Path to single-file YAML rule config.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = build_middle_layer_artifacts(REPO_ROOT, args.rule_config)
    write_middle_layer_artifacts(bundle, args.output_dir)
    print(f"wrote middle-layer artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
