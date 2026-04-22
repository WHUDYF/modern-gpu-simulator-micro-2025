"""Export mainline A-line frontend artifacts."""

from __future__ import annotations

from statistics import variance
from typing import Any


def export_anchor_table(selector_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total_members = sum(group["member_count"] for group in selector_groups) or 1
    total_exec_time = sum(
        member.get("exec_time") or 0
        for group in selector_groups
        for member in group["members"]
    ) or 1.0

    table = []
    for idx, group in enumerate(selector_groups, start=1):
        members = group["members"]
        anchor = group["anchor_record"]
        member_ids = [m["kernel_invocation_id"] for m in members]
        member_exec = [m.get("exec_time") or 0 for m in members]
        table.append(
            {
                "output_role": "mainline_anchor",
                "rep_kernel_id": f"rep-{group['method']}-{idx}",
                "kernel_name": anchor["kernel_name"],
                "cluster_id": group["cluster_id"],
                "member_invocations": member_ids,
                "coverage_count": len(member_ids),
                "coverage_weight": len(member_ids) / total_members,
                "coverage_weight_source": "derived_from_member_count",
                "time_weight": sum(member_exec) / total_exec_time,
                "time_weight_source": "derived_from_exec_time",
                "trace_order_summary": {
                    "start": members[0]["trace_order"],
                    "end": members[-1]["trace_order"],
                },
                "shape_hint_summary": None,
                "grid_dim": anchor.get("grid_dim"),
                "block_dim": anchor.get("block_dim"),
                "member_invocations_status": "full_list",
                "heterogeneity_flag": group["heterogeneity_flag"],
                "squash_boundary_crossing_flag": group.get("squash_boundary_crossing_flag", False),
                "notes": group.get("guardrail_note"),
            }
        )
    _validate_anchor_table(table)
    return table


def _safe_variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(variance(values))


def _validate_anchor_table(table: list[dict[str, Any]]) -> None:
    required = {
        "rep_kernel_id",
        "kernel_name",
        "cluster_id",
        "member_invocations",
        "coverage_count",
        "coverage_weight",
        "time_weight",
    }
    forbidden = {
        "family_id",
        "regime_id",
        "route_primitive",
        "execution_template",
        "execution_template_label",
        "simulator_lane_id",
    }
    for row in table:
        missing = required - set(row.keys())
        if missing:
            raise ValueError(f"anchor table row missing required fields: {sorted(missing)}")
        leaked = forbidden & set(row.keys())
        if leaked:
            raise ValueError(f"anchor table row contains forbidden downstream keys: {sorted(leaked)}")


def build_comparison_table(all_groups_by_method: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    table = []
    for method, groups in all_groups_by_method.items():
        total_exec = sum(
            member.get("exec_time") or 0
            for group in groups
            for member in group["members"]
        ) or 1.0
        anchor_time_weights = sorted(
            [
                sum(member.get("exec_time") or 0 for member in group["members"]) / total_exec
                for group in groups
            ],
            reverse=True,
        )
        top3_coverage = sum(anchor_time_weights[:3])
        avg_cluster_size = sum(group["member_count"] for group in groups) / max(len(groups), 1)
        exec_vars = []
        inst_vars = []
        split_count = 0
        split_tracker: dict[str, int] = {}
        for group in groups:
            split_tracker.setdefault(group["anchor_record"]["kernel_name"], 0)
            split_tracker[group["anchor_record"]["kernel_name"]] += 1
            exec_vals = [m.get("exec_time") or 0 for m in group["members"]]
            inst_vals = [m.get("dynamic_inst_count") or 0 for m in group["members"]]
            exec_vars.append(_safe_variance(exec_vals))
            inst_vars.append(_safe_variance(inst_vals))
        split_count = sum(1 for count in split_tracker.values() if count > 1)
        table.append(
            {
                "output_role": "evidence_only",
                "method": method,
                "num_anchors": len(groups),
                "time_weight_covered": top3_coverage,
                "time_weight_covered_definition": "cumulative time weight covered by the top-3 anchors in this method",
                "avg_cluster_size": avg_cluster_size,
                "intra_cluster_exec_time_var": sum(exec_vars) / max(len(exec_vars), 1),
                "intra_cluster_inst_var": sum(inst_vars) / max(len(inst_vars), 1),
                "split_cases_count": split_count,
                "notes": "evidence-only output; not a downstream mainline table",
            }
        )
    return table


def build_case_note(all_groups_by_method: dict[str, list[dict[str, Any]]]) -> str:
    by_method = {method: {} for method in all_groups_by_method}
    for method, groups in all_groups_by_method.items():
        for group in groups:
            by_method[method].setdefault(group["anchor_record"]["kernel_name"], []).append(group)

    lines = [
        "# Frontend Anchor Case Note",
        "",
        "_Output role: evidence_only_",
        "",
        "## Representative split cases",
        "",
    ]
    for kernel_name, hybrid_groups in by_method.get("hybrid", {}).items():
        coarse_groups = by_method.get("pka-like-coarse", {}).get(kernel_name, [])
        name_groups = by_method.get("name-only", {}).get(kernel_name, [])
        if len(hybrid_groups) <= max(len(coarse_groups), len(name_groups)):
            continue
        lines.append(f"### {kernel_name}")
        lines.append(
            f"- `name-only` groups: {len(name_groups)}; "
            f"`PKA-like coarse` groups: {len(coarse_groups)}; "
            f"`hybrid` groups: {len(hybrid_groups)}"
        )
        for coarse_group in coarse_groups:
            coarse_member_ids = [m["kernel_invocation_id"] for m in coarse_group["members"]]
            lines.append(
                f"- coarse cluster `{coarse_group['cluster_id']}` keeps merged members: "
                f"{', '.join(coarse_member_ids)}"
            )
        for group in hybrid_groups:
            members = group["members"]
            member_ids = [m["kernel_invocation_id"] for m in members]
            grid_dims = sorted({str(m.get("grid_dim")) for m in members})
            dynamic_insts = sorted({m.get("dynamic_inst_count") for m in members})
            lines.append(f"- hybrid cluster `{group['cluster_id']}` members: {', '.join(member_ids)}")
            lines.append(
                f"  - evidence: grid_dim={grid_dims}, dynamic_inst_count={dynamic_insts}"
            )
        lines.append(
            "  - interpretation: `PKA-like coarse` still merges these invocations inside one coarse bucket, "
            "while `hybrid` splits out the subgroup(s) whose grid size or dynamic instruction volume diverges "
            "enough to justify a separate frontend anchor."
        )
        lines.append("")

    if len(lines) == 6:
        lines.extend(["No additional hybrid split case was observed on this input.", ""])
    return "\n".join(lines)
