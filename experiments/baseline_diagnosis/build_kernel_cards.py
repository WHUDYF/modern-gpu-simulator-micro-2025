from pathlib import Path


def load_sources(repo_root: Path) -> dict[str, Path]:
    result_dir = (
        repo_root
        / "experiments"
        / "baseline_diagnosis"
        / "results"
        / "mini_transformer_v4"
    )
    return {
        "E0_baseline": result_dir / "E0_baseline.md",
        "E1_squash": result_dir / "E1_squash.md",
        "E2_batch": result_dir / "E2_batch.md",
        "E4_full": result_dir / "E4_full.md",
        "E5_stageC_validation": result_dir / "E5_stageC_validation.md",
        "baseline_ape": result_dir / "baseline_ape.json",
    }


def default_kernel_names() -> list[str]:
    return [
        "gemm_tiled",
        "attention_score",
        "residual_add",
        "softmax_kernel",
        "context_mul",
        "layernorm_kernel",
    ]
