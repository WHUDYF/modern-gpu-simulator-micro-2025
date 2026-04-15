from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.baseline_diagnosis.check_analysis_cards import validate_card_directory


CARD_TEMPLATE = """# Kernel Analysis Card: `{name}`

## Basic Info

- kernel name: `{name}`
- operator semantics: sample op
- workload role: sample role

## Execution Mode

- tentative mode: `mixed`

## Key Observed Metrics

- sample metric

## Dominant Resource Candidates

- primary: `cache / locality`

## Family Decision

- tentative family: `mixed -> sample`
- boundary note: sample boundary note

## Evidence References

- [ref1](ref1.txt): sample
- [ref2](ref2.txt): sample
- [baseline_ape.json](baseline_ape.json): sample
"""


def test_validate_card_directory_fails_when_expected_cards_are_missing(tmp_path):
    (tmp_path / "ref1.txt").write_text("ref1")
    (tmp_path / "ref2.txt").write_text("ref2")
    (tmp_path / "baseline_ape.json").write_text("{}")
    only_card = tmp_path / "gemm_tiled.md"
    only_card.write_text(CARD_TEMPLATE.format(name="gemm_tiled"))

    failures = validate_card_directory(tmp_path)

    assert any("missing expected cards" in failure for failure in failures)


def test_validate_card_directory_passes_for_canonical_card_set(tmp_path):
    (tmp_path / "ref1.txt").write_text("ref1")
    (tmp_path / "ref2.txt").write_text("ref2")
    (tmp_path / "baseline_ape.json").write_text("{}")
    names = [
        "gemm_tiled",
        "attention_score",
        "residual_add",
        "softmax_kernel",
        "context_mul",
        "layernorm_kernel",
    ]
    for name in names:
        (tmp_path / f"{name}.md").write_text(CARD_TEMPLATE.format(name=name))

    failures = validate_card_directory(tmp_path)

    assert failures == []


def test_validate_card_directory_fails_when_schema_fields_are_missing(tmp_path):
    (tmp_path / "ref1.txt").write_text("ref1")
    (tmp_path / "ref2.txt").write_text("ref2")
    (tmp_path / "baseline_ape.json").write_text("{}")
    broken = tmp_path / "gemm_tiled.md"
    broken.write_text(
        """# Kernel Analysis Card: `gemm_tiled`

## Basic Info

- kernel name: `gemm_tiled`

## Execution Mode

- tentative mode: `mixed`

## Key Observed Metrics

- sample metric

## Dominant Resource Candidates

- primary: `cache / locality`

## Family Decision

- boundary note: sample boundary note

## Evidence References

- [ref1](ref1.txt): sample
- [ref2](ref2.txt): sample
- [baseline_ape.json](baseline_ape.json): sample
"""
    )

    names = [
        "attention_score",
        "residual_add",
        "softmax_kernel",
        "context_mul",
        "layernorm_kernel",
    ]
    for name in names:
        (tmp_path / f"{name}.md").write_text(CARD_TEMPLATE.format(name=name))

    failures = validate_card_directory(tmp_path)

    assert any("missing required field content: - operator semantics:" in failure for failure in failures)
    assert any("missing required field content: - tentative family:" in failure for failure in failures)


def test_validate_card_directory_fails_when_evidence_target_is_missing(tmp_path):
    (tmp_path / "ref1.txt").write_text("ref1")
    (tmp_path / "baseline_ape.json").write_text("{}")

    names = [
        "gemm_tiled",
        "attention_score",
        "residual_add",
        "softmax_kernel",
        "context_mul",
        "layernorm_kernel",
    ]
    for name in names:
        (tmp_path / f"{name}.md").write_text(CARD_TEMPLATE.format(name=name))

    failures = validate_card_directory(tmp_path)

    assert any("missing evidence target: ref2.txt" in failure for failure in failures)
