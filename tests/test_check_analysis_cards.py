from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.baseline_diagnosis.check_analysis_cards import validate_card_directory


CARD_TEMPLATE = """# Kernel Analysis Card: `{name}`

## Basic Info

- kernel name: `{name}`

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

- [ref1](/tmp/ref1): sample
- [ref2](/tmp/ref2): sample
"""


def test_validate_card_directory_fails_when_expected_cards_are_missing(tmp_path):
    only_card = tmp_path / "gemm_tiled.md"
    only_card.write_text(CARD_TEMPLATE.format(name="gemm_tiled"))

    failures = validate_card_directory(tmp_path)

    assert any("missing expected cards" in failure for failure in failures)


def test_validate_card_directory_passes_for_canonical_card_set(tmp_path):
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
