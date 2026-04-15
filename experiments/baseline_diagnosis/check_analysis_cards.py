from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.baseline_diagnosis.build_kernel_cards import default_kernel_names


REQUIRED_HEADERS = [
    "## Basic Info",
    "## Execution Mode",
    "## Key Observed Metrics",
    "## Dominant Resource Candidates",
    "## Family Decision",
    "## Evidence References",
]


def validate_card(path: Path) -> list[str]:
    text = path.read_text()
    errors: list[str] = []

    for header in REQUIRED_HEADERS:
        if header not in text:
            errors.append(f"missing header: {header}")

    evidence_lines = [line for line in text.splitlines() if line.strip().startswith("- [")]
    if len(evidence_lines) < 2:
        errors.append("fewer than 2 evidence references")
    if "baseline_ape.json" not in text:
        errors.append("missing baseline_ape.json evidence reference")

    if "boundary note:" not in text and "ambiguity / outlier note:" not in text:
        errors.append("missing boundary or uncertainty notes")

    return errors


def validate_card_directory(card_dir: Path) -> list[str]:
    failures = []
    existing = {path.stem for path in card_dir.glob("*.md")}
    expected = set(default_kernel_names())

    missing_cards = sorted(expected - existing)
    extra_cards = sorted(existing - expected)

    if missing_cards:
        failures.append(
            "missing expected cards: " + ", ".join(missing_cards)
        )
    if extra_cards:
        failures.append(
            "unexpected cards present: " + ", ".join(extra_cards)
        )

    for path in sorted(card_dir.glob("*.md")):
        errors = validate_card(path)
        if errors:
            failures.append(
                f"{path.name}: " + "; ".join(errors)
            )

    return failures


def main() -> int:
    card_dir = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else REPO_ROOT / "docs" / "family_criteria" / "mini_transformer_v4" / "analysis_cards"
    )
    failures = validate_card_directory(card_dir)
    if failures:
        for error in failures:
            print(error)
        return 1

    print(f"validated {len(default_kernel_names())} analysis cards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
