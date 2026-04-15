from pathlib import Path
import sys


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

    if "boundary note:" not in text and "ambiguity / outlier note:" not in text:
        errors.append("missing boundary or uncertainty notes")

    return errors


def main() -> int:
    card_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/home/dyf/modern-gpu-simulator-micro-2025/docs/family_criteria/mini_transformer_v4/analysis_cards"
    )
    failures = []
    for path in sorted(card_dir.glob("*.md")):
        errors = validate_card(path)
        if errors:
            failures.append((path.name, errors))

    if failures:
        for name, errors in failures:
            print(f"{name}:")
            for err in errors:
                print(f"  - {err}")
        return 1

    print(f"validated {len(list(card_dir.glob('*.md')))} analysis cards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
