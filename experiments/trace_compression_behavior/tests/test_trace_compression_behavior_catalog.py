from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.trace_compression_behavior.catalog import load_catalog


FIXTURE = ROOT / "experiments/trace_compression_behavior/fixtures/synthetic_catalog.json"


def test_load_catalog_reads_entries_with_stable_ids():
    catalog = load_catalog(FIXTURE)

    assert catalog.catalog_id == "synthetic_fixture_probe"
    assert [entry.id for entry in catalog.entries] == [
        "regular_memory_fixture",
        "irregular_gather_fixture",
        "branch_heavy_fixture",
    ]
    assert catalog.entries[0].role == "target"
