from pathlib import Path
import sys
import json


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.trace_compression_behavior.catalog import load_catalog, load_catalog_records


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


def test_load_catalog_records_resolves_catalog_relative_fixture_paths():
    catalog = load_catalog(FIXTURE)
    records = load_catalog_records(catalog)

    assert records["regular_memory_fixture"]["dynamic_stats"]["total_dynamic_insts"] == 1000000.0


def test_load_catalog_resolves_relative_source_paths_from_catalog_directory(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source = data_dir / "record.json"
    source.write_text(json.dumps({"record": {"value": 7}}), encoding="utf-8")
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "catalog_id": "external_catalog",
                "entries": [
                    {
                        "id": "external_record",
                        "label": "External Record",
                        "role": "target",
                        "source_path": "data/record.json",
                        "record_pointer": "/record",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    catalog = load_catalog(catalog_path)
    records = load_catalog_records(catalog)

    assert catalog.entries[0].source_path == source.resolve()
    assert records["external_record"]["value"] == 7
