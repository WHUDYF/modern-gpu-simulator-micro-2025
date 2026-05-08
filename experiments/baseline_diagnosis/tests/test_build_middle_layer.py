from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.baseline_diagnosis.build_middle_layer import load_middle_layer_sources


def test_middle_layer_sources_do_not_require_unused_e5_report(tmp_path):
    repo_root = tmp_path
    experiment_dir = repo_root / "experiments" / "mini_transformer"
    mechanism_dir = experiment_dir / "mechanisms"
    result_dir = repo_root / "experiments" / "baseline_diagnosis" / "results" / "mini_transformer_v4"
    mechanism_dir.mkdir(parents=True)
    result_dir.mkdir(parents=True)
    (experiment_dir / "mini_transformer_v4_full.json").write_text(json.dumps({"per_kernel": {}}))
    (mechanism_dir / "squash.json").write_text(json.dumps({}))
    (mechanism_dir / "batch.json").write_text(json.dumps({}))
    (result_dir / "baseline_ape.json").write_text(json.dumps({}))

    sources = load_middle_layer_sources(repo_root)

    assert set(sources) == {"full", "squash", "batch", "baseline_ape"}
