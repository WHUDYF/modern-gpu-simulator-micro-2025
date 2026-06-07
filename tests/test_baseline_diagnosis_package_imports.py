import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def test_baseline_diagnosis_entrypoints_import_via_package_path():
    for module_name in [
        "experiments.baseline_diagnosis.pka_m0_pipeline",
        "experiments.baseline_diagnosis.pka_m1_selector",
        "experiments.baseline_diagnosis.run_m1_measured_loop",
        "experiments.baseline_diagnosis.m1_measured_feature_extractor",
        "experiments.baseline_diagnosis.m1_ncu_capture_dispatcher",
        "experiments.baseline_diagnosis.m1_selector_eligibility",
        "experiments.baseline_diagnosis.pka_selector_core",
    ]:
        assert importlib.import_module(module_name)
