from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_measured_loop_modules_import_under_package_namespace():
    from experiments.baseline_diagnosis import (
        m1_measured_feature_extractor,
        m1_ncu_capture_dispatcher,
        m1_selector_eligibility,
        m1_workload_resolver,
        pka_m0_pipeline,
        pka_m1_selector,
        pka_selector_core,
    )

    assert m1_measured_feature_extractor.ARTIFACT_DIR
    assert m1_ncu_capture_dispatcher.ARTIFACT_DIR
    assert m1_selector_eligibility.ARTIFACT_DIR
    assert m1_workload_resolver.ARTIFACT_DIR
    assert pka_m0_pipeline.FIXTURE_PATH
    assert pka_m1_selector.ELIGIBILITY_PATH
    assert pka_selector_core.FEATURE_ORDER
