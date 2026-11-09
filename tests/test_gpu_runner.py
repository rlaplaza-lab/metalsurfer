"""GPU runner collects node ids instead of a hardcoded list."""

from pathlib import Path

from tests.gpu_nodes import node_ids_from_collect_output

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "run_gpu_tests.sh"


def test_node_ids_from_collect_output_keeps_nodeids_only():
    text = """
tests/test_bayesian.py::test_bayesian_two_generations_on_defect_surface
tests/test_integration_mlip_pipeline.py::test_mlip_pipeline[ethene_ru]
tests/test_integration_water_cu_slab.py::test_run_adsorption_water_on_cu111

3/9 tests collected (6 deselected) in 0.40s
"""
    assert node_ids_from_collect_output(text) == [
        "tests/test_bayesian.py::test_bayesian_two_generations_on_defect_surface",
        "tests/test_integration_mlip_pipeline.py::test_mlip_pipeline[ethene_ru]",
        "tests/test_integration_water_cu_slab.py::test_run_adsorption_water_on_cu111",
    ]


def test_node_ids_from_collect_output_ignores_blank_and_summary():
    assert node_ids_from_collect_output("\nno tests collected\n") == []


def test_gpu_runner_collects_marker_and_does_not_hardcode_nodes():
    script = _SCRIPT.read_text()
    assert "--collect-only" in script
    assert 'MARKER="gpu_smoke"' in script
    assert 'MARKER="gpu"' in script
    assert "node_ids_from_collect_output" in script
    assert "METALSURFER_MP_SPAWN=1" in script
    assert "METALSURFER_CLEAR_GPU_PYTHON:-0" in script
    assert "test_mlip_pipeline[ethene_ru]" not in script
    assert "test_run_saturation_screening_h2_ni111_real_gpu" not in script
    assert "FAILED_NODES" in script
    assert "set +e" in script
