"""Optional dependency availability for conditional test skips.

Import this module to get module-level booleans (e.g. has_torch) so tests
can use pytest.mark.skipif(has_torch, reason="...") at collection time.
"""


def _check(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


has_torch = _check("torch")
has_torch_sim = _check("torch_sim")
has_fairchem = _check("fairchem")
has_fairchem_data_oc = _check("fairchem.data.oc")

# Full MLIP pipeline (calculator + torch-sim-atomistic for batching + fairchem-data-oc for slabs)
has_mlip_stack = has_torch and has_fairchem and has_torch_sim and has_fairchem_data_oc


# GPU availability: torch installed AND CUDA available (e.g. for CI: typically False)
def _cuda_available() -> bool:
    if not has_torch:
        return False
    try:
        import torch as _t

        return _t.cuda.is_available()
    except (ImportError, RuntimeError, AttributeError):
        return False


cuda_available = _cuda_available()
