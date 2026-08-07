"""Human-readable campaign and run summary formatting."""

from pathlib import Path


def format_failure_summary_text(failure_summary: dict[str, object]) -> str:
    """Produce a human-readable multi-line summary from a failure_summary dict."""
    lines = ["Failure summary:"]
    stage = failure_summary.get("stage", "unknown")
    lines.append(f"  Stage: {stage}")

    if stage in {"reference", "conformers"}:
        reason = failure_summary.get("reason", "")
        if reason:
            lines.append(f"  Reason: {reason}")
    elif stage == "placement":
        n_attempted = failure_summary.get("n_placements_attempted", "?")
        n_initial = failure_summary.get("n_initial_placements", 0)
        lines.append(f"  Placements attempted: {n_attempted}")
        lines.append(f"  Initial placements: {n_initial}")
        if "n_candidate_specs" in failure_summary:
            lines.append(
                f"  Candidate specs: {failure_summary.get('n_candidate_specs', '?')}"
            )
        if "n_valid_pool" in failure_summary:
            lines.append(f"  Valid pool: {failure_summary.get('n_valid_pool', '?')}")
        generation_failures = failure_summary.get("generation_failures")
        if isinstance(generation_failures, dict) and generation_failures:
            lines.append("  Generation failures:")
            items = [
                (str(reason), int(count))
                for reason, count in generation_failures.items()
                if isinstance(count, int)
            ]
            for reason, count in sorted(items, key=lambda x: -x[1]):
                lines.append(f"    {reason}: {count}")
    elif stage == "validation":
        n_initial = failure_summary.get("n_initial_placements", "?")
        n_opt = failure_summary.get("n_optimized", "?")
        n_opt_fail = failure_summary.get("n_optimization_failed", 0)
        lines.append(f"  Initial placements: {n_initial}")
        lines.append(f"  Optimized: {n_opt} ({n_opt_fail} failed)")
        lines.append("  Passed validation: 0")
        if "n_evaluated" in failure_summary:
            lines.append(f"  BO evaluated: {failure_summary.get('n_evaluated', '?')}")
        if "n_valid_results" in failure_summary:
            lines.append(
                f"  BO valid results: {failure_summary.get('n_valid_results', '?')}"
            )
        validation_failures = failure_summary.get("validation_failures")
        if isinstance(validation_failures, dict):
            items = [
                (str(reason), int(count))
                for reason, count in validation_failures.items()
                if isinstance(count, int)
            ]
            if items:
                lines.append("  Validation failures:")
                for reason, count in sorted(items, key=lambda x: -x[1]):
                    lines.append(f"    {reason}: {count}")
    elif stage == "filter":
        n_before = failure_summary.get("n_before_filter", "?")
        n_after = failure_summary.get("n_after_filter", 0)
        lines.append(f"  Before filter: {n_before}")
        lines.append(f"  After filter: {n_after}")

    return "\n".join(lines)


def results_output_suffix(*, write_vasp_inputs: bool) -> str:
    return "(XYZ, POSCAR, CSV)" if write_vasp_inputs else "(XYZ, CSV)"


def format_results_saved_line(
    *,
    results_dir: str,
    write_vasp_inputs: bool = False,
) -> str:
    suffix = results_output_suffix(write_vasp_inputs=write_vasp_inputs)
    return f"Results saved to {Path(results_dir).as_posix()}/ {suffix}"


def format_saturation_completion(
    *,
    label: str,
    n_molecules_at_saturation: int,
    n_steps: int,
    results_dir: str,
    write_vasp_inputs: bool = False,
) -> str:
    suffix = results_output_suffix(write_vasp_inputs=write_vasp_inputs)
    return "\n".join(
        [
            f"{label} complete:",
            f"  Molecules at saturation: {n_molecules_at_saturation}",
            f"  Total steps: {n_steps}",
            f"  Results saved to {Path(results_dir).as_posix()}/ {suffix}",
        ]
    )
