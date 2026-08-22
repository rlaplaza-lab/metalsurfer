"""Human-readable campaign and run summary formatting."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, assert_never


@dataclass(frozen=True)
class ReferenceFailure:
    """Failure while resolving reference energies."""

    reason: str
    stage: Literal["reference"] = "reference"


@dataclass(frozen=True)
class ConformerFailure:
    """Failure while generating or loading conformers."""

    reason: str
    stage: Literal["conformers"] = "conformers"


@dataclass(frozen=True)
class PlacementFailure:
    """Core placement-generation bailout (no valid initial placements)."""

    n_placements_attempted: int
    n_initial_placements: int
    generation_failures: dict[str, int] = field(default_factory=dict)
    n_retry_attempts: int | None = None
    stage: Literal["placement"] = "placement"


@dataclass(frozen=True)
class BOPlacementFailure:
    """BO placement/pool bailout before evaluation."""

    n_candidate_specs: int
    n_valid_pool: int
    stage: Literal["placement"] = "placement"


@dataclass(frozen=True)
class OptimizationFailure:
    """All placements failed during batched optimisation."""

    n_placements_attempted: int
    n_initial_placements: int
    n_optimized: int = 0
    n_optimization_failed: int = 0
    validation_failures: dict[str, int] = field(default_factory=dict)
    stage: Literal["optimization"] = "optimization"


@dataclass(frozen=True)
class ValidationFailure:
    """Core path: optimised structures all failed validation."""

    n_initial_placements: int
    n_optimized: int
    n_optimization_failed: int
    validation_failures: dict[str, int] = field(default_factory=dict)
    stage: Literal["validation"] = "validation"


@dataclass(frozen=True)
class BOValidationFailure:
    """BO path: evaluations produced no valid adsorption results."""

    n_evaluated: int
    n_valid_results: int
    n_candidate_specs: int = 0
    n_valid_pool: int = 0
    stage: Literal["validation"] = "validation"


@dataclass(frozen=True)
class FilterFailure:
    """All results removed by post-optimisation filters."""

    n_before_filter: int
    n_after_filter: int = 0
    stage: Literal["filter"] = "filter"


FailureSummary = (
    ReferenceFailure
    | ConformerFailure
    | PlacementFailure
    | BOPlacementFailure
    | OptimizationFailure
    | ValidationFailure
    | BOValidationFailure
    | FilterFailure
)


def format_failure_summary_text(failure_summary: FailureSummary) -> str:
    """Produce a human-readable multi-line summary from a failure summary.

    Parameters
    ----------
    failure_summary
        Stage-typed failure summary dataclass.
    """
    lines = ["Failure summary:"]
    stage = failure_summary.stage
    lines.append(f"  Stage: {stage}")

    if isinstance(failure_summary, (ReferenceFailure, ConformerFailure)):
        lines.append(f"  Reason: {failure_summary.reason}")
    elif isinstance(failure_summary, PlacementFailure):
        lines.append(
            f"  Placements attempted: {failure_summary.n_placements_attempted}"
        )
        lines.append(f"  Initial placements: {failure_summary.n_initial_placements}")
        if failure_summary.n_retry_attempts is not None:
            lines.append(f"  Retry attempts: {failure_summary.n_retry_attempts}")
        if failure_summary.generation_failures:
            lines.append("  Generation failures:")
            for reason, count in sorted(
                failure_summary.generation_failures.items(), key=lambda x: -x[1]
            ):
                lines.append(f"    {reason}: {count}")
    elif isinstance(failure_summary, BOPlacementFailure):
        lines.append(f"  Candidate specs: {failure_summary.n_candidate_specs}")
        lines.append(f"  Valid pool: {failure_summary.n_valid_pool}")
    elif isinstance(failure_summary, OptimizationFailure):
        lines.append(
            f"  Placements attempted: {failure_summary.n_placements_attempted}"
        )
        lines.append(f"  Initial placements: {failure_summary.n_initial_placements}")
        if failure_summary.n_optimized or failure_summary.n_optimization_failed:
            lines.append(
                f"  Optimized: {failure_summary.n_optimized} "
                f"({failure_summary.n_optimization_failed} failed)"
            )
        _append_validation_failure_lines(lines, failure_summary.validation_failures)
    elif isinstance(failure_summary, ValidationFailure):
        lines.append(f"  Initial placements: {failure_summary.n_initial_placements}")
        lines.append(
            f"  Optimized: {failure_summary.n_optimized} "
            f"({failure_summary.n_optimization_failed} failed)"
        )
        lines.append("  Passed validation: 0")
        _append_validation_failure_lines(lines, failure_summary.validation_failures)
    elif isinstance(failure_summary, BOValidationFailure):
        lines.append(f"  BO evaluated: {failure_summary.n_evaluated}")
        lines.append(f"  BO valid results: {failure_summary.n_valid_results}")
    elif isinstance(failure_summary, FilterFailure):
        lines.append(f"  Before filter: {failure_summary.n_before_filter}")
        lines.append(f"  After filter: {failure_summary.n_after_filter}")
    else:
        assert_never(failure_summary)

    return "\n".join(lines)


def _append_validation_failure_lines(
    lines: list[str], validation_failures: dict[str, int]
) -> None:
    if not validation_failures:
        return
    lines.append("  Validation failures:")
    for reason, count in sorted(validation_failures.items(), key=lambda x: -x[1]):
        lines.append(f"    {reason}: {count}")


def results_output_suffix(*, write_vasp_inputs: bool) -> str:
    """Return a human-readable suffix listing output formats.

    Parameters
    ----------
    write_vasp_inputs
        Whether POSCAR outputs are included.
    """
    return "(XYZ, POSCAR, CSV)" if write_vasp_inputs else "(XYZ, CSV)"


def format_results_saved_line(
    *,
    results_dir: str,
    write_vasp_inputs: bool = False,
) -> str:
    """Format a "Results saved to ..." message.

    Parameters
    ----------
    results_dir
        Path to the results directory.
    write_vasp_inputs
        Whether POSCAR outputs are included.
    """
    suffix = results_output_suffix(write_vasp_inputs=write_vasp_inputs)
    return f"Results saved to {Path(results_dir).as_posix()}/{suffix}"


def format_saturation_completion(
    *,
    label: str,
    n_molecules_at_saturation: int,
    n_steps: int,
    results_dir: str,
    write_vasp_inputs: bool = False,
) -> str:
    """Format a saturation-run completion message.

    Parameters
    ----------
    label
        Run label.
    n_molecules_at_saturation
        Number of molecules at saturation.
    n_steps
        Total number of saturation steps.
    results_dir
        Path to the results directory.
    write_vasp_inputs
        Whether POSCAR outputs are included.
    """
    suffix = results_output_suffix(write_vasp_inputs=write_vasp_inputs)
    return "\n".join(
        [
            f"{label} complete:",
            f"  Molecules at saturation: {n_molecules_at_saturation}",
            f"  Total steps: {n_steps}",
            f"  Results saved to {Path(results_dir).as_posix()}/{suffix}",
        ]
    )
