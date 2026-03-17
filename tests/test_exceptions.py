"""Tests for domain exception classes and their messages."""

from metalsurfer.exceptions import (
    DependencyMissingError,
    GeometryValidationError,
    OptimizationError,
)


def test_dependency_missing_message():
    exc = DependencyMissingError("torch", "setup_calculator", "pip install torch")
    assert "torch" in str(exc)
    assert "setup_calculator" in str(exc)
    assert "pip install torch" in str(exc)
    assert exc.dependency == "torch"
    assert exc.feature == "setup_calculator"


def test_dependency_missing_without_hint():
    exc = DependencyMissingError("fairchem", "some_function")
    assert "fairchem" in str(exc)
    assert isinstance(exc, RuntimeError)


def test_geometry_validation_error():
    exc = GeometryValidationError("atoms too close: 0.3 A")
    assert "0.3 A" in str(exc)
    assert exc.reason == "atoms too close: 0.3 A"
    assert isinstance(exc, ValueError)


def test_optimization_error():
    exc = OptimizationError("did not converge after 200 steps")
    assert "200 steps" in str(exc)
    assert exc.reason == "did not converge after 200 steps"
    assert isinstance(exc, RuntimeError)
