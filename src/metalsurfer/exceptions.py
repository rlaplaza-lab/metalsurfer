"""Domain-specific exceptions for metalsurfer."""


class DependencyMissingError(RuntimeError):
    """A required optional dependency is not installed or cannot be imported."""

    def __init__(self, dependency: str, feature: str, hint: str = "") -> None:
        """Instantiate with dependency name, feature, and optional hint.

        Parameters
        ----------
        dependency
            Name of the missing package.
        feature
            Feature that requires the dependency.
        hint
            Optional installation hint.
        """
        self.dependency = dependency
        self.feature = feature
        msg = f"{feature} requires '{dependency}'"
        if hint:
            msg += f". {hint}"
        super().__init__(msg)


class GeometryValidationError(ValueError):
    """An atomic structure failed geometric sanity checks."""

    def __init__(self, reason: str) -> None:
        """Instantiate with failure reason.

        Parameters
        ----------
        reason
            Human-readable description of the validation failure.
        """
        self.reason = reason
        super().__init__(f"Geometry validation failed: {reason}")


class OptimizationError(RuntimeError):
    """A geometry optimisation did not converge or encountered a fatal error."""

    def __init__(self, reason: str) -> None:
        """Instantiate with failure reason.

        Parameters
        ----------
        reason
            Human-readable description of the optimization failure.
        """
        self.reason = reason
        super().__init__(f"Optimization failed: {reason}")
