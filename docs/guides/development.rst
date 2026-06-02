Development
=============

Setup
-----

Requires **Python 3.12 or newer**.

Core install:

.. code-block:: bash

   pip install -e .

Developer tooling (ruff, mypy, pytest, coverage):

.. code-block:: bash

   pip install -e ".[dev]"

For TorchSim/FairChem-backed relaxation plus dev tools:

.. code-block:: bash

   pip install -e ".[mlip,dev]"

Linting
-------

Ruff enforces style (pycodestyle, pyflakes), import sorting, bugbear checks,
pyupgrade modernizations, simplifications, and a small de-slop set (commented-out
code, stale ``noqa`` comments, and similar).

.. code-block:: bash

   ruff check .
   ruff format --check .    # use ``ruff format .`` to apply formatting

Scripts and examples may use ``print``; tests may use ``assert`` — see
``[tool.ruff.lint.per-file-ignores]`` in ``pyproject.toml``.

Type checking
-------------

Mypy is configured in ``pyproject.toml`` and run on the library only:

.. code-block:: bash

   mypy src/metalsurfer

The package ships a ``py.typed`` marker. Tests, scripts, and examples are not
gated in CI; focus is on ``src/metalsurfer``.

Logging
-------

Call ``configure_logging()`` at the start of driver scripts. The library uses
structured context (``molecule``, ``surface_type``, ``placement_id``, ``seed``)
via ``log_context`` in screening and saturation workflows.

Environment overrides:

- ``METALSURFER_LOG_LEVEL`` (default: ``INFO``)
- ``TORCHSIM_LOG_LEVEL`` (default: ``WARNING``)

By default, INFO logs go to **stdout** so HPC schedulers capture progress in
``.out`` files. TorchSim stdout/stderr during relaxation is routed through
``torchsim_output_capture``. See also the **Logging** section in ``README.md``.

Tests
-----

Fast unit tests (matches the ``test-full`` CI job, excluding slow/MLIP markers):

.. code-block:: bash

   python -m pytest tests/ \
     -m "not dependency_behavior and not mlip and not gpu and not slow" \
     --cov=src/metalsurfer --cov-report=term-missing --tb=short -v
   coverage report --fail-under=74

Additional CI jobs locally:

.. code-block:: bash

   python -m pytest tests/test_dependency_behavior.py -v --tb=short
   python -m pytest tests/test_integration_seeded.py -v --tb=short

GPU / MLIP integration tests (optional, often run in separate processes):

.. code-block:: bash

   ./scripts/run_gpu_tests.sh

CI parity
---------

+---------------------------+------------------------------------------+
| Local command             | GitHub Actions job                       |
+===========================+==========================================+
| ``ruff check .``          | ``lint`` → Ruff lint and format          |
| ``ruff format --check .`` | ``lint`` → Ruff lint and format          |
| ``mypy src/metalsurfer``  | ``lint`` → Mypy typecheck                |
| Fast pytest + coverage    | ``test-full``                            |
| ``test_dependency_behavior`` | ``test-dependency-behavior``          |
| ``test_integration_seeded``  | ``test-integration``                  |
+---------------------------+------------------------------------------+

Read the Docs builds Sphinx docs but does not run mypy; type checking runs in CI.

Fixing failures
---------------

- **Ruff:** run ``ruff check . --fix`` for auto-fixable rules, then ``ruff format .``.
- **Mypy:** read the error code (e.g. ``[arg-type]``) and fix the annotation or add a
  targeted ignore only when third-party stubs are missing.
- **Coverage:** the gate is ``--fail-under=74`` on ``src/metalsurfer``; add tests for
  new branches rather than lowering the threshold.
