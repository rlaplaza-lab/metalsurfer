Development
=============

Setup
-----

Requires **Python 3.12 or newer**.

Core install (library import and CPU-only workflow tests):

.. code-block:: bash

   pip install -e .

Running ``examples/``, ``scripts/``, or any ``run_*`` campaign requires the MLIP
stack:

.. code-block:: bash

   pip install -e ".[mlip]"

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

Per-file pydocstyle (``D``) ignores for scripts, examples, and tests are listed
under ``[tool.ruff.lint.per-file-ignores]`` in ``pyproject.toml``.

Type checking
-------------

Mypy is configured in ``pyproject.toml`` and run on the library only:

.. code-block:: bash

   python - <<'PY'
   import pathlib
   import shutil
   import site

   for root in site.getsitepackages():
       stubs = pathlib.Path(root) / "rdkit-stubs"
       if stubs.is_dir():
           shutil.rmtree(stubs)
   PY
   mypy src/metalsurfer

RDKit wheels bundle a broken ``rdkit-stubs/`` tree; CI removes it before mypy.

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
``torchsim_output_capture``.

Tests
-----

Three suites (``quick`` / ``cpu`` / ``gpu`` are applied automatically in
``tests/conftest.py``). Activate the ``metalsurfer`` conda env (or any env with
``pip install -e ".[mlip,dev]"``) from the repo root:

.. code-block:: bash

   python -m pytest tests/ -m quick --cov=src/metalsurfer --tb=short -v   # CI default
   python -m pytest tests/ -m cpu --tb=short -v                           # full CPU
   python -m pytest tests/ -m gpu --tb=short -v                           # CUDA + [mlip]
   ./scripts/run_gpu_tests.sh                                             # GPU, VRAM-safe
   ./scripts/run_all_tests.sh                                             # all three phases

CPU MLIP tests (``cpu and mlip``) need ``pip install -e ".[mlip]"`` and a
HuggingFace token for the gated UMA model. CI runs them in ``test-mlip-cpu``
when ``HF_TOKEN`` is set. GPU tests are local-only (no CUDA runners in Actions).

CI parity
---------

+----------------------------------+------------------------------------------+
| Local command                    | GitHub Actions job                       |
+==================================+==========================================+
| ``ruff check .``                 | ``lint``                                 |
| ``ruff format --check .``        | ``lint``                                 |
| ``mypy src/metalsurfer``         | ``lint``                                 |
| ``pytest -m quick`` + coverage   | ``test-quick``                           |
| ``pytest -m dependency_behavior``| ``test-dependency-behavior``             |
| ``pytest -m "cpu and mlip"``     | ``test-mlip-cpu`` (skipped if unset)     |
| ``pytest -m gpu``                | local only                               |
+----------------------------------+------------------------------------------+

Fixing failures
---------------

- **Ruff:** run ``ruff check . --fix`` for auto-fixable rules, then ``ruff format .``.
- **Mypy:** read the error code (e.g. ``[arg-type]``) and fix the annotation or add a
  targeted ignore only when third-party stubs are missing.
- **Coverage:** the gate is ``--fail-under=85`` on ``src/metalsurfer``; add tests for
   new branches rather than lowering the threshold.

Publishing
----------

Release builds are uploaded manually via the **Publish to PyPI** GitHub Actions
workflow (``workflow_dispatch``).

1. Bump ``version`` in ``pyproject.toml`` and ``src/metalsurfer/__init__.py``
   (``docs/conf.py`` reads the installed package version automatically).
2. Merge to ``main`` and wait for CI to pass.
3. On PyPI (and TestPyPI if used), configure a **trusted publisher** for this repo:
   owner ``rlaplaza-lab``, repository ``metalsurfer``, workflow
   ``publish-pypi.yml``, environment ``pypi`` or ``testpypi``.
4. In GitHub, create matching **environments** (``pypi``, ``testpypi``) if you want
   release approvals.
5. Actions → **Publish to PyPI** → Run workflow → choose target → type ``publish``.

The workflow builds with ``python -m build``, runs ``twine check``, and uploads via
OIDC trusted publishing (no long-lived API token required when configured).
