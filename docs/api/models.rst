Result Models
=============

These dataclasses represent the typed outputs returned by the run-mode
functions and workflow helpers. Only the public API types exported from
the top-level ``metalsurfer`` module are documented here.

Campaign Results
----------------

.. autoclass:: metalsurfer.BindingCampaignResult
   :members:
   :undoc-members:

.. autoclass:: metalsurfer.MoleculeCampaignSummary
   :members:
   :undoc-members:

Screening Results
-----------------

.. autoclass:: metalsurfer.ScreeningResult
   :members:
   :undoc-members:

.. autoclass:: metalsurfer.ScreeningRunResult
   :members:
   :undoc-members:

.. autoclass:: metalsurfer.MoleculeSummary
   :members:
   :undoc-members:

Saturation Results
------------------

.. autoclass:: metalsurfer.SaturationStepResult
   :members:
   :undoc-members:

.. autoclass:: metalsurfer.SaturationCampaignResult
   :members:
   :undoc-members:

.. autoclass:: metalsurfer.SaturationRunResult
   :members:
   :undoc-members:

.. autoclass:: metalsurfer.MultiMolSaturationStepResult
   :members:
   :undoc-members:

.. autoclass:: metalsurfer.MultiMolSaturationRunResult
   :members:
   :undoc-members:

Reference Energies
------------------

.. autoclass:: metalsurfer.ReferenceEnergies
   :members:
   :undoc-members:

Timing
------

.. autoclass:: metalsurfer.TimingInfo
   :members:
   :undoc-members:

Placement specifications
------------------------

.. autoclass:: metalsurfer.models.PlacementSpec
   :members:
   :undoc-members:

.. autoclass:: metalsurfer.models.PlacementDescriptor
   :members:
   :undoc-members:

Bayesian transfer bookkeeping
-----------------------------

Embedded on :class:`~metalsurfer.SaturationStepResult.transfer` and
:class:`~metalsurfer.MultiMolSaturationStepResult.transfer_by_molecule`.
CSV exports still flatten to stable ``bo_transfer_*`` column names.

.. autoclass:: metalsurfer.models.BOTransferInfo
   :members:
   :undoc-members:

Workflow screen outcome
-----------------------

Returned by mid-level ``process_molecule`` /
``process_molecule_bayesian`` (campaign APIs consume these internally).

.. autoclass:: metalsurfer.workflow.MoleculeScreenOutcome
   :members:
   :undoc-members:

Exceptions
----------

.. autoexception:: metalsurfer.GeometryValidationError

.. autoexception:: metalsurfer.DependencyMissingError

.. autoexception:: metalsurfer.OptimizationError

ML helpers
----------

``PlacementRecord`` (in :mod:`metalsurfer.ml.schema`) stores geometry as a
nested :class:`~metalsurfer.models.PlacementDescriptor` plus energies,
labels, and :class:`~metalsurfer.ml.schema.ComputationContext`. CSV
``to_flat_dict`` / ``from_flat_dict`` keep a flat column layout for
compatibility.

.. autoclass:: metalsurfer.ml.DatasetLogger
   :members:
   :undoc-members:

