# Pt₄ water / OH tutorial

One script — [`water_oh_pt4_saturation.py`](water_oh_pt4_saturation.py) — lets
water and hydroxide compete for adsorption sites on a four-atom platinum cluster
while you change pH through simple activity numbers. Comments in the script
explain each setting.

## Install

From the repository root (Python 3.12 or newer):

```bash
pip install -e ".[mlip]"
```

That installs metalsurfer plus the machine-learning model used for relaxations.
The tutorial runs on CPU; a GPU is faster but not required.

## Run

From the project root:

```bash
python scripts/tutorials/water_oh_pt4_saturation.py
```

Results appear in folders named ``results_water_oh_pt4/`` (cluster prep) and
``results_water_oh_pt4_ph7/``, ``_ph10/``, ``_ph14/`` (one run per pH).

## What you are exploring

Metalsurfer does not have a “pH” button. You set pH by giving each adsorbate an
**activity** (how abundant it is in the reservoir):

- water always has activity 1  
- hydroxide uses ``10**(pH - 14)`` (more OH⁻ at higher pH)

At each step water and OH⁻ are tried on the cluster. The one with the better
**ranking energy** (called Ω in the output) wins and stays on the surface.
Saturation stops when nothing binds favorably anymore. The CSV files still list
ordinary adsorption energies (``E_ads``); Ω is what decides the winner.

## Try other Pt₄ shapes

**Built-in tetrahedron** — leave ``CLUSTER_XYZ = None`` at the top of the script.

**Your own cluster** — save any four-atom Pt structure as a ``.xyz`` file in
this folder (see ``pt4_tetrahedron.xyz`` for an example). Then either:

- set ``CLUSTER_XYZ = "my_cluster.xyz"`` in the script, or  
- run ``python scripts/tutorials/water_oh_pt4_saturation.py --cluster my_cluster.xyz``

The file must contain exactly four platinum atoms. A larger vacuum box (about
20 Å or more) avoids the adsorbate seeing itself across periodic boundaries.

Worth comparing: the default symmetric tetrahedron vs a slightly distorted
geometry (e.g. move one atom by hand in the ``.xyz``) to see how much the
cluster shape changes which species wins.

## Settings worth changing

Open the script and edit the block marked “Tutorial knobs” and the commented
``make_config`` section, then re-run.

| Setting | Why change it |
|---------|----------------|
| ``PH_VALUES`` | Which pH values to compare (default 7, 10, 14). |
| ``CLUSTER_XYZ`` / ``--cluster`` | Different Pt₄ geometry from a ``.xyz`` file. |
| ``DEVICE`` | ``"cpu"`` or ``"cuda"``. |
| ``activities_for_ph`` | How pH maps to water vs OH activities — the heart of the demo. |
| ``saturation_temperature`` | Temperature in the ranking formula (default 298 K). |
| ``saturation_max_steps`` | How many adsorbates can stack on the cluster. |
| ``num_placements`` | More tries per step = slower but more structures to pick from. |
| ``slab_relaxation_mode`` | ``"ionic_only"`` relaxes the cluster before adsorption; ``"none"`` keeps your input geometry. |
| ``saturation_save_all_placements`` | ``True`` saves **every** valid try, not just the winner (see DFT section). |

The script also clears freeze constraints after prep so the small cluster can
move while adsorbates relax — helpful for binding on Pt₄.

## Structures for DFT

Treat machine-learning relaxed geometries as **suggestions** to check with your
own quantum-chemistry code (VASP, Quantum ESPRESSO, etc.), not as final answers.

### Best (winning) structures

In each pH results folder (e.g. ``results_water_oh_pt4_ph14/``):

- ``saturation_details.csv`` — which species won each step and the energies.  
- ``xyz_structures/hydroxide_saturation/step_001_Eads_….xyz`` (and similar) —
  the relaxed cluster **plus adsorbate** for each committed step.

Copy a file into your DFT workflow, or in Python:

```python
from ase.io import read, write

atoms = read("results_water_oh_pt4_ph14/xyz_structures/hydroxide_saturation/step_001_Eads_-1.7933.xyz")
write("for_dft/POSCAR", atoms, format="vasp")
```

To have metalsurfer write VASP input folders directly, set
``write_vasp_inputs=True`` in ``make_config`` and run again.

### Other placements (not just the winner)

By default the tutorial only keeps the best pose per step
(``saturation_save_all_placements=False``). To save **all valid tries** —
including worse binding energies — set:

```python
saturation_save_all_placements=True,
```

inside ``make_config``, then re-run. You will get:

- ``saturation_placements_detailed.csv`` — every valid placement with ``E_ads``,
  molecule name, and step number. Sort by energy to see winners and runners-up.  
- ``xyz_structures/…/step_001_placements/`` — one ``.xyz`` per tried placement
  for that step (full system and adsorbate-only variants when written).

Use these for DFT when you want to compare the winning pose against nearby
alternatives, or when the best ML structure looks suspicious and you prefer
to verify a slightly higher-energy placement.

### Practical notes for DFT

- Increase vacuum if the box feels tight (20 Å is a teaching default).  
- OH⁻ is charged in the SMILES; your DFT setup must treat charge and
  neutralization consistently — metalsurfer does not add solvation or an electrode.  
- Re-relax in DFT and compare **geometries** (O–Pt distance, OH tilt) and
  **trends** across pH, not raw eV numbers copied from the machine-learning model.
