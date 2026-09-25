#!/usr/bin/env python3
"""Generate high-placement, non-BO saturation data for iMes NHC on pristine Ag(111).

Same workflow as ``scripts/imes_nhc_ag111_defects_saturation_raw.py``, but without
Ag adatom defects (clean Ag(111) from mp-124).

Purpose:
- Build a pristine Ag(111) slab (no adatoms).
- Run saturation screening with BO disabled so each step is sampled without BO bias.
- Tag the carbene carbon ``[C-:1]`` so EN-down sampling prioritizes C–surface bonds.
- Write ``results_<surface>/`` per ``AdsorptionConfig`` (README, saturation).
"""

import logging

from metalsurfer import AdsorptionConfig, configure_logging, run_saturation
from metalsurfer.conformers import create_conformers_from_smiles
from metalsurfer.surface_prep import prepare_substrate, resize_substrate_for_molecule

SURFACE_TYPE = "imes_nhc_ag111_pristine_saturation_raw"
RESULTS_DIR = f"results_{SURFACE_TYPE}"
# ``[C-:1]`` tags the carbene carbon so EN-down sampling prioritizes C–surface bonds.
IMES_NHC_SMILES = "CC1=CC(C)=CC(C)=C1N2[C-:1]N(C3=C(C)CC(C)=CC3C)C=C2"

configure_logging(default_level="INFO")
logger = logging.getLogger(__name__)


def main():
    config = AdsorptionConfig(
        material_type="slab",
        model_name="uma-s-1p2",
        task_name="oc25",
        seed=42,
        num_conformers=20,
        num_placements=1000,
        device="cuda",
        fmax=0.05,
        stage1_steps=80,
        stage2_steps=500,
        # Default 8 A keeps the 3x3 Ag(111) cell valid for iMes (~14.5 A);
        # 10 A forced a (2,2,1) resize (~1100 Ag) that exceeded autobatcher max_metric.
        min_pbc_image_separation=8.0,
        # UMA oc25 does not expose stress; full cell+ionic prep fails.
        slab_relaxation_mode="ionic_only",
        slab_relaxation_optimizer="lbfgs",
        slab_relaxation_steps=250,
        autobatcher_max_memory_padding=0.8,
        # Must exceed slab+n_adsorbates (step 3 hit 422 atoms with two prior iMes).
        autobatcher_max_memory_scaler=800,
        debug_write_initial_placements=True,
        save_benchmark_dataset=True,
    )

    slab = prepare_substrate(
        bulk_id="mp-124",
        miller_indices=(1, 1, 1),
        supercell=(3, 3, 1),
        config=config,
        results_dir=RESULTS_DIR,
    )
    logger.info("Pristine Ag(111) slab atoms: %d", len(slab.atoms))

    conformer_pack = create_conformers_from_smiles(IMES_NHC_SMILES, config=config)
    if conformer_pack is None:
        logger.error("Conformer generation failed for iMes NHC.")
        return 1
    conformers, _ = conformer_pack
    slab = resize_substrate_for_molecule(slab, conformers, config)
    logger.info("Resized pristine Ag(111) slab atoms: %d", len(slab.atoms))

    campaign = run_saturation(
        slab=slab,
        molecules=[(IMES_NHC_SMILES, "imes_nhc")],
        config=config,
        surface_type=SURFACE_TYPE,
        skip_existing=False,
    )

    if not campaign.runs:
        logger.error("No saturation results produced.")
        if campaign.failure_summary:
            logger.error(campaign.format_failure_summary())
        return 1

    print(
        campaign.format_completion(
            label="iMes NHC saturation on pristine Ag(111)",
            results_dir=RESULTS_DIR,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
