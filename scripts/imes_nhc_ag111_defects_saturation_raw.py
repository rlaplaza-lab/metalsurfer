#!/usr/bin/env python3
"""Generate high-placement, non-BO saturation data for iMes NHC on defected Ag(111).

Same workflow as ``examples/bipyridine_au111_defects_saturation_raw.py`` (for HPC batch jobs).

Purpose:
- Create an irregular Ag(111) surface by depositing extra Ag adatoms (defects).
- Run saturation screening with BO disabled so each step is sampled without BO bias.
- Write ``results_<surface>/`` per ``AdsorptionConfig`` (README, saturation).
"""

import logging

from metalsurfer import AdsorptionConfig, configure_logging, run_saturation
from metalsurfer.surface_prep import prepare_substrate

SURFACE_TYPE = "imes_nhc_ag111_defects_saturation_raw"
RESULTS_DIR = f"results_{SURFACE_TYPE}"
IMES_NHC_SMILES = "CC1=CC(C)=CC(C)=C1N2[C-]N(C3=C(C)CC(C)=CC3C)C=C2"

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
        min_pbc_image_separation=10.0,
        slab_relaxation_mode="full",
        slab_relaxation_optimizer="lbfgs",
        slab_relaxation_steps=250,
        autobatcher_max_memory_padding=0.8,
        autobatcher_max_memory_scaler=650,
        debug_write_initial_placements=True,
        save_benchmark_dataset=True,
    )

    slab = prepare_substrate(
        bulk_id="mp-124",
        miller_indices=(1, 1, 1),
        supercell=(3, 3, 1),
        adatom_symbol="Ag",
        adatom_coverage=0.20,
        config=config,
        results_dir=RESULTS_DIR,
        adatom_relaxation_mode="ionic_only",
    )
    logger.info("Defected Ag(111) slab atoms: %d", len(slab.atoms))

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
            label="iMes NHC saturation on defected Ag(111)",
            results_dir=RESULTS_DIR,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
