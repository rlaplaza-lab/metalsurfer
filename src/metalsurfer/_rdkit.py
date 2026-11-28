"""Shared optional RDKit ``Chem`` / ``AllChem`` accessors."""

import logging
from typing import Any

from ._messages import RDKIT_INSTALL_HINT
from .exceptions import DependencyMissingError

logger = logging.getLogger(__name__)

Chem: Any = None
try:
    from rdkit import Chem as _Chem

    Chem = _Chem
except ImportError:
    pass

AllChem: Any = None
try:
    from rdkit.Chem import AllChem as _rdkit_allchem

    AllChem = _rdkit_allchem
except (ImportError, AttributeError):
    logger.debug("RDKit AllChem optional import unavailable", exc_info=True)


def require_rdkit_chem(feature: str):
    if Chem is None:
        raise DependencyMissingError("rdkit", feature, RDKIT_INSTALL_HINT)
    return Chem


def require_rdkit_allchem(feature: str):
    if AllChem is None:
        raise DependencyMissingError("rdkit", feature, RDKIT_INSTALL_HINT)
    return AllChem
