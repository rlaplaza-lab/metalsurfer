"""Shared optional RDKit ``Chem`` accessor."""

from typing import Any

from ._messages import RDKIT_INSTALL_HINT
from .exceptions import DependencyMissingError

Chem: Any = None
try:
    from rdkit import Chem as _Chem

    Chem = _Chem
except ImportError:
    pass


def require_rdkit_chem(feature: str):
    if Chem is None:
        raise DependencyMissingError("rdkit", feature, RDKIT_INSTALL_HINT)
    return Chem
