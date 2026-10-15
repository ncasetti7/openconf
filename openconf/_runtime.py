"""Track runtime compatibility state."""

import sys

_NVMOLKIT_EMBED_IMPORTED_BEFORE_RDKIT: bool = (
    "nvmolkit._embedMolecules" in sys.modules and "rdkit.Chem.rdchem" not in sys.modules
)
