"""IndoFraud-Synth v1.0: calibrated synthetic Indonesian multi-rail payment fraud generator.

Accompanies the research extension of Ghourabi & Khaldi (2026), Appl. Sci. 16, 4508
(federated ensemble fraud detection under realistic cross-silo heterogeneity).

Methodological precedent: PaySim (Lopez-Rojas et al., 2016) — synthetic transaction
data calibrated on public aggregate statistics. All calibration constants and their
sources are documented in `calibration.py`. Data are SYNTHETIC; never describe them
as real transactions.
"""

__version__ = "1.0.0"

from .config import SynthConfig
from .generator import generate, save_dataset
