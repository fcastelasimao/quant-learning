from __future__ import annotations

import sys
from pathlib import Path


WAVE_RIDER_ROOT = Path(__file__).resolve().parents[1]
if str(WAVE_RIDER_ROOT) not in sys.path:
    sys.path.insert(0, str(WAVE_RIDER_ROOT))
