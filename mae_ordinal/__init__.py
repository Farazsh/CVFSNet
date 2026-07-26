"""VideoMAE / VideoMAE2 experiments for mTICI grading.

Supports CORN ordinal (``fuse01``) and binary T012a vs T2b3 classification
(AP / sagittal / dual). Binary monitoring matches ``dinov3`` (``val/auroc``,
threshold tuning, CSV/JSON/XLSX artifacts). See ``mae_ordinal/README.md``.
"""

from __future__ import annotations
