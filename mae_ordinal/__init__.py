"""VideoMAE + ordinal-regression experiment for coronal-only mTICI grading.

Self-contained package that *imports* the existing CVFSNet/AmTICIS code (data
pipeline + metrics) without modifying it. See
``mae_ordinal/docs/ORDINAL_EXPERIMENT_PLAN.md``.
"""

from __future__ import annotations
