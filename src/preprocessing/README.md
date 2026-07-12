# src/preprocessing — the CN (Qlib) panel pipeline

The project-faithful preprocessing that ports the CHRL feature pipeline onto the Qlib csi300/csi500
universe.

| Module | Role |
|---|---|
| `csi300_pipeline.py` | the full project-faithful preprocessing pipeline on the Qlib csi300 universe |
| `csi300_smoke.py` | smoke test: the preprocessed panel loads into the env end-to-end |

Data-integrity lesson baked in here (see the paper's data section): a CN panel once passed every
internal-consistency check while carrying a mixed clock (close/volume from yesterday, high/low from
today). Always ground-truth a panel against an independent source before trusting a signal on it.
