# src/series_g — the designed market

The synthetic environment of paper §8 — *"can a market be built that rewards only a clean mind?"*.
Used strictly as **instrument calibration** (behind the VoI gate), never as market evidence.

| Module | Role |
|---|---|
| `regime_pomdp.py` | the tabular regime-switching POMDP — the belief-driven substrate (Phase 0) |
| `family_env.py` | the G-regime rotation env (the polygon as a parametric family of markets) |
| `generators.py` | cross-model-family toxicity generators for the HC-3 test |
| `multiasset_env.py` | the high-dimensional multi-asset regime-POMDP (Extension 1) |
| `phase0_gate.py`, `phase0_validate.py` | the solver + the gate, and its teeth-validation suite |
| `phase1_falsifiers.py` | the four pre-CrystalScore falsifiers |
| `phase2_hierarchy.py`, `phase3_anchor.py` | H3 hierarchical factoring; the HX frontier anchor |
| `ext1..ext3_*.py` | the extension experiments (high-dim train + H3, cross-family, the complexity zoo) |

**Entry point:** `regime_pomdp.py` (what the market IS) → `family_env.py` (how it's parameterized) →
`phase0_gate.py` (the solver that proves belief carries value here).
