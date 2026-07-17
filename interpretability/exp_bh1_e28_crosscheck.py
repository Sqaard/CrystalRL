"""BH1 stage-3 cross-check — does the strict-probe blindness change the E-28 REAL-panel fidelity?

Stage 3 proved the strict fixed-delta probe is blind on saturated-belief step policies. The E-28 /
paper-section-6 number (head CrystalScore F=0.033) was measured with a strict monotone probe on the
REAL panel. Two reasons it may still stand: (i) the real belief spends most days low (P(bear) well
under t1=0.30) where +0.2/+0.4 writes DO cross the certified thresholds; (ii) the champion rule
scored F=1.0 under the same probe — the probe sees crisp rules on the real distribution. But
honesty requires measuring, not arguing: re-probe every SAVED E-27 head (cold + warm) with the
boundary-aware contrast-write probe (0.2 vs 0.8, spanning both certified thresholds) next to the
strict probe, on the real hold-window belief stream.

If contrast-write stays ~0 on all heads -> the E-28 conclusion (dial, not command surface) and the
paper's F=0.033 SURVIVE the metric critique. If any head jumps -> the paper number needs a
correction note.

Run: python interpretability/exp_bh1_e28_crosscheck.py    (~1 min, loads saved heads)
"""
from __future__ import annotations
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from stable_baselines3 import PPO  # noqa: E402
from interpretability.crystal_ppo import get_streams, MODELS  # noqa: E402
from interpretability.exp_bh1_pressure import fidelity_probe  # noqa: E402
from interpretability.exp_bh1_pressure3 import contrast_write_probe  # noqa: E402

OUT = HERE / "exp_bh1_e28_crosscheck_report.json"
HEAD_NAMES = ["pure_return", "dd12", "dd08", "dd05", "pure_return_warm", "dd12_warm", "dd08_warm", "dd05_warm"]


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    print("=== E-28 cross-check — strict vs contrast-write fidelity on the saved real-panel heads ===")
    streams, _ = get_streams()
    bl_hold = streams["hold"][1]
    rows = {}
    for name in HEAD_NAMES:
        p = MODELS / f"{name}.zip"
        if not p.exists():
            print(f"  {name}: (not saved, skipped)"); continue
        m = PPO.load(p, device="cpu")
        s = fidelity_probe(m, bl_hold)
        cw = contrast_write_probe(m)
        rows[name] = {"strict": round(s, 3), "contrast_write": round(cw, 3)}
        print(f"  {name:18s}: strict {s:.2f} | contrast-write {cw:.2f}")
    any_jump = any(v["contrast_write"] >= 0.3 and v["contrast_write"] - v["strict"] >= 0.25
                   for v in rows.values())
    verdict = ("E-28 NUMBER NEEDS A CORRECTION NOTE: at least one real-panel head shows substantial "
               "contrast-write fidelity the strict probe missed — re-derive CrystalScore F with the "
               "boundary-aware probe and amend paper section 6." if any_jump else
               "E-28 CONCLUSION SURVIVES: contrast-write agrees with the strict probe on the real-panel "
               "heads — they are dials under BOTH measurements; the F=0.033 story and the paper stand. "
               "The strict probe's blindness is specific to saturated-belief step policies (the designed "
               "market), which the real belief distribution does not produce.")
    rep = {"experiment": "E-28 cross-check after the stage-3 metric-blindness finding",
           "heads": rows, "any_jump": bool(any_jump), "verdict": verdict}
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("VERDICT:", verdict); print("wrote", OUT.name)


if __name__ == "__main__":
    main()
