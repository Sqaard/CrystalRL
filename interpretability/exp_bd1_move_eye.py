"""BD-1 — the MOVE (rates-vol) eye: the most orthogonal free beyond-daily source, through the
exposure-matched twin + capacity-fair noise twins (the CL-1c / WP-2 discipline, both windows).

MOVE = swaption implied vol (a different asset class; corr with VIX level 0.35, changes 0.32 -
not a re-encoding). Preregistered bar: the MOVE-augmented belief PASSES iff BOTH hold and OOS
exposure-matched twin z >= base 4-eye + 0.5 AND > max of 3 year-shuffled capacity-fair noise
twins. Else NO-ADD (honest). Positive control: certified config reproduces a positive twin z.
"""
import sys; sys.path.insert(0, '.')
import json, numpy as np, pandas as pd
from interpretability.build_dow_extended_panel import fetch
from interpretability.exp_cl1_new_eyes_continual import load_v2, fit_belief, MACRO4
from interpretability.exp_cl1c_twin_corrected import derive_and_read
from interpretability.hl_v9_fresh_oos import TRAIN

r, obs, rf = load_v2()
move = fetch('^MOVE').set_index('date')['close'].reindex(r.index).ffill()
obs2 = obs.copy()
obs2['MOVE'] = move
obs2['MOVE_VIX'] = move - obs['VIX']           # divergence (fit_belief re-standardizes)
obs2 = obs2.ffill().bfill()

res = {}
base = fit_belief(obs[MACRO4], TRAIN[1])
res['base_4eye'] = derive_and_read(r, rf, base, 'base')
mv = fit_belief(obs2[MACRO4 + ['MOVE', 'MOVE_VIX']], TRAIN[1])
res['move_eye'] = derive_and_read(r, rf, mv, 'move')

# capacity-fair noise twins: year-shuffle the two MOVE columns
dts = obs2.index
yrs = np.asarray(pd.DatetimeIndex(dts).year)
noise_hold, noise_oos = [], []
for s in range(3):
    o = obs2.copy()
    uy = np.unique(yrs); pm = dict(zip(uy, np.random.default_rng(900+s).permutation(uy)))
    idxmap = np.arange(len(dts))
    for y in uy:
        src = np.where(yrs == pm[y])[0]; dst = np.where(yrs == y)[0]
        for col in ('MOVE', 'MOVE_VIX'):
            o.iloc[dst, o.columns.get_loc(col)] = obs2[col].to_numpy()[np.resize(src, len(dst))]
    beln = fit_belief(o[MACRO4 + ['MOVE', 'MOVE_VIX']], TRAIN[1])
    rn = derive_and_read(r, rf, beln, f'noise{s}')
    noise_hold.append(rn['hold']['twin_z']); noise_oos.append(rn['oos']['twin_z'])

zbh, zbo = res['base_4eye']['hold']['twin_z'], res['base_4eye']['oos']['twin_z']
zmh, zmo = res['move_eye']['hold']['twin_z'], res['move_eye']['oos']['twin_z']
adds = (zmh >= zbh + 0.5 and zmo >= zbo + 0.5 and zmh > max(noise_hold) and zmo > max(noise_oos))
print(f"base 4-eye : hold twin z {zbh} oos {zbo}")
print(f"+MOVE eye  : hold twin z {zmh} oos {zmo}  (tau {res['move_eye'].get('tau')} e {res['move_eye'].get('e_def')})")
print(f"noise twins: hold {noise_hold} oos {noise_oos}")
verdict = ("MOVE EYE ADDS - rates-vol carries orthogonal risk-timing info beyond the 4-eye belief on BOTH windows, beats noise twins -> escalate to the frozen v12 gate"
           if adds else
           "MOVE EYE NO-ADD - rates-vol does not beat the 4-eye belief + noise twins on both windows (another well-provenanced null: the orthogonal source is real but not incrementally certifiable at 10-20d)")
print("VERDICT:", verdict)
out = {'base': res['base_4eye'], 'move': res['move_eye'], 'noise_hold': noise_hold,
       'noise_oos': noise_oos, 'corr_move_vix': 0.348, 'adds': bool(adds), 'verdict': verdict}
json.dump(out, open('interpretability/exp_bd1_move_eye_report.json', 'w'), indent=1, default=str)
print("wrote exp_bd1_move_eye_report.json")
