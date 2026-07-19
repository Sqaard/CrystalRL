import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from interpretability.exp_cl1_new_eyes_continual import load_v2
from interpretability.exp_wp2_queue import candidate_dailies
from interpretability.exp_wb1_augmented_battery import build_aug_level
from interpretability.exp_w1_ktb_v2 import assemble
from interpretability.hl_v9_fresh_oos import TRAIN, DEV, HOLD
from src.hl.r6c_tension_adapter import block_z
from sklearn.linear_model import Ridge

def evalcell(bl, ctx, k, forced_null=None):
    asm = assemble(bl, ctx, (1,2,4,8))
    S = bl['S']
    tr = np.where(asm['masks']['train'])[0]
    purged = tr[asm['tgt_dates'][k][tr] <= pd.Timestamp(TRAIN[1])]
    di = np.where(asm['masks']['dev'])[0]
    best=None
    for a in (0.1,1,10,100,1000,10000):
        reg=Ridge(alpha=a).fit(S[asm['ctx_of'][purged]].reshape(len(purged),-1), S[asm['tgt_of'][k][purged]])
        e=((reg.predict(S[asm['ctx_of'][di]].reshape(len(di),-1))-S[asm['tgt_of'][k][di]])**2).sum(1)
        if best is None or e.mean()<best[0]: best=(e.mean(),a,reg)
    _,alpha,reg=best
    mean_v=S[np.unique(asm['tgt_of'][k][purged])].mean(0)
    cur=lambda ii:S[asm['cur_of'][ii]]
    mean_f=lambda ii:np.repeat(mean_v[None,:],len(ii),0)
    if forced_null=='mean': null_f=mean_f
    elif forced_null=='pers': null_f=cur
    else:
        cands={'pers':cur,'mean':mean_f}
        for lam in (0.25,0.5,0.75): cands[f'sh{lam}']=(lambda l:lambda ii:l*S[asm['cur_of'][ii]]+(1-l)*mean_v[None,:])(lam)
        nb=None
        for nm,f in cands.items():
            e=((f(di)-S[asm['tgt_of'][k][di]])**2).sum(1)
            if nb is None or e.mean()<nb[0]: nb=(e.mean(),nm,f)
        null_f=nb[2]
    ii=np.where(asm['masks']['hold'])[0]
    ep=((reg.predict(S[asm['ctx_of'][ii]].reshape(len(ii),-1))-S[asm['tgt_of'][k][ii]])**2).sum(1)
    en=((null_f(ii)-S[asm['tgt_of'][k][ii]])**2).sum(1)
    d=en-ep; _,se=block_z(d,block=ctx,n_boot=1000,seed=7)
    return round(float(d.mean()/(en.mean()+1e-12)),4), round(float(d.mean()/se),2)

r,obs,rf=load_v2()
# PLACEBO MARKET: iid shuffle
rng=np.random.default_rng(777); perm=rng.permutation(len(r))
r_p=pd.Series(r.to_numpy()[perm],index=r.index)
obs_p=pd.DataFrame(obs.to_numpy()[perm],index=obs.index,columns=obs.columns)
sb_p=candidate_dailies(r_p,obs_p,r_p.index)['stockbond']
bl_p=build_aug_level(r_p,obs_p,sb_p,5,12)
print('=== PLACEBO MARKET (iid-shuffled) L1 k=1, ctx=2 ===')
print('  dev-chosen null :', evalcell(bl_p,2,1))
print('  FORCED mean null:', evalcell(bl_p,2,1,'mean'))
print('  FORCED pers null:', evalcell(bl_p,2,1,'pers'))
# REAL market, same forced-mean probe on the SIG cells
sb=candidate_dailies(r,obs,r.index)['stockbond']
bl=build_aug_level(r,obs,sb,5,12)
print('=== REAL augmented market, forced-mean null ===')
for k,ctx in ((1,2),(4,4)):
    print(f'  L1 k={k}: dev-null {evalcell(bl,ctx,k)} | forced-mean {evalcell(bl,ctx,k,"mean")}')
