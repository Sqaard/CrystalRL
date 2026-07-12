# Code-level unicity: Rashomon set of LATENT-codebook descriptions (code -> cash) for R6c.
# The "code" description = KMeans(K) over (a subset of) the 64-d penultimate latent, predicting cash.
# Contrast with the stance-level (autoregressive) Rashomon. OOS CV; unicity-vs-n curve.
import numpy as np, pandas as pd, json, time
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
rng=np.random.default_rng(11); t0=time.time()
NPZ="artifacts/stage4/R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_frozen_2022_2023_for_Joseph/hidden_activations/r6c_frozen_hidden_activations.npz"
LOG="artifacts/stage4/R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_frozen_2022_2023_for_Joseph/frozen_test_behavior_log_daily.csv"
d=np.load(NPZ,allow_pickle=True)
lat=d['policy_net.5.ReLU'].astype(float)              # (289,64) the readable code layer
cash=pd.read_csv(LOG)['cash_target'].to_numpy(float)[:len(lat)]
print("latent",lat.shape,"cash",cash.shape,flush=True)

def oos_loss(S,y,feats,K,seed,folds=3):
    Xf=S[:,feats]; kf=KFold(folds,shuffle=True,random_state=seed); e=0;nte=0
    for tr,te in kf.split(Xf):
        mu=Xf[tr].mean(0);sd=Xf[tr].std(0)+1e-9
        km=KMeans(K,n_init=1,random_state=seed).fit((Xf[tr]-mu)/sd)
        cm=np.array([y[tr][km.labels_==c].mean() if (km.labels_==c).any() else y[tr].mean() for c in range(K)])
        e+=np.sum((y[te]-cm[km.predict((Xf[te]-mu)/sd)])**2); nte+=len(te)
    return e/nte

def rashomon(S,y,N=300):
    base=float(np.var(y)); nd=S.shape[1]; losses=[]
    for i in range(N):
        K=int(rng.integers(2,10)); k=int(rng.integers(2,17)); feats=sorted(rng.choice(nd,k,replace=False).tolist())
        losses.append(oos_loss(S,y,feats,K,i))
    losses=np.array(losses); Lstar=float(min(losses.min(),base)); struct=max(0.,1-Lstar/base)
    curve={str(e):round((losses<=Lstar*(1+e)).mean(),3) for e in [0.02,0.05,0.1,0.25,0.5]}
    return {'n':int(len(y)),'structure_frac':round(struct,3),'ratio_curve':curve}

res=rashomon(lat,cash,N=300)
print("CODE-LEVEL (latent->cash):",res,flush=True)
# unicity vs evidence length n (code level)
uni={}
for n in [60,120,180,240,len(cash)]:
    r=rashomon(lat[:n],cash[:n],N=150); uni[str(n)]={'ratio_e10':r['ratio_curve']['0.1'],'structure_frac':r['structure_frac']}
    print("code-unicity n",n,uni[str(n)],flush=True)
out={'code_level':res,'code_unicity_vs_n':uni}
json.dump(out,open("interpretability/k_codelevel_report.json","w"),indent=2)
print("DONE %.1fs"%(time.time()-t0),flush=True)
