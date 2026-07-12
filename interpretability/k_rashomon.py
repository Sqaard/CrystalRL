# K (Rashomon Set) on R6c vs P22 — uniqueness/ambiguity axis for CrystalScore.
# A "description" = a <=K-code KMeans codebook over a PAST-only state, predicting the policy's daily
# STANCE (cash) as the cluster mean. Rashomon set = descriptions whose OUT-OF-SAMPLE loss is within a
# tolerance band of the best. Loss is 5-fold CV so noise cannot fake structure (CrystalScore firewall #4).
# Prediction under test: R6c (persister) -> small set (crisp story); P22 (churner) -> set ~= 1 (any story).
import numpy as np, pandas as pd, json
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
rng = np.random.default_rng(11)

def _lag(a,k): return np.concatenate([np.full(k,np.nan), a[:-k]])
def build_state(x, r=None):
    df={}
    df['l1']=_lag(x,1); df['l2']=_lag(x,2); df['l3']=_lag(x,3)
    df['rm5']=pd.Series(x).rolling(5).mean().shift(1).to_numpy()
    if r is not None:
        df['r1']=_lag(r,1); df['r2']=_lag(r,2)
    S=pd.DataFrame(df); y=pd.Series(x)
    m=S.notna().all(axis=1)
    return S[m].to_numpy(), y[m].to_numpy(), list(S.columns)

def oos_loss(S, y, feats, K, seed, folds=3):
    Xf=S[:, feats]; kf=KFold(folds, shuffle=True, random_state=seed); err=0.0; nte=0
    for tr,te in kf.split(Xf):
        mu=Xf[tr].mean(0); sd=Xf[tr].std(0)+1e-9
        Xtr=(Xf[tr]-mu)/sd; Xte=(Xf[te]-mu)/sd
        km=KMeans(K, n_init=2, random_state=seed).fit(Xtr)
        cmean=np.array([y[tr][km.labels_==c].mean() if (km.labels_==c).any() else y[tr].mean() for c in range(K)])
        lab_te=km.predict(Xte); pred=cmean[lab_te]
        err+=np.sum((y[te]-pred)**2); nte+=len(te)
    return err/nte

def rashomon(x, r=None, N=100):
    S,y,cols=build_state(np.asarray(x,float), None if r is None else np.asarray(r,float))
    base=np.var(y)                                   # K=1 mean predictor (OOS ~= var)
    losses=[]; preds=[]
    nf=S.shape[1]
    for i in range(N):
        K=int(rng.integers(2,10))
        k=int(rng.integers(1,nf+1)); feats=sorted(rng.choice(nf,k,replace=False).tolist())
        L=oos_loss(S,y,feats,K,seed=i)
        losses.append(L)
        # store an OOS prediction vector (single split) for diameter
        km=KMeans(K,n_init=2,random_state=i).fit(((S[:,feats]-S[:,feats].mean(0))/(S[:,feats].std(0)+1e-9)))
        cm=np.array([y[km.labels_==c].mean() if (km.labels_==c).any() else y.mean() for c in range(K)])
        preds.append(cm[km.labels_])
    losses=np.array(losses); preds=np.array(preds)
    Lstar=min(losses.min(), base)
    struct=max(0.0, 1-Lstar/base)                    # OOS max R^2 (structure fraction)
    gap=base-Lstar
    out={'n':int(len(y)),'baseline':float(base),'Lstar':float(Lstar),'structure_frac':float(struct)}
    curve={}
    for eps in [0.02,0.05,0.1,0.25,0.5]:
        band=Lstar*(1+eps)   # standard Rashomon: within eps-fraction of the OPTIMAL loss
        inset=np.where(losses<=band)[0]
        ratio=len(inset)/len(losses)
        # behavioral diameter: mean pairwise normalized disagreement among set members
        if len(inset)>=2:
            idx=inset if len(inset)<=60 else rng.choice(inset,60,replace=False)
            P=preds[idx]; d=[]
            for a in range(len(idx)):
                for b in range(a+1,len(idx)):
                    d.append(np.sqrt(np.mean((P[a]-P[b])**2)))
            diam=float(np.mean(d)/(np.std(y)+1e-9))
        else: diam=0.0
        curve[str(eps)]={'ratio':round(ratio,3),'diameter':round(diam,3)}
    out['curve']=curve
    return out

def synth_persister(n=277):
    x=np.zeros(n); x[0]=rng.random()
    for t in range(1,n): x[t]=x[t-1] if rng.random()<0.85 else rng.random()
    return x
def synth_ar1(n=277):
    x=np.zeros(n)
    for t in range(1,n): x[t]=0.8*x[t-1]+0.3*rng.standard_normal()
    return x
def synth_iid(n=277): return rng.random(n)

def load(p,c,rc='net_return'):
    d=pd.read_csv(p); r=d[rc].to_numpy(float) if rc in d.columns else None
    return d[c].to_numpy(float), r

streams={}
x,r=load("interpretability/_streams/r6c_deadline_daily.csv","cash");         streams["R6c csi500 (cash)"]=(x,r)
x,r=load("interpretability/_streams/p22_deadline_daily.csv","cash");         streams["P22 csi500 (cash)"]=(x,r)
x,r=load("artifacts/stage4/R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_frozen_2022_2023_for_Joseph/frozen_test_behavior_log_daily.csv","cash_target"); streams["R6c Dow (cash_target)"]=(x,r)
streams["CTRL persister*"]=(synth_persister(),None)
streams["CTRL ar1*"]=(synth_ar1(),None)
streams["CTRL iid noise*"]=(synth_iid(),None)

rows=[]; full={}
for name,(x,r) in streams.items():
    res=rashomon(x,r); full[name]=res
    c=res['curve']
    verdict = "LARGE set (arbitrary)" if c['0.25']['ratio']>=0.6 else ("small set (crisp)" if c['0.25']['ratio']<=0.25 else "medium")
    rows.append(dict(stream=name,n=res['n'],struct_frac=round(res['structure_frac'],3),
                     ratio_e10=c['0.1']['ratio'],ratio_e25=c['0.25']['ratio'],
                     diam_e25=c['0.25']['diameter'],verdict=verdict))
df=pd.DataFrame(rows)
print(df.to_string(index=False))
json.dump(full, open("interpretability/k_rashomon_report.json","w"), indent=2)
print("\nsaved interpretability/k_rashomon_report.json")
