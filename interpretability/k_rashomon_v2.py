# K (Rashomon) paper-grade: N=500 descriptions, cross-seed-ARI canonical tie-breaker,
# + empirical unicity-distance curve (Rashomon ratio vs evidence length n).
import numpy as np, pandas as pd, json, time
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.metrics import adjusted_rand_score
rng = np.random.default_rng(11)
t0=time.time()

def _lag(a,k): return np.concatenate([np.full(k,np.nan), a[:-k]])
def build_state(x, r=None):
    df={'l1':_lag(x,1),'l2':_lag(x,2),'l3':_lag(x,3),
        'rm5':pd.Series(x).rolling(5).mean().shift(1).to_numpy()}
    if r is not None: df['r1']=_lag(r,1); df['r2']=_lag(r,2)
    S=pd.DataFrame(df); y=pd.Series(x); m=S.notna().all(axis=1)
    return S[m].to_numpy(), y[m].to_numpy(), list(S.columns)
def std(X): return (X-X.mean(0))/(X.std(0)+1e-9)

def oos_loss(S,y,feats,K,seed,folds=3):
    Xf=S[:,feats]; kf=KFold(folds,shuffle=True,random_state=seed); err=0.0;nte=0
    for tr,te in kf.split(Xf):
        mu=Xf[tr].mean(0);sd=Xf[tr].std(0)+1e-9
        km=KMeans(K,n_init=1,random_state=seed).fit((Xf[tr]-mu)/sd)
        cm=np.array([y[tr][km.labels_==c].mean() if (km.labels_==c).any() else y[tr].mean() for c in range(K)])
        pred=cm[km.predict((Xf[te]-mu)/sd)]; err+=np.sum((y[te]-pred)**2); nte+=len(te)
    return err/nte
def full_pred(S,y,feats,K,seed):
    km=KMeans(K,n_init=1,random_state=seed).fit(std(S[:,feats]))
    cm=np.array([y[km.labels_==c].mean() if (km.labels_==c).any() else y.mean() for c in range(K)])
    return cm[km.labels_]
def stability(S,feats,K,seeds=(1,2,3,4,5)):
    labs=[KMeans(K,n_init=1,random_state=s).fit(std(S[:,feats])).labels_ for s in seeds]
    a=[adjusted_rand_score(labs[i],labs[j]) for i in range(len(labs)) for j in range(i+1,len(labs))]
    return float(np.mean(a))

def rashomon(x,r=None,N=500,tiebreak=True):
    S,y,cols=build_state(np.asarray(x,float),None if r is None else np.asarray(r,float))
    base=float(np.var(y)); nf=S.shape[1]; descs=[]; preds=[]
    for i in range(N):
        K=int(rng.integers(2,10)); k=int(rng.integers(1,nf+1))
        feats=sorted(rng.choice(nf,k,replace=False).tolist())
        descs.append((oos_loss(S,y,feats,K,i),K,tuple(feats),i))
        preds.append(full_pred(S,y,feats,K,i))
    losses=np.array([d[0] for d in descs]); preds=np.array(preds)
    Lstar=float(min(losses.min(),base)); struct=max(0.,1-Lstar/base)
    curve={}
    for eps in [0.02,0.05,0.1,0.25,0.5]:
        inset=np.where(losses<=Lstar*(1+eps))[0]; ratio=len(inset)/len(losses)
        if len(inset)>=2:
            idx=inset if len(inset)<=60 else rng.choice(inset,60,replace=False)
            P=preds[idx]; d=[np.sqrt(np.mean((P[a]-P[b])**2)) for a in range(len(idx)) for b in range(a+1,len(idx))]
            diam=float(np.mean(d)/(np.std(y)+1e-9))
        else: diam=0.0
        curve[str(eps)]={'ratio':round(ratio,3),'diameter':round(diam,3),'set_size':int(len(inset))}
    canon=None
    if tiebreak:
        inset=np.where(losses<=Lstar*1.1)[0]
        cand=inset if len(inset)<=40 else inset[np.argsort(losses[inset])[:40]]
        best=None
        for j in cand:
            L,K,feats,seed=descs[j]; stab=stability(S,list(feats),K)
            sc=(stab,-K,-len(feats))
            if best is None or sc>best[0]:
                best=(sc,{'K':K,'features':[cols[f] for f in feats],'oos_loss':round(L,5),'stability_ari':round(stab,3)})
        canon=best[1] if best else None
    return {'n':int(len(y)),'baseline':round(base,5),'Lstar':round(Lstar,5),
            'structure_frac':round(struct,3),'curve':curve,'canonical':canon}

def persister(n=277):
    x=np.zeros(n);x[0]=rng.random()
    for t in range(1,n): x[t]=x[t-1] if rng.random()<0.85 else rng.random()
    return x
def ar1(n=277):
    x=np.zeros(n)
    for t in range(1,n): x[t]=0.8*x[t-1]+0.3*rng.standard_normal()
    return x
def iid(n=277): return rng.random(n)
def load(p,c):
    d=pd.read_csv(p); return d[c].to_numpy(float),(d['net_return'].to_numpy(float) if 'net_return' in d.columns else None)

import sys; group=sys.argv[1] if len(sys.argv)>1 else "headline"
R6C="artifacts/stage4/R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_frozen_2022_2023_for_Joseph/frozen_test_behavior_log_daily.csv"
r6c_x,r6c_r=load("interpretability/_streams/r6c_deadline_daily.csv","cash")
p22_x,p22_r=load("interpretability/_streams/p22_deadline_daily.csv","cash")

if group=="headline":
    streams={"R6c csi500 (cash)":(r6c_x,r6c_r),"P22 csi500 (cash)":(p22_x,p22_r),"R6c Dow (cash_target)":load(R6C,"cash_target")}
elif group=="controls":
    streams={"CTRL persister*":(persister(),None),"CTRL ar1*":(ar1(),None),"CTRL iid noise*":(iid(),None)}
else:
    streams={}

full={}
if streams:
    print("stream                  n   struct  r@.10  r@.25  canon_stab  canon", flush=True)
    for name,(x,r) in streams.items():
        res=rashomon(x,r,N=500); full[name]=res; c=res['curve']; cc=res['canonical'] or {}
        print(f"{name:22s} {res['n']:4d}  {res['structure_frac']:.3f}  {c['0.1']['ratio']:.2f}   {c['0.25']['ratio']:.2f}   {cc.get('stability_ari','-')}   K{cc.get('K','?')} {cc.get('features','')}", flush=True)
        print(f"   +{round(time.time()-t0,1)}s", flush=True)
if group=="unicity":
    uni={}
    for name,(x,r) in [("R6c csi500",(r6c_x,r6c_r)),("P22 csi500",(p22_x,p22_r))]:
        row={}
        for n in [60,110,160,210,len(x)]:
            rr=rashomon(x[:n], None if r is None else r[:n], N=150, tiebreak=False)
            row[str(n)]={'ratio_e10':rr['curve']['0.1']['ratio'],'structure_frac':rr['structure_frac']}
            print("unicity",name,n,row[str(n)], flush=True)
        uni[name]=row
    full['_unicity_vs_n']=uni

json.dump(full, open(f"interpretability/k_v2_{group}.json","w"), indent=2)
print(f"GROUP {group} DONE {round(time.time()-t0,1)}s", flush=True)
