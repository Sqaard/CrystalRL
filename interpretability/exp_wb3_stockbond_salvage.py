import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from interpretability.exp_cl1_new_eyes_continual import load_v2
from interpretability.exp_w1_ktb_hierarchy import block_features
from interpretability.exp_wp1_completeness import strong_null_margin
from interpretability.exp_wp2_queue import etf_ret
from interpretability.exp_w1_ktb_v2 import build_blocks, assemble, cell
from interpretability.hl_v9_fresh_oos import TRAIN, DEV, HOLD, OOS
from sklearn.decomposition import PCA

def build_wl(r, obs, wl_daily, B, D, tr_end):
    # within-block: each block gets ONE stat computed only from its own B days (no cross-block window)
    rows,b_end,idx=block_features(r,obs,B)
    valid0=[i for i,x in enumerate(rows) if x is not None]
    bk=r.to_numpy(); wl=wl_daily.reindex(r.index).to_numpy()
    X,valid=[],[]
    for i in valid0:
        e=b_end[i]; s=e-B+1
        seg_r=bk[s:e+1]; seg_w=wl[s:e+1]
        if not (np.isfinite(seg_r).all() and np.isfinite(seg_w).all()): continue
        # block-local corr of book vs bond within this block only
        c=float(np.corrcoef(seg_r,seg_w)[0,1]) if seg_r.std()>1e-9 and seg_w.std()>1e-9 else 0.0
        X.append(list(rows[i])+[c]); valid.append(i)
    X=np.array(X,dtype=np.float32); remap={i:j for j,i in enumerate(valid)}
    dts=idx[b_end[valid]]; m_tr=np.asarray(dts<=pd.Timestamp(tr_end))
    mu,sd=X[m_tr].mean(0),X[m_tr].std(0)+1e-9; Z=(X-mu)/sd
    pca=PCA(n_components=min(D,Z.shape[1]),random_state=0).fit(Z[m_tr])
    return {'S':pca.transform(Z).astype(np.float32),'remap':remap,'valid':set(valid),
            'dates_all':idx[b_end],'n':len(rows),'B':B}

r,obs,rf=load_v2()
ief=etf_ret('IEF',r.index)
# placebo market check first (the decisive guard)
rng=np.random.default_rng(777); perm=rng.permutation(len(r))
r_p=pd.Series(r.to_numpy()[perm],index=r.index)
obs_p=pd.DataFrame(obs.to_numpy()[perm],index=obs.index,columns=obs.columns)
ief_p=pd.Series(ief.to_numpy()[perm],index=r.index)
bl_p=build_wl(r_p,obs_p,ief_p,5,11,TRAIN[1])
cp=cell(bl_p,assemble(bl_p,2,(1,2,4,8)),1,boot_block=2)
print('WITHIN-BLOCK stockbond, PLACEBO market L1 k=1:', cp['hold'])
# real market: does within-block stockbond ADD over base, beat noise twins?
rows,b_end,idx=block_features(r,obs,5)
valid0=[i for i,x in enumerate(rows) if x is not None]
bk=r.to_numpy(); wl=ief.reindex(r.index).to_numpy()
Xc,ok,valid=[],[],[]
for i in valid0:
    e=b_end[i]; s=e-4; seg_r=bk[s:e+1]; seg_w=wl[s:e+1]
    c=float(np.corrcoef(seg_r,seg_w)[0,1]) if (np.isfinite(seg_w).all() and seg_r.std()>1e-9 and seg_w.std()>1e-9) else np.nan
    Xc.append([c]); ok.append(np.isfinite(c))
Xc=np.array(Xc,dtype=np.float32); ok=np.array(ok)
valid=[v for v,o in zip(valid0,ok) if o]
X10=np.array([rows[i] for i in valid],dtype=np.float32); Xc=Xc[ok]
remap={i:j for j,i in enumerate(valid)}; dts=idx[b_end[valid]]
m_tr=np.asarray(dts<=pd.Timestamp(TRAIN[1])); yrs=np.asarray(pd.DatetimeIndex(dts).year)
def mk(Xf,D):
    mu,sd=Xf[m_tr].mean(0),Xf[m_tr].std(0)+1e-9; Z=(Xf-mu)/sd
    return PCA(n_components=min(D,Z.shape[1]),random_state=0).fit(Z[m_tr]).transform(Z).astype(np.float32)
def al(ctx,KS=(1,4)):
    n=max(valid)+1; vs=set(valid)
    samp=np.array([i for i in range(ctx-1,n-4) if all(j in vs for j in list(range(i-ctx+1,i+1))+[i+k for k in KS])])
    dd=idx[b_end[samp]]
    mk2={w:np.asarray((dd>=pd.Timestamp(a))&(dd<=pd.Timestamp(b))) for w,(a,b) in dict(train=TRAIN,dev=DEV,hold=HOLD,oos=OOS).items()}
    co=np.array([[remap[j] for j in range(i-ctx+1,i+1)] for i in samp])
    return {'masks':mk2,'ctx_of':co,'cur_of':co[:,-1],'tgt_of':{k:np.array([remap[i+k] for i in samp]) for k in KS},'tgt_dates':{k:idx[b_end[samp+k]] for k in KS}}
for k,ctx in ((1,2),(4,4)):
    asm=al(ctx)
    base=strong_null_margin(mk(X10,10),asm['ctx_of'],asm['cur_of'],asm['tgt_of'][k],asm['tgt_dates'][k],asm['masks'],k,ctx)
    cand=strong_null_margin(mk(np.concatenate([X10,Xc],1),11),asm['ctx_of'],asm['cur_of'],asm['tgt_of'][k],asm['tgt_dates'][k],asm['masks'],k,ctx)
    nz=[]
    for sdd in range(5):
        Xn=Xc.copy(); uy=np.unique(yrs); pm=dict(zip(uy,np.random.default_rng(700+sdd).permutation(uy)))
        for y in uy:
            src=np.where(yrs==pm[y])[0]; dst=np.where(yrs==y)[0]; Xn[dst]=Xc[np.resize(src,len(dst))]
        nz.append(strong_null_margin(mk(np.concatenate([X10,Xn],1),11),asm['ctx_of'],asm['cur_of'],asm['tgt_of'][k],asm['tgt_dates'][k],asm['masks'],k,ctx)['hold']['margin'])
    print(f'REAL within-block k={k}: base {base["hold"]["margin"]} | +cand {cand["hold"]["margin"]} z {cand["hold"]["z"]} | noise {nz} | OOS cand {cand["oos"]["margin"]}')
