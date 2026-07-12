# R (Ontogeny) first curve: interpretability-vs-training-step on 6 real PPO checkpoints.
# Torch-free: load SB3 policy weights from the .pth zips, run the MLP forward in numpy on a FIXED probe
# observation set, and measure per checkpoint: (a) Simulatability = OOS R^2 of a K-code regime story
# reproducing the policy's action; (b) behavioral complexity = participation ratio of the action;
# (c) policy width = mean action std. Then read the curve vs training step (mirage control: K=4 vs K=9).
import zipfile, pickle, io, glob, re, json, numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
rng=np.random.default_rng(0)
DT={'FloatStorage':np.float32,'DoubleStorage':np.float64,'HalfStorage':np.float16,'LongStorage':np.int64,
    'IntStorage':np.int32,'ShortStorage':np.int16,'CharStorage':np.int8,'ByteStorage':np.uint8,'BoolStorage':np.bool_}
class St:
    def __init__(s,dt,key): s.dt=dt; s.key=key
def load_pth_bytes(b):
    z=zipfile.ZipFile(io.BytesIO(b)); pk=[n for n in z.namelist() if n.endswith('data.pkl')][0]; pre=pk[:-8]
    def rb(st,off,size,stride,*r):
        a=np.frombuffer(z.read(pre+'data/'+st.key),dtype=st.dt); n=int(np.prod(size)) if len(size) else 1
        return a[off:off+n].reshape(size) if len(size) else a[off:off+1]
    class U(pickle.Unpickler):
        def find_class(s,m,nm):
            if m=='torch._utils' and nm.startswith('_rebuild_tensor'): return rb
            if m=='torch' and nm.endswith('Storage'): return nm
            if m=='collections' and nm=='OrderedDict':
                from collections import OrderedDict; return OrderedDict
            try: return super().find_class(m,nm)
            except Exception: return lambda *a,**k: None
        def persistent_load(s,pid):
            t=pid[1]; k=str(pid[2]); tn=t if isinstance(t,str) else getattr(t,'__name__',str(t))
            return St(DT.get(tn,np.float32),k)
    return U(z.open(pk)).load()

def forward(sd,O):
    def L(x,w,b): return x@sd[w].T+sd[b]
    h=np.tanh(L(O,'mlp_extractor.policy_net.0.weight','mlp_extractor.policy_net.0.bias'))
    h=np.tanh(L(h,'mlp_extractor.policy_net.2.weight','mlp_extractor.policy_net.2.bias'))
    return L(h,'action_net.weight','action_net.bias')          # (n,29) mean action

def simulatability(O,A,K,folds=5):
    # OOS R^2 of predicting each action dim from K regime-codes over observations (piecewise-constant)
    km=KMeans(K,n_init=3,random_state=0).fit(O); lab=km.labels_
    kf=KFold(folds,shuffle=True,random_state=0); sse=np.zeros(A.shape[1]); sst=np.zeros(A.shape[1])
    for tr,te in kf.split(O):
        for c in range(K):
            m=lab[tr]==c
            pred=A[tr][m].mean(0) if m.any() else A[tr].mean(0)
            mm=lab[te]==c
            if mm.any(): sse+=((A[te][mm]-pred)**2).sum(0)
        sst+=((A[te]-A[tr].mean(0))**2).sum(0)
    r2=1-sse/(sst+1e-12); return float(np.clip(r2,0,1).mean())

def complexity_PR(A):
    C=np.cov(A.T); w=np.linalg.eigvalsh(C); w=np.clip(w,0,None)
    pr=(w.sum()**2)/((w**2).sum()+1e-12); return float(pr/A.shape[1])   # 0..1 effective-dim fraction

def regime_r2(A,g,G):
    tot=((A-A.mean(0))**2).sum(); within=0.0
    for c in range(G):
        m=g==c
        if m.any(): within+=((A[m]-A[m].mean(0))**2).sum()
    return float(np.clip(1-within/(tot+1e-12),0,1))

import glob,re
files=sorted(glob.glob("PPO_configurations_comparison/Experiments/1st_experiment_best_timestep/ppo_checkpoints/ppo_model_*_steps.zip"),
             key=lambda f:int(re.search(r'_(\d+)_steps',f).group(1)))
sd0=load_pth_bytes(zipfile.ZipFile(files[0]).read('policy.pth'))
obs_dim=sd0['mlp_extractor.policy_net.0.weight'].shape[1]
O=rng.standard_normal((500,obs_dim)).astype(np.float32)          # iid probe (distribution-free floor)
G=9; centers=(rng.standard_normal((G,obs_dim))*0.7).astype(np.float32)
g=rng.integers(0,G,500); Os=(centers[g]+0.7*rng.standard_normal((500,obs_dim))).astype(np.float32)  # regime probe
print(f"obs_dim={obs_dim} probe n=500 checkpoints={len(files)}\n")
print("step     simul_iid  regimeR2  action_gain  complexity_PR  policy_width")
rows={}
for f in files:
    step=int(re.search(r'_(\d+)_steps',f).group(1))
    sd=load_pth_bytes(zipfile.ZipFile(f).read('policy.pth'))
    A=forward(sd,O); As=forward(sd,Os)
    s_iid=simulatability(O,A,9); rr=regime_r2(As,g,G)
    gain=float(np.mean(As.std(0))); pr=complexity_PR(A)
    width=float(np.exp(sd['log_std']).mean()) if 'log_std' in sd else float('nan')
    rows[step]=dict(simul_iid=round(s_iid,3),regime_r2=round(rr,3),action_gain=round(gain,3),complexity_PR=round(pr,3),policy_width=round(width,3))
    print(f"{step:7d}   {s_iid:.3f}     {rr:.3f}     {gain:.3f}       {pr:.3f}         {width:.3f}")
json.dump(rows,open("interpretability/r_crystallization_report.json","w"),indent=2)
print("\nsaved interpretability/r_crystallization_report.json")
