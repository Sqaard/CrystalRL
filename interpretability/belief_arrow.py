# N7-v2 "belief-arrow": repair the refuted N7 gate by measuring time-irreversibility on the BELIEF
# stream (the regime posterior the policy filters), not the action stream. Bayesian filtering of an
# asymmetric world is irreversible (slow evidence build -> fast reset); a contemporaneous action a=f(belief)
# can still read symmetric (which is why action-N7 was refuted). Validate on known-truth controls.
import numpy as np, itertools
rng=np.random.default_rng(7)

def perm_irrev(x,m=3):
    x=np.asarray(x,float); x=x+1e-9*rng.standard_normal(x.size)
    P={p:i for i,p in enumerate(itertools.permutations(range(m)))}
    idx=[P[tuple(np.argsort(x[k:k+m],kind='stable'))] for k in range(len(x)-m+1)]
    if len(idx)<12: return np.nan
    H=np.bincount(idx,minlength=len(P)).astype(float); H/=H.sum()
    inv={v:k for k,v in P.items()}; mir=np.array([P[tuple(reversed(inv[i]))] for i in range(len(P))])
    return 0.5*np.abs(H-H[mir]).sum()

def pct(x,reps=300):
    r=perm_irrev(x); nul=np.array([perm_irrev(rng.permutation(x)) for _ in range(reps)])
    nul=nul[~np.isnan(nul)]; return r,(nul<=r).mean()*100

def hmm_filter(n, asym=True):
    # 2 regimes: 0=benign,1=toxic. Transition + emissions; run exact Bayes filter -> belief P(toxic).
    if asym:
        T=np.array([[0.985,0.015],[0.10,0.90]])       # toxic episodes SHORT (fast clear)
        mu=np.array([0.0,0.6]); sig=np.array([1.0,1.4]) # toxic weakly separated -> evidence builds SLOWLY
    else:
        T=np.array([[0.95,0.05],[0.05,0.95]]); mu=np.array([0.0,1.2]); sig=np.array([1.0,1.0])
    r=np.zeros(n,int)
    for t in range(1,n): r[t]=rng.random()>T[r[t-1],r[t-1]] and 1-r[t-1] or r[t-1]
    sig_obs=mu[r]+sig[r]*rng.standard_normal(n)
    b=np.zeros(n); b[0]=0.5
    for t in range(1,n):
        pri=b[t-1]*T[1,1]+(1-b[t-1])*T[0,1]           # prior P(toxic)
        L1=np.exp(-0.5*((sig_obs[t]-mu[1])/sig[1])**2)/sig[1]
        L0=np.exp(-0.5*((sig_obs[t]-mu[0])/sig[0])**2)/sig[0]
        b[t]=pri*L1/(pri*L1+(1-pri)*L0+1e-12)
    # contemporaneous action with hysteresis: provide/wait/aggress by belief band
    a=np.zeros(n,int); state=0
    for t in range(n):
        if state==0 and b[t]>0.6: state=1
        elif state==1 and b[t]<0.4: state=0
        a[t]= 2 if b[t]>0.75 else (1 if state==1 else 0)
    return b,a

N=1500
b_asym,a_asym=hmm_filter(N,asym=True)
b_sym ,a_sym =hmm_filter(N,asym=False)
flat=0.5+0.02*rng.standard_normal(N)                 # persister: no regime inference (flat belief)
noise=rng.random(N)
streams={
 "BELIEF asym-regime (Bayes filter)":b_asym,
 "ACTION asym-regime (contemporaneous)":a_asym.astype(float),
 "BELIEF sym-regime (control)":b_sym,
 "ACTION sym-regime":a_sym.astype(float),
 "BELIEF flat / persister (control)":flat,
 "iid noise (control)":noise,
}
print(f"{'stream':40s}  T_perm  shuffle_pct  verdict")
for nm,x in streams.items():
    t,p=pct(x); v="ASYMMETRIC (arrow)" if p>=95 else ("border" if p>=90 else "symmetric")
    print(f"{nm:40s}  {t:.3f}   {p:5.1f}      {v}")
