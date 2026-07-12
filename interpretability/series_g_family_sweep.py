# S2 — Series-G family sweep: trace the (complexity, interpretability) frontier as #regimes G rises,
# and locate the interpretability collapse C*. Self-contained numpy (analytic belief-MAP optimum, no torch).
# Predicts (W3): simulatability@K stays high while G<=K, collapses when G>K  => C* ~= K (the concept budget).
import numpy as np, itertools, json
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
rng=np.random.default_rng(3)

def run_regime_pomdp(G, n=3000, p_stay=0.9, sep=3.0, sigma=1.0):
    # hidden Markov regime r_t (G states); each regime r emits obs ~ N(sep*e_r, sigma) in G-dim.
    T=np.full((G,G),(1-p_stay)/(G-1)); np.fill_diagonal(T,p_stay)
    r=np.zeros(n,int)
    for t in range(1,n):
        r[t]=rng.choice(G,p=T[r[t-1]])
    mu=sep*np.eye(G)
    obs=mu[r]+sigma*rng.standard_normal((n,G))
    # exact Bayesian filter -> belief b_t (G-dim)
    b=np.zeros((n,G)); b[0]=np.ones(G)/G
    logC=-0.5/sigma**2
    for t in range(1,n):
        pri=b[t-1]@T
        L=np.exp(logC*((obs[t]-mu)**2).sum(1)); post=pri*L; b[t]=post/(post.sum()+1e-12)
    a=b.argmax(1)                                   # belief-MAP optimal action (A=G actions)
    return b,a

def cond_entropy_bits(a, G):                        # h_mu proxy: H(a_t | a_{t-1}) in bits/action
    n=len(a); joint=np.zeros((G,G))
    for t in range(1,n): joint[a[t-1],a[t]]+=1
    joint/=joint.sum(); px=joint.sum(1,keepdims=True)+1e-12
    with np.errstate(divide='ignore',invalid='ignore'):
        cond=joint/px; term=joint*np.log2(np.where(cond>0,cond,1))
    return float(-term.sum())

def simulatability(b, a, K, folds=5):              # OOS: K-code story over BELIEF -> action, vs trivial base
    if len(set(a))<2: return 1.0
    km=KMeans(K,n_init=3,random_state=0).fit(b); lab=km.labels_
    kf=KFold(folds,shuffle=True,random_state=0); correct=0; base=0; ntot=0
    for tr,te in kf.split(b):
        maj={c:(np.bincount(a[tr][lab[tr]==c]).argmax() if (lab[tr]==c).any() else np.bincount(a[tr]).argmax()) for c in range(K)}
        pred=np.array([maj[l] for l in lab[te]])
        correct+=(pred==a[te]).sum(); base+=(np.bincount(a[tr]).argmax()==a[te]).sum(); ntot+=len(te)
    acc=correct/ntot; b0=base/ntot
    return float(np.clip((acc-b0)/(1-b0+1e-12),0,1))

def belief_entropy_bits(b):
    return float(np.mean(-(b*np.log2(np.clip(b,1e-12,1))).sum(1)))

K=9
print(f"Fixed concept budget K={K}.  Frontier as #regimes G rises:")
print(f"{'G':>3} {'bits/action(x)':>14} {'simul@K9(y)':>12} {'belief_H':>9}")
rows={}
for G in [2,3,4,5,6,8,10,12]:
    b,a=run_regime_pomdp(G)
    x=cond_entropy_bits(a,G); y=simulatability(b,a,K); bh=belief_entropy_bits(b)
    rows[G]=dict(x_bits=round(x,3),y_simul_K9=round(y,3),belief_H=round(bh,3))
    print(f"{G:>3} {x:>14.3f} {y:>12.3f} {bh:>9.3f}")
# K-sweep at high complexity (G=12) -> show C* ~= K
G=12; b,a=run_regime_pomdp(G)
ks={k:round(simulatability(b,a,k),3) for k in [2,4,6,9,12,16]}
print(f"\nK-sweep at G={G} (interpretability vs concept budget): {ks}")
json.dump({'fixed_K':K,'frontier':rows,'ksweep_G12':ks},open("interpretability/series_g_family_sweep_report.json","w"),indent=2)
print("saved interpretability/series_g_family_sweep_report.json")
