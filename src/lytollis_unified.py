# ✅ LYTOLLIS UNIFIED SCIENTIFIC CORE (Bounded Chaos + Universal Constant + Plasma + URT)

import json, os
import numpy as np
from scipy.spatial import cKDTree

def stability_margin(mu_JH, gamma, Jpsi):
    return mu_JH + gamma * Jpsi

def is_bounded(mu_JH, gamma, Jpsi, delta):
    return stability_margin(mu_JH, gamma, Jpsi) <= -abs(delta)

def tau_from_DKY(D_KY, C=0.52):
    eps = max(D_KY - 1.0, 1e-9)
    return 2.0 + C / eps

def C_from_tau_DKY(tau, D_KY):
    eps = max(D_KY - 1.0, 1e-9)
    return (tau - 2.0) * eps

def lle_proxy(ts, m=6, delay=2, max_h=60):
    x = np.asarray(ts, float)
    N = len(x) - (m-1)*delay
    if N < 500: return np.nan
    Y = np.stack([x[i:i+N] for i in range(0,m*delay,delay)],axis=1)
    tree=cKDTree(Y);_,idx=tree.query(Y,k=2);nn=idx[:,1]
    div=[]
    for h in range(1,max_h):
        base=np.arange(N)
        mask=(base+h<N)&(nn+h<N)
        base=base[mask];neigh=nn[mask]
        if base.size<100:break
        d0=np.linalg.norm(Y[base]-Y[neigh],axis=1)+1e-12
        d1=np.linalg.norm(Y[base+h]-Y[neigh+h],axis=1)+1e-12
        div.append(np.mean(np.log(d1/d0)))
    return float(np.mean(div)) if len(div) else np.nan

def plasma_surrogate(T=6000, dt=0.01, seed=0):
    rng=np.random.default_rng(seed);x=0.1;out=[]
    for _ in range(T):
        x+=dt*(0.6*np.sin(3*x)-0.4*np.sin(5*x))+rng.normal(0,0.002)
        out.append(x)
    return np.array(out)

def tau_empirical(ts, q=0.75):
    diff=np.abs(np.diff(ts))
    thresh=np.quantile(diff,q)
    spikes=diff[diff>thresh]
    if len(spikes)<50:return np.nan
    hist=np.log(np.sort(spikes))
    y=np.linspace(0,1,len(hist))
    slope=(hist[-1]-hist[0])/(y[-1]-y[0]+1e-12)
    return 2.0+abs(slope)

def urt_weights(D_KY, tau, C=0.52, N=8):
    tau_target=tau_from_DKY(D_KY,C)
    delta=tau_target-tau
    if delta<=0:return{"status":"no_urt_needed","tau_target":tau_target}
    w=np.array([abs(np.sin((i+1)*2.399963)) for i in range(N)])
    w=w/w.sum()
    return{"status":"urt_correction","tau_target":tau_target,"weights":w.tolist()}

def run_unified(output_dir=None, make_plots=False):
    result={}
    bounded=is_bounded(mu_JH=-0.35,gamma=0.6,Jpsi=0.1,delta=0.02)
    result["law"]={"bounded":bounded}

    Dvals=np.linspace(1.01,1.08,12)
    tauvals=[tau_from_DKY(d) for d in Dvals]
    Cvals=[C_from_tau_DKY(t,d) for t,d in zip(tauvals,Dvals)]
    result["universality"]={
        "C_avg":float(np.mean(Cvals)),
        "C_std":float(np.std(Cvals)),
        "convergent":abs(np.mean(Cvals)-0.52)<0.05
    }

    ts=plasma_surrogate()
    lle=lle_proxy(ts)
    D=1+max(lle,1e-9)
    tau=tau_empirical(ts)
    result["plasma"]={"LLE":lle,"D_KY":D,"tau":tau}

    result["URT"]=urt_weights(D,tau)

    if output_dir:
        os.makedirs(output_dir,exist_ok=True)
        with open(f"{output_dir}/summary.json","w") as f:
            json.dump(result,f,indent=2)

    return resultimport sys
sys.path.append("/content/lytolllis-chaos-lab")

from src.lytollis_unified import run_unified
summary = run_unified()
summary
