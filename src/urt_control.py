import numpy as np
def urt_prescribe(tau_sensed, D_KY, C=0.52, N=8, delta_target=0.02, base_level=1.0):
    if not (np.isfinite(tau_sensed) and np.isfinite(D_KY)):
        return {"status":"insufficient_metrics"}
    eps = max(D_KY-1.0,1e-9)
    tau_target = 2+ C/eps
    if tau_sensed>=0.99*tau_target:
        return {"status":"stable","tau_target":float(tau_target)}
    theta=np.deg2rad(137.5); g=np.pi/np.e
    w=np.zeros(N); w[0]=base_level*g*delta_target
    for i in range(1,N):
        phase=1+0.5*np.cos(theta*i)
        growth=g*(1+delta_target/max(i,1))
        w[i]=max(0.0,w[i-1]*phase*growth)
    if w.sum()<=0:w[:]=base_level/N
    else:w=(w/w.sum())*base_level*N
    return {"status":"urt_correction","tau_target":float(tau_target),"weights":[float(x) for x in w]}
