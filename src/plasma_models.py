def plasma_surrogate(T=20000, dt=0.01, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    x,y = 0.1,-0.1
    out = np.zeros(T)
    w1,w2 = 0.33,0.27
    alpha,beta = 0.15,0.12
    noise = 0.005
    for t in range(T):
        dx = w1*y - alpha*x*(x*x + y*y) + 0.2*np.sin(3*y)
        dy = -w2*x - beta*y*(x*x + y*y) + 0.2*np.sin(3*x)
        x += dt*dx + noise*rng.standard_normal()
        y += dt*dy + noise*rng.standard_normal()
        out[t] = (x*x+y*y)**0.5
    return out
