import numpy as np
from scipy import stats, optimize, special

def hedges_J(m):  # exact, m = n1+n2-2
    return np.exp(special.gammaln(m/2) - np.log(np.sqrt(m/2)) - special.gammaln((m-1)/2))

def cohens_d(m1, sd1, n1, m2, sd2, n2):
    sp = np.sqrt(((n1-1)*sd1**2 + (n2-1)*sd2**2)/(n1+n2-2))
    return (m2-m1)/sp

def se_d_rmd(d, n1, n2):  # the analysts' formula (Borenstein 4.20 / Hedges & Olkin)
    return np.sqrt((n1+n2)/(n1*n2) + d**2/(2*(n1+n2-2)))

def vi_metafor_LS(g, n1, n2):  # metafor escalc SMD default: uses g and 2*(n1+n2)
    return 1/n1 + 1/n2 + g**2/(2*(n1+n2))

def he_tau2(y, v):
    k = len(y); return (np.sum((y-y.mean())**2) - (k-1)/k*np.sum(v)) / (k-1)  # (RSS - tr(PV))/(k-1) with p=1

def reml_tau2(y, v, threshold=1e-5, maxiter=100, stepadj=1.0, tau2_min=0.0, tau2_init=None):
    y = np.asarray(y, float); v = np.asarray(v, float); k = len(y)
    tau2 = max(0.0, he_tau2(y, v), tau2_min) if tau2_init is None else tau2_init
    change = threshold + 1; it = 0
    while change > threshold:
        it += 1; old = tau2
        w = 1/(v+tau2); sw = w.sum()
        # P = W - w w'/sum(w) for intercept-only model
        P = np.diag(w) - np.outer(w, w)/sw
        PP = P @ P
        adj = (y @ PP @ y - np.trace(P)) / np.trace(PP)
        adj *= stepadj
        while tau2 + adj < tau2_min: adj /= 2   # step-halving
        tau2 = tau2 + adj
        change = abs(old - tau2)
        if it > maxiter: raise RuntimeError("Fisher scoring did not converge")
    # ll0 check (metafor): compare REML loglik at tau2=0 vs at estimate
    def ll_reml(t):
        w = 1/(v+t); b = np.sum(w*y)/w.sum(); rss = np.sum(w*(y-b)**2)
        return -0.5*(k-1)*np.log(2*np.pi) - 0.5*np.sum(np.log(v+t)) - 0.5*np.log(w.sum()) - 0.5*rss
    if ll_reml(0.0) - ll_reml(tau2) > np.finfo(float).eps**0.25 and tau2 > threshold:
        tau2 = 0.0
    return max(tau2_min, tau2), it

def dl_tau2(y, v):
    w = 1/v; b = np.sum(w*y)/w.sum(); Q = np.sum(w*(y-b)**2); k = len(y)
    C = w.sum() - np.sum(w**2)/w.sum()
    return max(0.0, (Q-(k-1))/C)

def pm_tau2(y, v, tau2_max=100.0):
    k = len(y)
    def f(t):
        w = 1/(v+t); b = np.sum(w*y)/w.sum(); return np.sum(w*(y-b)**2) - (k-1)
    if f(0) <= 0: return 0.0
    tau2_max = max(tau2_max, 10*stats.median_abs_deviation(y, scale='normal')**2)
    return optimize.brentq(f, 0, tau2_max, xtol=np.finfo(float).eps**0.25)

def qprofile_ci(y, v, level=0.95, tau2_max=100.0):
    k = len(y); df = k-1
    def Q(t):
        w = 1/(v+t); b = np.sum(w*y)/w.sum(); return np.sum(w*(y-b)**2)
    lo_crit = stats.chi2.ppf(1-(1-level)/2, df); hi_crit = stats.chi2.ppf((1-level)/2, df)
    tau2_max = max(tau2_max, 10*stats.median_abs_deviation(y, scale='normal')**2)
    lb = 0.0 if Q(0) < lo_crit else optimize.brentq(lambda t: Q(t)-lo_crit, 0, tau2_max, xtol=np.finfo(float).eps**0.25)
    ub = 0.0 if Q(0) < hi_crit else optimize.brentq(lambda t: Q(t)-hi_crit, 0, tau2_max, xtol=np.finfo(float).eps**0.25)
    return lb, ub

def pool(y, v, tau2, level=0.95, hk=False, pi="HTS"):
    y = np.asarray(y, float); v = np.asarray(v, float); k = len(y)
    w = 1/(v+tau2); mu = np.sum(w*y)/w.sum(); se = np.sqrt(1/w.sum())
    wf = 1/v; muf = np.sum(wf*y)/wf.sum(); Q = np.sum(wf*(y-muf)**2); df = k-1
    pQ = stats.chi2.sf(Q, df)
    I2_meta = max(0.0, (Q-df)/Q) if Q > 0 else 0.0
    vt = (k-1)/(wf.sum() - np.sum(wf**2)/wf.sum())            # metafor 'typical' v
    I2_metafor = 100*tau2/(vt+tau2); H2_metafor = tau2/vt + 1
    a = 1-level
    if hk:
        q = np.sum(w*(y-mu)**2)/(k-1); se_hk = np.sqrt(q/w.sum())
        stat = mu/se_hk; p = 2*stats.t.sf(abs(stat), k-1); crit = stats.t.ppf(1-a/2, k-1)
        ci = (mu-crit*se_hk, mu+crit*se_hk); se_used = se_hk
    else:
        stat = mu/se; p = 2*stats.norm.sf(abs(stat)); crit = stats.norm.ppf(1-a/2)
        ci = (mu-crit*se, mu+crit*se); se_used = se
    sepi = np.sqrt(tau2 + se_used**2)
    dfpi = {"HTS": k-2, "V": k-1}.get(pi)
    tq = stats.norm.ppf(1-a/2) if pi in ("S","metafor") else stats.t.ppf(1-a/2, dfpi)
    PI = (mu - tq*sepi, mu + tq*sepi)
    return dict(mu=mu, se=se_used, stat=stat, p=p, ci=ci, Q=Q, pQ=pQ, I2_meta=I2_meta, I2_metafor=I2_metafor, H2=H2_metafor, PI=PI, sepi=sepi, weights=w/w.sum())

def egger_lm(y, se):
    # lm(TE/seTE ~ 1/seTE): intercept = bias estimate (Egger)
    X = np.column_stack([np.ones_like(se), 1/se]); z = y/se
    beta, res, rank, sv = np.linalg.lstsq(X, z, rcond=None)
    resid = z - X@beta; k = len(y); s2 = np.sum(resid**2)/(k-2)
    cov = s2*np.linalg.inv(X.T@X); seb = np.sqrt(np.diag(cov))
    t = beta/seb; p = 2*stats.t.sf(abs(t), k-2)
    return beta, seb, t, p

if __name__ == "__main__":
    m1=np.array([10.0,12.5,8.0,15.0,11.0]); sd1=np.array([2.0,3.0,2.5,4.0,3.5]); n1=np.array([12,20,15,30,10])
    m2=np.array([11.2,11.0,9.5,17.0,13.0]); sd2=np.array([2.5,3.2,2.0,4.5,3.0]); n2=np.array([12,18,15,25,10])
    d = cohens_d(m1,sd1,n1,m2,sd2,n2); se = se_d_rmd(d,n1,n2); v = se**2
    J = hedges_J(n1+n2-2); g = J*d
    np.set_printoptions(precision=10)
    print("d", d); print("se", se); print("g", g); print("vi_LS", vi_metafor_LS(g,n1,n2))
    print("HE start", he_tau2(d,v))
    t_reml, it = reml_tau2(d, v); print("REML tau2", t_reml, "iters", it)
    print("DL", dl_tau2(d,v), "PM", pm_tau2(d,v))
    print("QP CI", qprofile_ci(d,v))
    r = pool(d, v, t_reml); print(r)
    print("V PI", pool(d,v,t_reml,pi="V")["PI"], "S PI", pool(d,v,t_reml,pi="S")["PI"])
    print("HK", pool(d, v, t_reml, hk=True))
    print("egger", egger_lm(d, se))
    y2=np.array([0.2,0.25,0.22,0.18]); v2=np.array([0.04,0.05,0.03,0.045])
    print("zero case", reml_tau2(y2,v2), pool(y2,v2,reml_tau2(y2,v2)[0])["PI"])
