import numpy as np
from util import *

# perturb a model parameter by sampling from a Gaussian distribution
def _perturb(msini, period, t0):
    m = msini[0]
    p = period[0]
    t = t0[0]
    param = np.random.randint(3)
    # perturb m*sin(i)
    if param == 0:
        m = np.random.normal(msini[0], msini[1])
        # m should be positive, change phase
        if m < 0:
            # -A*sin(x) = A*sin(x ± pi)
            m = -m
            t = t + PI*p/(2*PI*86400)
    # perturb period
    elif param == 1:
        p = np.random.normal(period[0], period[1])
    # perturb time zero
    else:
        t = np.random.normal(t0[0], t0[1])    
    return m, p, t, param

# the chi2 goodness of fit
def _chi2(m, p, t, data, mstar):
    _t = data[0]
    _rv = data[1]
    _rv_err = data[2]
    # get the RV for our model
    rv = radial_velocity(m, p * 86400, t, _t, mstar)
    # compare to the observed data to get the chi2
    return np.sum(((_rv - rv)/_rv_err)**2)

# accept the jump or not
def _accept(m, p, t, chi, data, mstar):
    prob = np.random.rand()
    chi2 = _chi2(m, p, t, data, mstar)
    # alpha is 1 going downhill (accept), e^(-Δchi2/2) uphill (probabilistic)
    np.seterr('ignore')
    alpha = min(1, np.exp(-(chi2-chi)/2))
    # if the new model is lower on the chi2 surface or passes alpha, accept it
    if prob < alpha:
        return 1, chi2
    # otherwise reject the new model
    else:
        return 0, chi

# produce a new model and choose whether to accept it
def _model(msini, period, t0, chi, data, mstar):
    m, p, t, param = _perturb(msini, period, t0)
    accept, chi = _accept(m, p, t, chi, data, mstar)
    return m, p, t, chi, accept, param

# perform the MCMC
def mcmc(burn, runs, msini, period, t0, mstar, filename):
    # read in the observed RV data
    data = read(filename)
    runs -= burn
    # the initial values for our model
    m, m_err = msini
    p, p_err = period
    t, t_err = t0
    chi = _chi2(m, p, t, data, mstar)
    # for the purpose of statistics
    models = []
    accept = []
    params = []
    # the burn-in phase
    for i in range(burn):
        m, p, t, chi2, accepted, param = \
            _model(msini, period, t0, chi, data, mstar)
        # accept the new model and update, otherwise repeat with old values
        if accepted:
            msini = (m, m_err)
            period = (p, p_err)
            t0 = (t, t_err)
            chi = chi2
    # the remaining runs
    for i in range(runs):
        m, p, t, chi2, accepted, param = \
            _model(msini, period, t0, chi, data, mstar)
        # store the results for statistics for the after-burn phase
        models.append((m, p, t))
        accept.append(accepted)
        params.append(param)
        # accept the new model and update, otherwise repeat with old values
        if accepted:
            msini = (m, m_err)
            period = (p, p_err)
            t0 = (t, t_err)
            chi = chi2
    return models, accept, params