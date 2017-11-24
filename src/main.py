import corner
import numpy as np
import matplotlib.pyplot as plt
from util import *
from mcmc import *

SHOW = [0, 0, 0, 0, 1]

# initial values for the model
mstar = 1.14 * M_SUN
# m*sin(i)
m = 0.7 * M_JUP
m_err = 0.02 * M_JUP
# orbital period
p = 3.525
p_err = 0.00005
# time zero
t = 2452854.8
t_err = 0.01
# the initial model M_0
msini = (m, m_err)
period = (p, p_err)
t0 = (t, t_err)
# burn-in and total trials for MCMC
burn = 5000
runs = 1000000

# get the MCMC data
models, accept, params = mcmc(burn, runs, msini, period, t0, mstar, 
    'HD209458_3_KECK.vels')
models2 = mcmc(burn, runs, msini, period, t0, mstar,
    'HD209458_3_KECK_transitless.vels')[0]

if SHOW[0]:
    # sort the arguments by parameter
    args = np.argsort(params)
    params_sort = np.sort(params)
    # find the regions splitting off the parameters
    loc_period = np.where(params_sort == 1)[0][0]
    loc_t0 = np.where(params_sort == 2)[0][0]
    # find the acceptances for jumps in each parameters
    accept_sort = [accept[i] for i in args]
    accept_msini = accept_sort[:loc_period]
    accept_period = accept_sort[loc_period:loc_t0]
    accept_t0 = accept_sort[loc_t0:]
    # get the acceptance rates for jumps in each parameters
    accept_msini = accept_msini.count(1) / len(accept_msini)
    accept_period = accept_period.count(1) / len(accept_period)
    accept_t0 = accept_t0.count(1) / len(accept_t0)
    print('Acceptance rate for jumps in M*sin(i):', sig_figs(accept_msini))
    print('Acceptance rate for jumps in period:', sig_figs(accept_period))
    print('Accentance rate for jumps in time zero:', sig_figs(accept_t0))

# the models' values of the parameters
msini = np.array([abs(model[0]) for model in models])
period = np.array([model[1] for model in models])
time = np.array([model[2] for model in models])
# the median values of the parameters
med_m = np.median(msini)
med_p = np.median(period)
med_t = np.median(time)
# the standard deviation of the parameters
std_m = sig_figs(np.std(msini) / M_JUP,2)
std_p = sig_figs(np.std(period))
std_t = sig_figs(np.std(time), 8)
if SHOW[1]:
    print('Median M*sin(i):', med_m / M_JUP, '±', std_m, 'M_J')
    print('Median period:', med_p, '±', std_p, 'days')
    print('Median time zero:', med_t, '±', std_t, 'days')

if SHOW[2]:
    # have confidence intervals of 1-3σ
    sig = [0.682689492137086, 0.954499736103642, 0.997300203936740]
    model = []
    for i in range(len(msini)):
        model.append((msini[i]/M_JUP, period[i]-3.5, time[i]-2452852))
    # contour plots of the posteriors with confidence intervals
    fig, axarr = plt.subplots(3,3)
    fig = corner.corner(model, fig=fig, use_math_text=1, levels=sig, 
        labels=[r'$M\cdot\sin(i)\ (\mathrm{M}_{\mathrm{J}})$', 
        r'$P\ (\mathrm{days}-3.5)$',r'$\tau_{0}\ (\mathrm{HJD}-2452852)$'], 
        tick_labelsize=8, quantiles=[0.5-sig[0]/2,0.5,0.5+sig[0]/2],)
    # plot the literature values on top
    truths = [0.69,0.02474541,2.825415]
    axes = np.array(fig.axes).reshape((len(truths), len(truths)))
    for i in range(len(truths)):
        ax = axes[i,i]
        ax.axvline(truths[i], color='b')
    for y in range(len(truths)):
        for x in range(y):
            ax = axes[y,x]
            ax.axvline(truths[x], color='b')
            ax.axhline(truths[y], color='b')
            ax.plot(truths[x], truths[y], 'sb')
    axarr[1,0].tick_params(axis='both',labelsize=8)
    axarr[2,1].tick_params(axis='both',labelsize=6)
    axarr[2,0].tick_params(axis='both',labelsize=8)
    axarr[2,2].tick_params(axis='x',labelsize=8)

# make a phased RV curve with data points (w/ error) and the best-fit solution
if SHOW[3]:
    plt.figure()
    # the observed phased RV curve
    t, rv, rv_err = read('HD209458_3_KECK.vels')
    tphase = (med_t - t) % med_p
    plt.errorbar(tphase, rv, yerr=rv_err, fmt='o')
    # the best-fit phased RV curve
    t = np.linspace(0, med_p, 50)
    rv_mcmc = radial_velocity(med_m, med_p*86400, t, 0, mstar)
    plt.plot(t, rv_mcmc)
    plt.xlabel('Phased Time (days)')
    plt.ylabel('Radial Velocity (m/s)')
    
# the MCMC with transitless data points
if SHOW[4]:
    # the models' values of the parameters
    msini = np.array([abs(model[0]) for model in models2])
    period = np.array([model[1] for model in models2])
    time = np.array([model[2] for model in models2])
    # the median values of the parameters
    med_m = np.median(msini)
    med_p = np.median(period)
    med_t = np.median(time)
    # the standard deviations of the parameters
    std_m = sig_figs(np.std(msini) / M_JUP,2)
    std_p = sig_figs(np.std(period))
    std_t = sig_figs(np.std(time), 8)
    print('Median M*sin(i):', med_m / M_JUP, '±', std_m, 'M_J')
    print('Median period:', med_p, '±', std_p, 'days')
    print('Median time zero:', med_t, '±', std_t, 'days')
    plt.figure()
    # the observed phased RV curve
    t, rv, rv_err = read('HD209458_3_KECK_transitless.vels')
    tphase = (med_t - t) % med_p
    plt.errorbar(tphase, rv, yerr=rv_err, fmt='o')
    # the best-fit phased RV curve
    t = np.linspace(0, med_p, 50)
    rv_mcmc = radial_velocity(med_m, med_p*86400, t, 0, mstar)
    plt.plot(t, rv_mcmc)
    plt.xlabel('Phased Time (days)')
    plt.ylabel('Radial Velocity (m/s)')

if SHOW[2] or SHOW[3] or SHOW[4]:
    plt.show()