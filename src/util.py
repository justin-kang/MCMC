import numpy as np
from astropy import constants

PI = np.pi
G = constants.G.value
M_SUN = constants.M_sun.value
M_JUP = constants.M_jup.value
M_EARTH = constants.M_earth.value

# returns 'num' with 'n' significant figures
def sig_figs(num, n=4):
    numstr = ("{0:.%ie}" % (n-1)).format(num)
    return float(numstr)

# read in RV data from the file
def read(filename):
    data = np.loadtxt(filename)
    # the date/time the data was recorded
    t = np.asarray(data[:,0])
    # the RV measurements
    rv = np.asarray(data[:,1])
    # the uncertainty in the RV measurements
    rv_err = np.asarray(data[:,2])
    return t, rv, rv_err

def radial_velocity(m, p, t, tref, mstar):
    # the orbital velocity of the planet
    a = (G * mstar * p**2 / (4 * PI**2))**(1/3)
    vplan = 2 * PI * a / p
    # by conservation of momentum, get the radial velocity of the star
    vsini = (m / mstar) * vplan
    # adjust vsin(i) to account for the phase
    return vsini * np.sin(2*PI*86400*(tref-t)/p)