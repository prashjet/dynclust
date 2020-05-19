import numpy as np
from mge import mge_fit_1d      # i.e. capellari's MGE code
import matplotlib.pyplot as plt
from scipy import stats

from astropy import units as u
import dynclust_src as dc

# generate some fake 2D data
n1 = 100
n2 = 300
sig1 = 5
sig2 = 20
q = 0.8
x_g1 = stats.norm(loc=0, scale=sig1)
x_g2 = stats.norm(loc=0, scale=sig2)
y_g1 = stats.norm(loc=0, scale=q*sig1)
y_g2 = stats.norm(loc=0, scale=q*sig2)
x = np.concatenate((x_g1.rvs(n1), x_g2.rvs(n2)))
y = np.concatenate((y_g1.rvs(n1), y_g2.rvs(n2)))
plt.scatter(x, y)
plt.show()

# find observed q value and convert to elliptical r
q_obs = np.std(y)/np.std(x)
r = np.sqrt(x**2 + (y/q_obs)**2)

# fit 1D profile with capellari's code
# ... (i) make logarithmically binned density profile
logr = np.log10(r)
Nbins = 8
h, loge, _ = plt.hist(logr, bins=Nbins) # must pick Nbins so no bin has 0 counts
e = 10.**loge
area = np.pi * e**2.
area_ring = area[1:] - area[0:-1]
rho = h/area_ring
c = np.sqrt(e[0:-1] * e[1:])
p = mge_fit_1d.mge_fit_1d(c, rho, ngauss=10)
p.plot()

# fit with my code
r = r*u.kpc     # assumes all values have astropy unit attached
myMGE1D = dc.mges.MGEDiscrete1D(r, ngauss=2)
myMGE1D.fit_mge()
myMGE1D.plot()




# end
