import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from astropy import units as u

# improrting the next module directly won't work!
# needs requirements I have locally and don't have time to put online yet
# if you just want to run this script then you'll only need to import
#     dc.mges.MGEDiscrete1D
# so I recommend commenting everything else out of the __init__ and the source
import dynclust_src as dc

# this is capellari's MGE code
# import from wherwever you have it installed
from mge import mge_fit_1d

# We do an MGE fit twice:
# 1) once using Capellari's code (binned profile, maximum likelihood)
# 2) once using my MGEDiscrete1D class (discrete fit, posterior samples)

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
# takes posterior samples but we only plot the maximum a posteriori solution 
p.plot()

# fit with my code
r = r*u.kpc     # assumes all values have astropy unit attached
myMGE1D = dc.mges.MGEDiscrete1D(r, ngauss=2)
myMGE1D.fit_mge()
myMGE1D.plot()




# end
