# dynclust

Tools I devloped for dynamical clustering project AKA GCs in integrals of motion space.

Requires:
- emcee
- cjam
- Capellari's MGE fitting
- astropy

The script 'mge_example.py' does an MGE fit twice, once using Capellari's code (binned profile, maximum likelihood) and once using my MGEDiscrete1D class (discrete fit, posterior samples).
