import numpy as np

from astropy.table import Table, QTable
import astropy.units as u

from . import mges, halos

def get_tracer_mge(filename,
	incl=90*u.deg,
	distance=1*u.Mpc,
	beta=0,
	kappa=0
	):

	tmp = Table.read(
		filename,
	    format='ascii',
	    names=['n', 'i', 's', 'q']
	    )
	mge = mges.TracerMGE(
	    (tmp['i'] * u.arcmin**-2 * u.rad**2 * distance**-2).to(u.pc**-2),
	    tmp['s'] * u.arcsec,
	    tmp['q']
	    )

	return mge

def get_mcmc_filename(halo):

	if halo=='cored':
	    smp_file = 'MCMC_chain_cored/sample_2016-03-24_13-18-24.dat'
	if halo=='cusped':
	    smp_file = 'MCMC_chain_cusped/sample_2016-03-24_03-03-21.dat'

	return smp_file

def get_mcmc_samples(filename):

	smp = Table.read(filename,
	    format='ascii',
	    names=['d1',
	        'd2',
	        'lb_blue',
	        'c0_blue',
	        'sc_blue',
	        'lb_red',
	        'c0_red',
	        'sc_red',
	        'kappa_blue',
	        'kappa_red',
	        'lb_PN',
	        'kappa_PN',
	        'M/L',
	        'logL'
	        ]
	    )
	smp['rho_s'] = 10.**smp['d2'] * u.Msun/u.pc**3
	smp['r_s'] = 10.**((smp['d1']-2.*smp['d2'])/3.) * u.pc
	smp['M/L'] *= u.Msun/u.Lsun
	for col in ['red', 'blue']:
	    smp['c0_{0}'.format(col)] *= u.mag
	    smp['sc_{0}'.format(col)] *= u.mag
	for col in ['red', 'blue', 'PN']:
	    smp['beta_z_{0}'.format(col)] = 1. - np.exp(-smp['lb_{0}'.format(col)])
	smp = QTable(smp)   # use Qtables instead of table

	return smp

def get_dm_halo(halo, rho_s, r_s):

	if halo=='cored':
	    dm_halo = halos.cored_DM(rho_s, r_s)
	if halo=='cusped':
	    dm_halo = halos.cusped_DM(rho_s, r_s)

	return dm_halo

def get_stellar_mge(filename):

	tmp = Table.read(
	    filename,
	    format='ascii',
	    names=['n', 'i', 's', 'q']
	    )
	stellar_mge = mges.StellarMGE(
	    tmp['i'] * u.Lsun/u.pc**2,
	    tmp['s'] * u.arcsec,
	    tmp['q']
	    )

	return stellar_mge

def get_gc_data(filename):

	gcs = Table.read(filename,
	    format='ascii'
	    )
	gcs['x'] *= u.arcsec
	gcs['y'] *= u.arcsec
	gcs['V_los'] *= u.km/u.s
	gcs['dV_los'] *= u.km/u.s
	gcs['g-i'] *= u.mag
	gcs['dg-i'] *= u.mag
	gcs = QTable(gcs)
	n_gcs = len(gcs)
	gcs['r'] = np.sqrt(gcs['x']**2. + gcs['y']**2)

	return gcs

def string_to_list(cstat, key_list=['grp_siz', 'grp_cnt', 'labs']):
    for key in key_list:
        for cs0 in cstat:
            s = cs0[key]
            s = s.replace('[', '')
            s = s.replace(']', '')
            s = s.split(',')
            s = [int(s0) for s0 in s]
            cs0[key] = s
    return cstat

def string_to_list_o_lists(a, key_list=['grp_siz', 'grp_cnt', 'labs']):
    for key in key_list:
        for i, x in enumerate(a[key]):
            x = x.split('],')
            x = [x0.replace('[', '') for x0 in x]
            x = [x0.replace(']', '') for x0 in x]
            x = [[int(x00) for x00 in x0.split(',')] for x0 in x]
            a[key][i] = x
    return a
