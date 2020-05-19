import numpy as np

import astropy.units as u
from astropy.coordinates.matrix_utilities import matrix_product
from astropy.table import QTable

from scipy.optimize import fmin
from scipy.stats import multivariate_normal

from . import mycoords
from . import momcalc

from IPython.core.debugger import Tracer

def rejection_sample(f, n, n_rnd=1000, f_max=None, x_min=-1, x_max=1, args={}):

	n = int(n)
	n_rnd = int(n_rnd)

	# find fmax if not given
	if f_max is None:
		tmp = fmin(lambda x: -1.*f(x, **args), 0, disp=False)
		f_max = 1.01*f(tmp[0])

	# rejection sample
	samples = np.array([])
	while np.size(samples)<n:
		x_rnd = np.random.uniform(x_min, x_max, n_rnd)
		f_rnd = np.random.uniform(0, f_max, n_rnd)
		idx = np.where(f(x_rnd) > f_rnd)
		if idx[0].size > 0:
			samples = np.concatenate([samples, x_rnd[idx]])

	samples = samples[0:n]	# trim excess

	return samples

def sample_distance(gc, tracer, nsmp=100, rmax=300*u.kpc, ret_RZphi=True):

	def f(z):
		x = np.zeros(z.shape) * u.arcsec + gc['x']
		y = np.zeros(z.shape) * u.arcsec + gc['y']
		rho = tracer.density(x, y, z*u.kpc)
		return rho.to(u.pc**-3).value

	x_pc = (gc['x']*tracer.D/u.rad).to(u.pc)
	y_pc = (gc['y']*tracer.D/u.rad).to(u.pc)
	z_max = np.sqrt(rmax**2. - x_pc**2 - y_pc**2).to(u.kpc)
	z_min = - z_max

	if np.all(tracer.Q == tracer.Q[0]):

		Q = tracer.Q[0]
		ci = np.cos(tracer.inc)
		si = np.sin(tracer.inc)
		z0 = ci * si * (Q**2.-1.) / (Q**2.*si**2.+ci**2.) * y_pc
		z0 = np.array([z0.to(u.kpc).value])
		f0 = f(z0)
		z = rejection_sample(f, nsmp,
						x_min=z_min.value,
						x_max=z_max.value,
						f_max=1.01*f0[0],
						n_rnd=10*nsmp)
	else:
		z = rejection_sample(f, nsmp,
						x_min=z_min.value,
						x_max=z_max.value,
						n_rnd=10*nsmp)
	z *= u.kpc

	if ret_RZphi==True:
		tmp = np.zeros(z.shape)
		X, Y, Z = mycoords.projected_to_aligned3D(
		            tmp + (gc['x']/u.rad*tracer.D).to(u.pc),
		            tmp + (gc['y']/u.rad*tracer.D).to(u.pc),
		            z,
		            tracer.inc
					)
		R = np.sqrt(X**2 + Y**2)
		phi = np.arctan2(Y, X)
		return R.to(u.kpc), Z.to(u.kpc), phi.to(u.rad), z
	else:
		return z

def sample_velocity(tracer, moments, nsmp):

	sig_vR = np.sqrt(moments['v2rr']).to(u.km/u.s).value
	sig_vphi = np.sqrt(moments['v2ff']).to(u.km/u.s).value
	sig_vz = np.sqrt(moments['v2zz']).to(u.km/u.s).value
	mu_vphi = tracer.kappa * np.sqrt(moments['v2rr'] - moments['v2ff'])
	mu_vphi = mu_vphi.to(u.km/u.s).value

	# get means and covariance
	mu = np.array([0., 0., mu_vphi])
	cov = np.array([
		[sig_vR**2, 0., 0.],
		[0., sig_vz**2, 0.],
		[0., 0., sig_vphi**2],
		])

	# sample
	mn = multivariate_normal(mean=mu, cov=cov, allow_singular=True)
	tmp = mn.rvs(nsmp)
	if nsmp==1:
		tmp = tmp[np.newaxis, :]

	# return table
	smp = QTable()
	smp['vR'] = tmp[:,0] * u.km/u.s
	smp['vZ'] = tmp[:,1] * u.km/u.s
	smp['vf'] = tmp[:,2] * u.km/u.s
	return smp

def sample_velocity_given_vLOS(gc, tracer, moments, phi, nsmp):

	inc = tracer.inc
	mu_vzp = gc['V_los'].to(u.km/u.s).value
	sig_vzp = gc['dV_los'].to(u.km/u.s).value
	sig_vR = np.sqrt(moments['v2rr']).to(u.km/u.s).value
	sig_vphi = np.sqrt(moments['v2ff']).to(u.km/u.s).value
	sig_vz = np.sqrt(moments['v2zz']).to(u.km/u.s).value
	mu_vphi = tracer.kappa * np.sqrt(moments['v2rr'] - moments['v2ff'])
	if mu_vphi!=mu_vphi:
		mu_vphi = tracer.kappa * np.sqrt(moments['v2ff'] - moments['v2rr'])
	mu_vphi = mu_vphi.to(u.km/u.s).value

	si = np.sin(inc)
	s2i = si**2.
	ci = np.cos(inc)
	c2i = ci**2.

	sf = np.sin(phi)
	s2f = sf**2.
	cf = np.cos(phi)
	c2f = cf**2.

	# sighat = [sighat_R, sighat_z, sighat_phi]
	sighat = np.zeros(3)
	sighat[0] = (s2i*s2f*sig_vzp**-2.+sig_vR**-2.)**-0.5
	sighat[1] = (c2i*sig_vzp**-2.+sig_vz**-2.)**-0.5
	sighat[2] = (s2i*c2f*sig_vzp**-2.+sig_vphi**-2.)**-0.5

	# get covariance
	pre = np.zeros((3,3))
	for i in range(3): pre[i,i] = sighat[i]**-2.
	pre[0,1] = si*ci*sf*sig_vzp**-2.
	pre[0,2] = s2i*sf*cf*sig_vzp**-2.
	pre[1,2] = si*ci*cf*sig_vzp**-2.
	pre[1,0] = pre[0,1]
	pre[2,0] = pre[0,2]
	pre[2,1] = pre[1,2]
	cov = np.linalg.inv(pre)

	# get means
	vec = np.array([
					mu_vzp*si*sf*sig_vzp**-2.,
					mu_vzp*ci*sig_vzp**-2.,
					mu_vzp*si*cf*sig_vzp**-2. + 2.*mu_vphi*sig_vphi**-2.
					])
	muhat = np.dot(cov, vec)

	mn = multivariate_normal(mean=muhat, cov=cov, allow_singular=True)
	tmp = mn.rvs(nsmp)
	smp = QTable()
	smp['vR'] = tmp[:,0] * u.km/u.s
	smp['vZ'] = tmp[:,1] * u.km/u.s
	smp['vf'] = tmp[:,2] * u.km/u.s

	return smp

def sample_E_Lz(gc,
			potential,
			tracer,
			n_dsmp=100,
			rmax=300*u.kpc,
			n_vsmp=1000,
			momint=None,
			return_all=False,
			scale_sigma=1.):

	# take 10% extra distance samples to cover those lost to v_phi^2 < 0
	n_dsmp_trial = int(1.1*n_dsmp)
	R, Z, phi, z = sample_distance(gc, tracer, nsmp=n_dsmp_trial, rmax=rmax)
	pot = potential.potential(R, Z)
	vv = momcalc.velocity_moments(R, Z, tracer, potential, momint=momint)
	for c in vv.colnames:
	    vv[c] *= scale_sigma**2.

	n_smp = n_dsmp*n_vsmp
	Erun = np.zeros(n_smp)
	LZrun = np.zeros(n_smp)
	if return_all:
		M = mycoords.rotate(tracer.inc)
		zrun = np.zeros(n_smp)
		Rrun = np.zeros(n_smp)
		Zrun = np.zeros(n_smp)
		phirun = np.zeros(n_smp)
		potrun = np.zeros(n_smp)
		vxrun = np.zeros(n_smp)
		vyrun = np.zeros(n_smp)
		vzrun = np.zeros(n_smp)
		vRrun = np.zeros(n_smp)
		vZrun = np.zeros(n_smp)
		vfrun = np.zeros(n_smp)

	idx_dsmp = 0
	cnt = 0
	while cnt < n_dsmp:

		vv0 = vv[idx_dsmp]

		if (vv0['v2ff'] > 0.*(u.km/u.s)**2):

			sliceObj = slice(n_vsmp*cnt, n_vsmp*cnt+n_vsmp)

			z0 = z[idx_dsmp]
			R0, Z0, phi0 = R[idx_dsmp], Z[idx_dsmp], phi[idx_dsmp]
			pot0 = pot[idx_dsmp]
			v0 = sample_velocity_given_vLOS(gc, tracer, vv0, phi0, n_vsmp)

			tmp = pot0 + 0.5*(v0['vR']**2. + v0['vf']**2. + v0['vZ']**2.)
			tmp = tmp.to((u.km/u.s)**2).value
			Erun[sliceObj] = tmp

			tmp = R0 * v0['vf']
			tmp = tmp.to(u.kpc*u.km/u.s).value
			LZrun[sliceObj] = tmp

			if return_all:

				tmp = np.repeat(pot0.to((u.km/u.s)**2).value, n_vsmp)
				potrun[sliceObj] = tmp

				tmp = np.repeat(z0.to(u.kpc).value, n_vsmp)
				zrun[sliceObj] = tmp

				tmp = mycoords.cylin_to_aligned3D_vels(v0['vR'], v0['vf'], v0['vZ'], phi0)
				vX, vY, vZ = tmp
				vxyz = matrix_product(M, tmp)
				vxyz = vxyz.to(u.km/u.s).value
				tmp = vxyz[0,:]
				vxrun[sliceObj] = tmp
				tmp = vxyz[1,:]
				vyrun[sliceObj] = tmp
				tmp = vxyz[2,:]
				vzrun[sliceObj] = tmp

				tmp = np.repeat(R0.to(u.kpc).value, n_vsmp)
				Rrun[sliceObj] = tmp
				tmp = np.repeat(Z0.to(u.kpc).value, n_vsmp)
				Zrun[sliceObj] = tmp
				tmp = np.repeat(phi0.to(u.deg).value, n_vsmp)
				phirun[sliceObj] = tmp

				tmp = v0['vR'].to(u.km/u.s).value
				vRrun[sliceObj] = tmp
				tmp = v0['vZ'].to(u.km/u.s).value
				vZrun[sliceObj] = tmp
				tmp = v0['vf'].to(u.km/u.s).value
				vfrun[sliceObj] = tmp

			cnt += 1

		idx_dsmp += 1

	smp = QTable()
	smp['E'] = Erun * (u.km/u.s)**2
	smp['Lz'] = LZrun * u.kpc*u.km/u.s
	if return_all:
		smp['z'] = zrun * u.kpc
		smp['R'] = Rrun * u.kpc
		smp['Z'] = Zrun * u.kpc
		smp['phi'] = phirun * u.deg
		smp['pot'] = potrun * (u.km/u.s)**2
		smp['vx'] = vxrun * u.km/u.s
		smp['vy'] = vyrun * u.km/u.s
		smp['vz'] = vzrun * u.km/u.s
		smp['vR'] = vRrun * u.km/u.s
		smp['vZ'] = vZrun * u.km/u.s
		smp['vf'] = vfrun * u.km/u.s

	X, Y, Z, vX, vY, vZ = mycoords.cylin_to_aligned3D(smp['R'],
													  smp['phi'],
													  smp['Z'],
													  smp['vR'],
													  smp['vf'],
													  smp['vZ'])
	L = np.cross([X, Y, Z],
	             [vX, vY, vZ],
	             axisa=0,
	             axisb=0)
	Lz = L[:, 2] * u.kpc* u.km/u.s
	modL = np.linalg.norm(L, axis=1) * u.kpc* u.km/u.s
	smp['Lz2'] = Lz
	smp['modL'] = modL

	return QTable(smp)
