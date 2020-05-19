import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table, QTable
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline
from scipy import integrate
from scipy import optimize

import cjam
from jam import mge_vcirc

import emcee

from . import mycoords

class MGE:

	def __init__(self, I, sigma, q):

		self.I = I				    # intensity e.g. L/pc^2, kpc^-2, M/m^2
		self.sigma = sigma			# angular sigma
		self.q = q				    # flattenings
		self.N = len(self.I)

	def surface_brightness(self, x, y):

		# reshape inputs for array operations
		if np.ndim(x)>0:
			x = x.T
			x = np.repeat(x[np.newaxis,:], self.N, axis=0)
			x = x.T
		if np.ndim(y)>0:
			y = y.T
			y = np.repeat(y[np.newaxis,:], self.N, axis=0)
			y = y.T

		r2 = x**2. + (y/self.q)**2.
		Sig_i = self.I * np.exp(-r2/2./self.sigma**2.)
		Sig = np.sum(Sig_i, axis=-1)

		return Sig

	def deproject(self, inc):

		P = np.ones(self.N)
		Q = self.q**2.-np.cos(inc)**2.
		if np.any(Q <= 0.0):
			raise ValueError('Inclination too low for deprojection')
		Q = np.sqrt(Q)/np.sin(inc)
		if np.any(np.abs(Q) <= 0.05):
			raise ValueError('Q < 0.05 components')

		self.inc = inc
		self.Q = Q
		self.P = P

	def distance_scale(self, D):

		self.D = D
		self.sigma_pc = mycoords.arcsec_to_pc(self.sigma, D)
		self.L = self.I * 2.*np.pi * self.sigma_pc**2. * self.q

	def density(self, x, y, z_pc):

		X, Y, Z = mycoords.deproject_coords(x, y, z_pc, self.D, self.inc)

		# reshape inputs for array operations
		if np.ndim(X)>0:
			X = X.T
			X = np.repeat(X[np.newaxis,:], self.N, axis=0)
			X = X.T
		if np.ndim(Y)>0:
			Y = Y.T
			Y = np.repeat(Y[np.newaxis,:], self.N, axis=0)
			Y = Y.T
		if np.ndim(Z)>0:
			Z = Z.T
			Z = np.repeat(Z[np.newaxis,:], self.N, axis=0)
			Z = Z.T

		r2 = X**2. + (Y/self.P)**2. + (Z/self.Q)**2.
		rho_0 = self.L * (self.sigma_pc**2.*2.*np.pi)**-1.5 /(self.P*self.Q)
		rho_i = rho_0 * np.exp(-r2/2./self.sigma_pc**2.)
		rho = np.sum(rho_i, axis=-1)
		return rho

	def density_RZ(self, R, Z):

		# reshape inputs for array operations
		if np.ndim(R)>0:
			R = R.T
			R = np.repeat(R[np.newaxis,:], self.N, axis=0)
			R = R.T
		if np.ndim(Z)>0:
			Z = Z.T
			Z = np.repeat(Z[np.newaxis,:], self.N, axis=0)
			Z = Z.T
		# calculate
		r2 = R**2. + (Z/self.Q)**2.
		rho_0 = self.L * (self.sigma_pc**2.*2.*np.pi)**-1.5 /(self.P*self.Q)
		rho_i = rho_0 * np.exp(-r2/2./self.sigma_pc**2.)
		rho = np.sum(rho_i, axis=-1)
		return rho

	def ddensity_dR_RZ(self, R, Z):

		# reshape inputs for array operations
		if np.ndim(R)>0:
			R = R.T
			R = np.repeat(R[np.newaxis,:], self.N, axis=0)
			R = R.T
		return -R/self.sigma_pc**2 * self.density_RZ(R, Z)

class TracerMGE(MGE):

	def set_anisotropy(self, beta):
		self.beta = beta

	def set_rotation(self, kappa):
		self.kappa = kappa

class HaloMGE(MGE):

	def potential(self, x, y, z_pc, kwargs={}):

		X, Y, Z = mycoords.deproject_coords(x, y, z_pc, D=self.D, inc=self.inc)
		R = np.sqrt(X**2 + Y**2)
		mge_table = Table()
		mge_table['i'] = (self.I * u.Lsun/u.Msun).to(u.Lsun/u.pc**2)
		mge_table['s'] = self.sigma
		mge_table['q'] = self.q
		pot = cjam.axisymmetric_pot(
			R,
			Z,
		    mge_table,
		    self.D,
		    incl = self.inc,
		    mscale =  1.*u.Msun/u.Lsun,
			**kwargs
		    )

		return pot

class StellarMGE(MGE):

	def set_m2l(self, m2l):

		self.m2l = m2l

	def mass_density(self, x, y, z_pc):

		return self.m2l * self.density(x, y, z_pc)

	def potential(self, x, y, z_pc, kwargs={}):

		X, Y, Z = mycoords.deproject_coords(x, y, z_pc, D=self.D, inc=self.inc)
		R = np.sqrt(X**2 + Y**2)
		mge_table = Table()
		mge_table['i'] = self.I
		mge_table['s'] = self.sigma
		mge_table['q'] = self.q
		pot = cjam.axisymmetric_pot(
			R,
			Z,
		    mge_table,
		    self.D,
		    incl=self.inc,
		    mscale=self.m2l,
			**kwargs
		    )

		return pot

class CombinedMGE(MGE):

	def __init__(self, stellar_mge, dm_halo):

		# concatenate stellar and dark mge arrays
		i = np.concatenate((
		    stellar_mge.I.to(u.Lsun/u.pc**2),
		    (dm_halo.mge.I*u.Lsun/u.Msun).to(u.Lsun/u.pc**2),
		    ))
		i = i.value * u.Lsun/u.pc**2
		s = np.concatenate((
		    stellar_mge.sigma.to(u.arcsec),
		    (dm_halo.mge.sigma).to(u.arcsec),
		    ))
		s = s.value * u.arcsec
		q = np.concatenate((stellar_mge.q, dm_halo.mge.q))
		m2l = stellar_mge.m2l.to(u.Msun/u.Lsun).value
		m2l = np.concatenate((
			np.ones(stellar_mge.N) * m2l,
			np.ones(dm_halo.mge.N)
			))
		m2l *= u.Msun/u.Lsun

		check = (stellar_mge.D==dm_halo.mge.D)
		assert check, "MGEs at different distances"
		check = (stellar_mge.inc==dm_halo.mge.inc)
		assert check, "MGEs at different inclinations"

		self.I = i
		self.sigma = s
		self.q = q
		self.m2l = m2l
		self.N = len(self.I)
		self.D = stellar_mge.D
		self.inc = stellar_mge.inc

	def potential(self,
				R,
				Z,
				kwargs={'nrad':1000, 'nang':1000},
				use_potint=False):

		if use_potint is False:
			pot_tab = Table()
			pot_tab['i'] = self.I
			pot_tab['s'] = self.sigma
			pot_tab['q'] = self.q
			pot_tab = QTable(pot_tab)
			pot = cjam.axisymmetric_pot(
				R,
				Z,
			    pot_tab,
			    self.D,
			    incl=self.inc,
			    mscale=self.m2l,
				**kwargs
			    )

		if (use_potint is True) and (hasattr(self, 'potint') is True):
			pot = self.potint(R, Z)

		if (use_potint is True) and (hasattr(self, 'potint') is False):
			self.get_potential_interp_func()
			pot = self.potint(R, Z)

		return pot.to((u.km/u.s)**2)

	def get_potential_interp_func(self,
								maxr=300*u.kpc,
								nrad=50,
								nang=30,
								frac_log=0.2,
								kw_interp={}):

		rmax = maxr.to(u.kpc).value

		# logarithmically space a fraction of the points to sample inner region
		lr1 = np.linspace(-3, np.log10(rmax), int(frac_log*nrad))
		r1 = 10.**lr1

		# linearly space the rest of the points
		r2 = np.linspace(0, rmax, int((1.-frac_log)*nrad))

		# sort r array
		rell = np.concatenate((r1, r2))
		rell = np.sort(rell)

		# make 2D grid
		ang = np.linspace(0., np.pi/2., nang) * u.rad
		R = np.outer(rell, np.cos(ang))
		Z = np.outer(rell, np.sin(ang))

		# get potential on grid
		potgrid = self.potential(R.ravel() * u.kpc, Z.ravel() * u.kpc)
		potgrid = np.reshape(potgrid, (rell.size, ang.size))

		# get interpolation function
		pot_interp = RectBivariateSpline(rell, ang, potgrid, **kw_interp)

		# function to evaluate interpolator
		def potint(R, Z):
			rell = np.sqrt(R**2 + Z**2)
			rell = rell.to(u.kpc).value
			ang = np.zeros(len(R)) * u.rad
			ang[R!=0*u.kpc] = np.abs(np.arctan(Z[R!=0*u.kpc]/R[R!=0*u.kpc]))
			ang[R==0*u.kpc] = np.pi/4. * u.rad
			ang = ang.to(u.rad).value
			pot = pot_interp(rell, ang, grid=False) * (u.km/u.s)**2
			return pot

		# store function to evaluate interpolator
		self.potint = potint

	def vcirc(self, R, dR=0.0001*u.pc, usejam=False):

		if usejam:
			vcirc = mge_vcirc.mge_vcirc(
			  	self.I.value,
			    (self.sigma).to(u.arcsec).value,
			    self.q,
			    self.inc.to(u.deg).value,
			    0.,
			    (self.D).to(u.Mpc).value,
				mycoords.pc_to_arcsec(R.to(u.pc), D).value,
				soft=1.
			    )
			vcirc *= u.km/u.s
		else:
			pot1 = self.potential(R, 0.*R)
			pot2 = self.potential(R+dR, 0.*R)
			dphi_dR = (pot2 - pot1)/dR
			vcirc = np.sqrt(R * dphi_dR)

		return vcirc.to(u.km/u.s)

	def get_E_Lz_boundary_interp_func(self, maxr=300*u.kpc, nr=300):

		R = np.linspace(0.*u.pc, maxr, nr)
		Z = 0.*R
		pot = self.potential(R, Z)
		vcirc = self.vcirc(R)
		E = pot + 0.5 * vcirc**2.
		Lz = R * vcirc
		E = E.to((u.km/u.s)**2).value/1e5
		Lz = Lz.to(u.kpc*u.km/u.s).value/1e4
		Lz_max_interp = interp1d(E,
						Lz,
						kind='cubic',
						fill_value=(Lz[0], Lz[-1]),
						bounds_error=False)
		self.Lz_max_interp = Lz_max_interp

	def Lz_max(self, E):

		if hasattr(self, 'Lz_max_interp') is False:
			self.get_E_Lz_boundary_interp_func()
		tmp = self.Lz_max_interp(E.to((u.km/u.s)**2).value/1e5)
		Lz_max = tmp * 1e4 * u.kpc*u.km/u.s

		return Lz_max

class MGEDiscrete1D:

	def __init__(self, re, ngauss):
		self.ngauss = ngauss
		self.re = re

	def logprior(self, p):
		# unpack params
		N = self.ngauss
		sig = p[0:N]
		I = p[N::]
		I = np.concatenate((I, [1.-np.sum(I)]))
		if np.min(I)<0:
			return -np.inf
		elif np.min(sig)<0:
			return -np.inf
		elif np.all(sig==np.sort(sig))==False:
			return -np.inf
		else:
			return 0.

	def logL(self, p, r, return_arr=False):
		# unpack params
		N = self.ngauss
		sig = np.array([p[0:N]]).T
		I = p[N::]
		I = np.concatenate((I, [1.-np.sum(I)]))
		I = np.array([I]).T
		gauss = I/(2*np.pi*sig**2.) * np.exp(-0.5*(r/sig)**2.)
		if return_arr:
			return np.log(np.sum(gauss, axis=0))
		return np.sum(np.log(np.sum(gauss, axis=0)))

	def logprob(self, p, r):
		logpri = self.logprior(p)
		if logpri==-np.inf:
			return logpri
		else:
			return logpri + self.logL(p, r)

	def fit_mge(self, nwalkers=100, nsteps=1000):
		self.nwalkers = nwalkers
		# get starting positions
		sig0 = np.random.uniform(np.min(self.re.value),
		                         np.max(self.re.value),
		                         (self.nwalkers, self.ngauss))
		sig0 = np.sort(sig0, 1)
		f0 = np.random.dirichlet(np.ones(self.ngauss), 100)
		f0 = f0[:, 0:-1]
		p0 = np.hstack((sig0, f0))
		# run emcee
		ndim = 2*self.ngauss-1
		sampler = emcee.EnsembleSampler(self.nwalkers,
										ndim,
										self.logprob,
										args=[self.re.value])
		sampler.run_mcmc(p0, nsteps)
		# get maxL pars
		idx = np.argmax(sampler.lnprobability)
		idx = np.unravel_index(idx, sampler.lnprobability.shape)
		pmaxprb = sampler.chain[idx[0], idx[1], :]
		self.maxLpars = pmaxprb
		# split pars
		self.sig = pmaxprb[0:self.ngauss]
		f = pmaxprb[self.ngauss::]
		f = np.concatenate((f, [1-np.sum(f)]))
		nobs = len(self.re)
		f *= nobs
		self.f = f

	def nu(self, r, pars):
		# unpack params
		N = self.ngauss
		sig = np.array([pars[0:N]]).T
		I = pars[N::]
		I = np.concatenate((I, [1.-np.sum(I)]))
		I = np.array([I]).T
		gauss = I/(2*np.pi*sig**2.) * np.exp(-0.5*(r/sig)**2.)
		return np.sum(gauss, axis=0)

	def plot(self, bins=15):
		h, r = np.histogram(np.log10(self.re.value), bins=bins)
		r = 10.**r
		c = (r[0:-1]  + r[1::])/2.
		A = np.pi * r**2.
		A = A[1::] - A[0:-1]
		rho_bin = h/A
		xx = np.linspace(np.min(c), np.max(c), 100)
		nobs = len(self.re)
		fig, ax = plt.subplots(1, 1)
		rho = nobs * self.nu(xx, self.maxLpars)
		ax.loglog(xx, rho, '-b')
		ax.loglog(c, rho_bin, '-or')
		plt.show()
		return

	def integrand(self, r, pars):
		return 2. * np.pi * r * self.nu(r, pars)

	def cumulative(self, r, pars):
		return integrate.quad(self.integrand, 0, r, args=pars)[0]

	def rescale_constant_density(self, r, pars):
		return np.sqrt(self.cumulative(r, pars))

	def get_percentiles(self, pc=[0.68, 0.95, 0.99]):

		def tmp(r, pars, pc):
			return self.cumulative(r, pars) - pc

		# ensure max_r is large enough
		k = 1.
		max_r = k*np.max(self.re.value)
		while self.cumulative(max_r, self.maxLpars) < np.max(pc):
			k += 1.
			max_r = k*np.max(self.re.value)

		r_pc = np.zeros_like(pc)
		for i, pc0 in enumerate(pc):
			r_pc[i] = optimize.brentq(tmp, 0, max_r, args=(self.maxLpars, pc0))

		self.pc=pc
		self.r_pc=r_pc

# end
