import numpy as np
from astropy.table import Table, QTable
import astropy.units as u

from scipy.interpolate import RectBivariateSpline
from scipy.integrate import quad

import cjam

def velocity_moments(R, Z, tracer, potential, nrad=200, nang=50, momint=None):

	if momint is None:

		# put mges into table format
		tracer_tab = Table()
		tracer_tab['i'] = tracer.I*u.Lsun
		tracer_tab['s'] = tracer.sigma
		tracer_tab['q'] = tracer.q
		tracer_tab = QTable(tracer_tab)
		pot_tab = Table()
		pot_tab['i'] = potential.I
		pot_tab['s'] = potential.sigma
		pot_tab['q'] = potential.q
		pot_tab = QTable(pot_tab)

		# call cjam
		moments = cjam.axisymmetric_cylin(
			R,
			Z,
			tracer_tab,
			pot_tab,
			tracer.D,
			beta=tracer.beta,
			kappa=tracer.kappa,
	    	mscale=potential.m2l,
			incl=tracer.inc,
			nrad=nrad,
			nang=nang
			)

		# fix negative second moments...
		fill = 0.#np.min(moments['v2ff'][moments['v2ff'] > 0.])
		moments['v2ff'][moments['v2ff'] < 0.] = fill

	else:

		moments = momint.evaluate(R, Z)

	return moments

class moment_interpolator(object):

	def __init__(self,
				tracer,
				potential,
				nrad=50,
				nang=20,
				rmax=300*u.kpc,
				kw_interp={},
				frac_log=0.2
				):

		self.tracer = tracer
		self.potential = potential
		self.nrad = nrad
		self.nang = nang
		self.rmax = rmax

		rmax = self.rmax.to(u.kpc).value
		# logarithmically space a fraction of the points to sample inner region
		lr1 = np.linspace(-3, np.log10(rmax), int(frac_log*self.nrad))
		r1 = 10.**lr1
		# linearly space the rest of the points
		r2 = np.linspace(0, rmax, int((1.-frac_log)*self.nrad))
		# sort r array
		rell = np.concatenate((r1, r2))
		rell = np.sort(np.unique(rell))
		# make 2D grid
		ang = np.linspace(0., np.pi/2., self.nang) * u.rad
		R = np.outer(rell, np.cos(ang))
		Z = np.outer(rell, np.sin(ang))
		# get moments on grid
		moments = velocity_moments(R.ravel() * u.kpc,
								Z.ravel() * u.kpc,
								self.tracer,
								self.potential,
								nrad=1e5,
								nang=1e5)
		self.R = R.ravel() * u.kpc
		self.Z = Z.ravel() * u.kpc
		# where v2ff<=0, replace it with smallest non-zero value
		fill = np.min(moments['v2ff'][moments['v2ff'] > 0.])
		moments['v2ff'][moments['v2ff'] <= 0.] = fill
		self.moments = moments

		# interpolate logs then exponentiaite to ensure positivity
		tmp = np.reshape(moments['v2rr'], (rell.size, ang.size))
		tmp = np.log(tmp.to((u.km/u.s)**2).value)
		self.v2rr = RectBivariateSpline(rell, ang, tmp, **kw_interp)
		tmp = np.reshape(moments['v2ff'], (rell.size, ang.size))
		tmp = np.log(tmp.to((u.km/u.s)**2).value)
		self.v2ff = RectBivariateSpline(rell, ang, tmp, **kw_interp)
		tmp = np.reshape(moments['v2zz'], (rell.size, ang.size))
		tmp = np.log(tmp.to((u.km/u.s)**2).value)
		self.v2zz = RectBivariateSpline(rell, ang, tmp, **kw_interp)

	def evaluate(self, R, Z):

		rell = np.sqrt(R**2 + Z**2)
		rell = rell.to(u.kpc).value
		ang = np.zeros(len(R)) * u.rad
		ang[R!=0*u.kpc] = np.abs(np.arctan(Z[R!=0*u.kpc]/R[R!=0*u.kpc]))
		ang[R==0*u.kpc] = np.pi/4. * u.rad
		ang = ang.to(u.rad).value
		moments = QTable()
		moments['v2rr'] = np.exp(self.v2rr(rell, ang, grid=False))
		moments['v2ff'] = np.exp(self.v2ff(rell, ang, grid=False))
		moments['v2zz'] = np.exp(self.v2zz(rell, ang, grid=False))
		for col in ['v2rr', 'v2ff', 'v2zz']:
			moments[col] *= (u.km/u.s)**2

		return moments

def integrated_v_los_moment(xp, yp, tracer, potential, nrad=200, nang=50):

	# put mges into table format
	tracer_tab = Table()
	tracer_tab['i'] = tracer.I*u.Lsun
	tracer_tab['s'] = tracer.sigma
	tracer_tab['q'] = tracer.q
	tracer_tab = QTable(tracer_tab)
	pot_tab = Table()
	pot_tab['i'] = potential.I
	pot_tab['s'] = potential.sigma
	pot_tab['q'] = potential.q
	pot_tab = QTable(pot_tab)

	# call cjam
	moments = cjam.axisymmetric_los(
		xp,
		yp,
		tracer_tab,
		pot_tab,
		tracer.D,
		beta=tracer.beta,
		kappa=tracer.kappa,
    	mscale=potential.m2l,
		incl=tracer.inc,
		nrad=nrad,
		nang=nang
		)

	return moments

def velocity_moments_direct(R, z, tracer, dm_halo):

	b = 1./(1.-tracer.beta)
	def dphi_dR(R,z):
		r = np.sqrt(R**2+z**2)
		return R/r * dm_halo.dphi_dr(r)

	def dphi_dz(R,z):
		r = np.sqrt(R**2+z**2)
		return z/r * dm_halo.dphi_dr(r)

	def d2phi_dRdz(R,z):
		r = np.sqrt(R**2+z**2)
		return R*z/r**2. * (dm_halo.d2phi_dr2(r) - dm_halo.dphi_dr(r)/r)

	def int_nuv2zz(z, R, z_unit):
		z = z * z_unit
		f = tracer.density_RZ(R, z) * dphi_dz(R,z)
		return f.to(u.pc**-3 * (u.km/u.s)**2 /u.kpc).value

	def int_nuv2ff1(z, R, z_unit):
		z = z * z_unit
		f = tracer.ddensity_dR_RZ(R, z) * dphi_dR(R,z)
		f += tracer.density_RZ(R, z) * d2phi_dRdz(R,z)
		return f.to(u.pc**-3 * (u.km/u.s)**2 * u.kpc**-2).value

	v2zz = np.zeros(len(R)) * (u.km/u.s)**2
	v2rr = np.zeros(len(R)) * (u.km/u.s)**2
	v2ff = np.zeros(len(R)) * (u.km/u.s)**2
	for i, (Ri, zi) in enumerate(zip(R, z)):
		# get v2zz
		tmp = quad(int_nuv2zz, zi.value, np.inf, args=(Ri, zi.unit))
		nuv2zz = tmp[0]
		nuv2zz *= u.pc**-3 * (u.km/u.s)**2
		v2zz[i] = nuv2zz/tracer.density_RZ(Ri, zi)
		# get v2rr
		v2rr[i] = b * v2zz[i]
		# get v2ff
		tmp = quad(int_nuv2ff1, zi.value, np.inf, args=(Ri, zi.unit))
		nuv2ff1 = tmp[0]
		nuv2ff1  *= u.pc**-3 * (u.km/u.s)**2 * u.kpc**-1
		nuv2ff = b * (Ri*nuv2ff1 + nuv2zz)
		nuv2ff += Ri*tracer.density_RZ(Ri, zi)*dphi_dR(Ri,zi)
		v2ff[i] = nuv2ff/tracer.density_RZ(Ri, zi)

	moments = Table(masked=True)
	moments['v2rr'] = v2rr.to((u.km/u.s)**2)
	moments['v2zz'] = v2zz.to((u.km/u.s)**2)
	moments['v2ff'] = v2ff.to((u.km/u.s)**2)
	moments['v2rr'].mask = (moments['v2rr']<0)
	moments['v2zz'].mask = (moments['v2zz']<0)
	moments['v2ff'].mask = (moments['v2ff']<0)

	return moments
