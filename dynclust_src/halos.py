import numpy as np
from mge.mge_fit_1d import mge_fit_1d
from jam import mge_pot, mge_vcirc
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.constants import G as G_newton
from astropy import cosmology as cosmo
from scipy.interpolate import interp1d

from . import mges

class dm_halo(object):

	def get_mge(self,
			rmin=1*u.pc,
			rmax=100*u.kpc,
			distance=1*u.Mpc,
			nr=100,
			ngauss=10,
			incl=90*u.deg
			):

		self.rmin = rmin
		self.rmax = rmax
		self.nr = nr

		lmin, lmax = np.log10(rmin/u.kpc), np.log10(rmax/u.kpc)
		x = np.logspace(lmin, lmax, nr) * u.kpc
		y = self.density(x)
		p = mge_fit_1d(x.value,
			y.value,
			ngauss = ngauss,
			plot = False,
			quiet = True,
			inner_slope = self.inner_slope,
			outer_slope = self.outer_slope,
			)
		ngauss = p.sol.shape[1]
		self.mge = p.sol
		self.mge = mges.HaloMGE(
			(p.sol[0,:] * y.unit * x.unit).to(u.Msun/u.pc**2),
			(p.sol[1,:] * x.unit / distance * u.rad).to(u.arcsec),
			np.ones(ngauss)
			)
		self.mge.deproject(incl)
		self.mge.distance_scale(distance)

	def check_mge(self):

		lmin, lmax = np.log10(self.rmin/u.kpc), np.log10(self.rmax/u.kpc)
		x = np.logspace(lmin, lmax, self.nr) * u.kpc
		x0 = np.zeros(self.nr)*u.rad

		# density calculations
		d1 = self.density(x)
		d2 = self.mge.density(x0, x0, x)

		# potential calculations
		p1 = self.potential(x)
		p2 = self.mge.potential(x0, x0, x)
		p3 = mge_pot.mge_pot(
		    (x/self.mge.D*u.rad).to(u.arcsec).value,
		    x0.value,
		    self.mge.I.value,
		    1.,
		    (self.mge.sigma).to(u.arcsec).value,
		    self.mge.q,
		    90.,
		    0.,
		    (self.mge.D).to(u.Mpc).value
		    )
		p3 *= (u.km/u.s)**2

		# v circ calculations
		v1 = self.vcirc(x)
		v2 = mge_vcirc.mge_vcirc(
		  	self.mge.I.value,
		    (self.mge.sigma).to(u.arcsec).value,
		    self.mge.q,
		    90.,
		    0.,
		    (self.mge.D).to(u.Mpc).value,
			(x/self.mge.D*u.rad).to(u.arcsec).value
		    )
		v2 *= u.km/u.s

		# plot
		fig, ax = plt.subplots(3, 1, sharex=True)
		ax[0].semilogx(x, 100*(d1-d2)/d1, '-k')
		ax[1].semilogx(x, 100*(p1-p2)/p1, '-k', label='CJAM')
		ax[1].semilogx(x, 100*(p1-p3)/p1, '--r', label='JAM')
		ax[2].semilogx(x, 100*(v1-v2)/v1, '-k')
		ax[1].legend(loc='upper left')
		ax[0].set_ylabel('$\\rho$ error [%]')
		ax[1].set_ylabel('$\Phi$ error [%]')
		ax[2].set_ylabel('$v_\mathrm{circ}$ error [%]')
		ax[2].set_xlabel('r [kpc]')
		fig.tight_layout()
		fig.subplots_adjust(top=0.92, hspace=0)
		fig.suptitle(self.name)
		plt.show()

		return fig

	def get_E_Lz_boundary_interp_func(self, maxr=300*u.kpc, nr=300):

		R = np.linspace(0.*u.pc, maxr, nr)
		pot = self.potential(R)
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


class cusped_DM(dm_halo):	# i.e. NFW

	def __init__(self, rho_s, r_s):

		self.rho_s = rho_s
		self.r_s = r_s
		self.inner_slope = 2
		self.outer_slope = 3
		self.name = 'Cusped halo'

	def density(self, r):

		x = r/self.r_s
		rho = self.rho_s * x**-1 * (1.+x)**-2
		return rho.to(u.Msun/u.pc**3)

	def potential(self, r):

		x = r/self.r_s
		C = -4. * np.pi * G_newton * self.rho_s * self.r_s**3
		pot = C * r**-1 * np.log(1.+x)
		if hasattr(r, '__iter__'):
			idx = np.where(r==0*u.pc)
			if idx[0].size>0:
				pot[idx] = C/self.r_s
		else:
			if r==0*u.pc:
				pot = C/self.r_s
		return pot.to((u.km/u.s)**2)

	def dphi_dr(self, r):

		x = r/self.r_s
		C = -4. * np.pi * G_newton * self.rho_s * self.r_s**3
		dphi_dr = (r*(r+self.r_s))**-1 - r**-2 * np.log(1.+x)
		dphi_dr *= C
		return dphi_dr

	def d2phi_dr2(self, r):

		x = r/self.r_s
		C = -4. * np.pi * G_newton * self.rho_s * self.r_s**3
		d2phi_dr2 = -1. /r /(r+self.r_s)**2.
		d2phi_dr2 -= 2. /r**2. /(r+self.r_s)
		d2phi_dr2 += 2.*np.log(1.+x) /r**3.
		d2phi_dr2 *= C
		return d2phi_dr2

	def vcirc(self, r):

		v_circ = np.sqrt(r * self.dphi_dr(r))
		if hasattr(r, '__iter__'):
			v_circ[r==0] = 0*u.km/u.s
		else:
			if r==0*u.pc:
				v_circ = 0*u.km/u.s
		return v_circ.to(u.km/u.s)

class cosmoNFW(cusped_DM):

	def __init__(self, mvir, r_s, a=1, cosm=cosmo.FlatLambdaCDM(H0=67.9, Om0=0.306, Neff=0., Tcmb0=0.)):
		self.mvir = mvir
		self.a = a
		z = (1.-a)/a
		x = 1./(1. + cosm.Ode0/(cosm.Om0/a**3.)) - 1.
		Del_vir = 18.*np.pi**2. + 82.*x - 39.*x**2
		self.Del_vir = Del_vir
		rhocrit = cosm.critical_density(z)
		rhocrit = rhocrit.to(u.Msun/u.kpc**3)
		rvir3 = 3./4./np.pi * mvir / Del_vir / rhocrit
		rvir3 = rvir3.to(u.kpc ** 3)
		self.rvir = rvir3**(1./3.)
		tmp = np.sqrt(G_newton * self.mvir/self.rvir)
		self.vvir = tmp.to(u.km/u.s)
		self.cvir = (self.rvir/r_s).decompose()
		tmp = self.mvir/(4.0*np.pi*r_s**3.)
		tmp = tmp/(np.log(1.+self.cvir)-self.cvir/(1.+self.cvir))
		rho_s = tmp.to(u.Msun/u.kpc**3)
		self.Evir = self.vvir**2.
		self.Lvir = self.rvir * self.vvir
		super().__init__(rho_s, r_s)


class cored_DM(dm_halo):		# i.e. modified Hubble profile

	def __init__(self, rho_s, r_s):

		self.rho_s = rho_s
		self.r_s = r_s
		self.inner_slope = 0
		self.outer_slope = 3
		self.name = 'Cored halo'

	def density(self, r):

		x = r/self.r_s
		rho = self.rho_s * (1.+x**2.)**-1.5
		return rho.to(u.Msun/u.pc**3)

	def potential(self, r):

		x = r/self.r_s
		C = -4. * np.pi * G_newton * self.rho_s * self.r_s**3
		pot = C * r**-1 * np.arcsinh(x).to(u.rad).value
		return pot.to((u.km/u.s)**2)

	def dphi_dr(self, r):

		x = r/self.r_s
		C = -4. * np.pi * G_newton * self.rho_s * self.r_s**3
		dphi_dr = (r*self.r_s)**-1 * (x**2+1.)**-0.5
		dphi_dr -= r**-2. * np.arcsinh(x).to(u.rad).value
		dphi_dr *= C
		return dphi_dr

	def d2phi_dr2(self, r):

		x = r/self.r_s
		C = -4. * np.pi * G_newton * self.rho_s * self.r_s**3
		d2phi_dr2 = -(r**2.+self.r_s**2.)**-1.5
		d2phi_dr2 -= 2.* r**-2. * (r**2.+self.r_s**2.)**-0.5
		d2phi_dr2 += 2. * r**-3. * np.arcsinh(x).to(u.rad).value
		d2phi_dr2 *= C
		return d2phi_dr2

	def vcirc(self, r):

		v_circ = np.sqrt(r * self.dphi_dr(r))
		return v_circ.to(u.km/u.s)
