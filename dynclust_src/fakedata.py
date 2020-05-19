import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.table import QTable
from astropy.coordinates.matrix_utilities import matrix_product

from scipy.spatial import KDTree
import scipy.optimize as op
from scipy.stats import binned_statistic_2d

import emcee
from sklearn import mixture
from prashpy import plotting

from . import sampling, momcalc, mycoords

from IPython.core.debugger import Tracer

def normalise_E_Lz(E, Lz,
                pot0=1,
                mu_eps=0,
                sig_eps=1,
                potential=None,
                circ_boundary=None,
                return_val='hat'):

    eps = E/pot0
    circ = Lz/potential.Lz_max(E)

    if return_val=='no_hat':
        return eps, circ
    if return_val=='hat':
        eps_hat = (eps - mu_eps)/sig_eps
        circ_hat = circ/circ_boundary(eps)
        return eps_hat, circ_hat
    else:
        return eps, circ, eps_hat, circ_hat

class quadratic_pdf_unit_interval():

	def __init__(self):

		self.init = True

	def get_pars_given_mu_var(self, mu, var):
	    M = np.array([[6, 3, 2], [6, 4, 3], [20, 15, 12]])
	    Minv = np.linalg.inv(M)
	    X = np.array([6, 12*mu, 60*(var+mu**2.)])
	    A = np.dot(Minv, X)
	    a, b, c = A
	    a = 1. - b/2. - c/3.
	    pred = np.array([b, c])
	    return pred

	def L(self, p, x):
	    b, c = p
	    a = 1. - b/2. - c/3.
	    L = a + b*x + c*x**2.
	    return L

	def lnlike(self, p, x):
	    return np.sum(np.log(self.L(p, x)))

	def lnprior(self, p):
	    b, c = p
	    a = 1. - b/2. - c/3.
	    if a+b+c>0 and c<0: #a>0 and
	        if (c!=0 and b**2./c/4.<1.) or (c==0 and b>-2.):
	            return 0.
	    return -np.inf

	def lnprob(self, p, x):
	    lp = self.lnprior(p)
	    if not np.isfinite(lp):
	        return -np.inf
	    return lp + self.lnlike(p, x)

	def max_posterior(self, x):
		nlp = lambda *args: -self.lnprob(*args)
		p0 = [0.5, -0.5]
		p = op.minimize(nlp, p0, args=((x,)))
		return p['x']

	def x_max_positive(self, p):
	    b, c = p
	    a = 1. - b/2. - c/3.
	    return -(b+np.sqrt(b**2 - 4*a*c))/2./c

class gen_sigmoid_pdf_unit_interval:

    def __init__(self):

        self.init = True

    def cdf(self, p, x): # generalised sigmoid
        a, b, c = p
        exp = np.exp(-a*(x-b))
        return (1. + exp)**(-1./c)

    def pdf(self, p, x):
        a, b, c = p
        exp = np.exp(-a*(x-b))
        return a/c * exp * (1. + exp)**(-1./c - 1.)

    def lnlike(self, p, x):
        a, b, c = p
        exp = np.exp(-a*(x-b))
        logL = np.log(a/c) - a*(x-b) + (-1./c - 1)*np.log(1. + exp)
        return np.sum(logL)

    def lnprior(self, p):
        a, b, c = p
        if a<=0.:
            return -np.inf
        if c<=0:
            return -np.inf
        return 0.

    def lnprob(self, p, x):
        lnpri = self.lnprior(p)
        if lnpri == -np.inf:
            return -np.inf
        else:
            return lnpri + self.lnlike(p, x)

    def max_posterior_mcmc(self, x):
        ndim, nwalkers = 3, 100
        a0 = np.random.uniform(8, 10, 100)
        b0 = np.random.uniform(0.3, 0.4, 100)
        c0 = np.random.uniform(0.9, 1.1, 100)
        p0 = np.vstack((a0, b0, c0)).T
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob, args=(x,))
        sampler.run_mcmc(p0, 1000)
        idx = np.where(sampler._lnprob==np.max(sampler._lnprob))
        p = sampler.chain[idx[0][0], idx[1][0],:]
        return p

    def get_max_pdf(self, p):
        result = op.minimize(lambda x: -self.pdf(p, x), [0.5], bounds=((0,1),))
        if result.success:
            xmax = result.x[0]
            fmax = self.pdf(p, xmax)
            return xmax, fmax
        else:
            raise ValueError

def get_2d_counts_bin_centers(x, y, bins, range=None):
    v = np.ones(len(x))
    count, ex, ey, bin = binned_statistic_2d(x, y, v, np.sum,
                                            bins=bins,
                                            range=range)
    pix, piy = np.unravel_index(bin, (len(ex)+1, len(ey)+1))
    pix -= 1
    piy -= 1
    bin_list = np.ravel_multi_index((pix, piy), (len(ex)-1, len(ey)-1))
    idx = np.where(count>0)
    count = count[idx]
    cx = (ex[idx[0]]+ex[idx[0]+1])/2.
    cy = (ey[idx[1]]+ey[idx[1]+1])/2.
    bin_pix = np.ravel_multi_index((idx[0], idx[1]), (len(ex)-1, len(ey)-1))
    return count, cx, cy, bin_list, bin_pix

def generate_fakedata(gcs,
			tracer,
			potential,
			p_red=0.5,
			p_blue=0.5,
			momint=None,
			use_potint=True,
			n=None,
			iom=False,
			r_max_fake=100*u.kpc,
			binned_samples=False,
			sbins=100,
            rad_smp_type='gen_sigmoid'):
    '''
    Make fake data set with same distribution of colour (Gaussian), angular
    distance (a quadratic pdf), and velocity errors (double Gaussian) as the
    input dataset gcs. LOS distances are sampled from the tracer model, and
    velocities sampled given the solutions to the Jeans equation in the
    potential provided.
    '''

    fake = QTable()
    if n is None:
        n = len(gcs)

    if rad_smp_type=='quadratic':

        # fit quadratic pdf to projected r of data
        r_min = 0.99*np.min(gcs['r']).value
        r_max = 1.01*np.max(gcs['r']).value
        x = (gcs['r'].value - r_min)/(r_max - r_min)
        q = quadratic_pdf_unit_interval()
        mu, var = np.mean(x), np.var(x)
        pred = q.get_pars_given_mu_var(mu, var)
        f = lambda *args: q.L(pred, *args)

        # check pdf is valid
        assert f(mu)>0
        if pred[1]<0:
            assert f(0)>0
            assert f(1)>0

        # sample distances
        if pred[1]<0:
            f_max = 1.01 * f(mu)
        else:
            f_max = 1.01 * np.max([f(0), f(1)])
        x = sampling.rejection_sample(f, n, f_max=f_max, x_min=0, x_max=1)
        r = x * (r_max - r_min) + r_min
        fake['r'] = r * u.arcsec

    if rad_smp_type=='gen_sigmoid':

        # fit quadratic pdf to projected r of data
        r_min = 0.99*np.min(gcs['r']).value
        r_max = 1.01*np.max(gcs['r']).value
        x = (gcs['r'].value - r_min)/(r_max - r_min)
        q = gen_sigmoid_pdf_unit_interval()
        p = q.max_posterior_mcmc(x)
        xmax, fmax = q.get_max_pdf(p)
        f = lambda *args: q.pdf(p, *args)
        x = sampling.rejection_sample(f, n, f_max=1.01*fmax, x_min=0, x_max=1)
        r = x * (r_max - r_min) + r_min
        fake['r'] = r * u.arcsec

    # co-ords
    th = np.random.uniform(0, 2.*np.pi, n) * u.rad
    fake['x'] = np.cos(th) * fake['r']
    fake['y'] = np.sin(th) * fake['r']

    # fake colors
    if 'g-i' in gcs.__dict__.keys():
        mu_col = np.median(gcs['g-i']).value
        sig_col = np.std(gcs['g-i']).value
        fake['g-i'] = np.random.normal(mu_col, sig_col, n) * u.mag

    # velocity errors
    mu = np.mean(gcs['dV_los'])
    sig = np.std(gcs['dV_los'])

    gmm = mixture.GaussianMixture(n_components=3)
    X = np.array([gcs['dV_los'].value]).T
    gmm.fit(X)
    err, _ = gmm.sample(n)
    fake['dV_los'] = err[:,0] * u.km/u.s

    for col, unit in zip(
                        ['R', 'Z', 'phi', 'z', 'vR', 'vZ', 'vf', 'V_los'],
                        [u.kpc, u.kpc, u.rad, u.pc] + [u.km/u.s for i in range(4)]
                        ):
                        fake[col] = np.zeros(n) * unit

    print('Sampling distances:')
    if binned_samples:
        max_x = np.max(np.abs(fake['x'])).value
        max_y = np.max(np.abs(fake['y'])).value
        rng = [(-max_x, max_x), (-max_y, max_y)]
        print('... binning in (x,y)')
        counts, cx, cy, bin_list, bin_pix = get_2d_counts_bin_centers(fake['x'],
                                                                    fake['y'],
                                                                    sbins,
                                                                    range=rng)
        tmp = QTable()
        tmp['counts'] = counts
        tmp['x'] = cx * u.arcsec
        tmp['y'] = cy * u.arcsec
        tmp['bin'] = bin_pix
        start = 0
        for i, gc0 in enumerate(tmp):
            print('\t', i, 'out of', len(tmp))
            try:
                z = sampling.sample_distance(gc0,
            								tracer,
            								nsmp=gc0['counts'],
            								rmax=r_max_fake,
                                            ret_RZphi=False)
            except OverflowError:
                Tracer()()
            fake['z'][bin_list==gc0['bin']] = z
        X, Y, Z = mycoords.projected_to_aligned3D(
            (fake['x']/u.rad*tracer.D).to(u.pc),
            (fake['y']/u.rad*tracer.D).to(u.pc),
            fake['z'],
            tracer.inc)
        R = np.sqrt(X**2 + Y**2)
        phi = np.arctan2(Y, X)
        fake['R'] = R.to(u.kpc)
        fake['Z'] = Z.to(u.kpc)
        fake['phi'] = phi
    else:
        for i, gc0 in enumerate(fake):
            if (i%1e4==0):
                print('\t', i, 'out of', n)
            R, Z, phi, z = sampling.sample_distance(gc0,
            								tracer,
            								nsmp=1,
            								rmax=r_max_fake)
            for (col, dat) in zip(['R', 'Z', 'phi', 'z'], [R, Z, phi, z]):
                fake[col][i] = dat

    M = mycoords.rotate(tracer.inc)

    print('Sampling velocities:')
    if binned_samples:
        print('... binning in (R,Z)')
        counts, cR, cZ, bin_list, bin_pix = get_2d_counts_bin_centers(fake['R'],
                                                                    fake['Z'],
                                                                    sbins)
        moments = momcalc.velocity_moments(cR * u.kpc,
        								cZ * u.kpc,
        								tracer,
        								potential,
        								momint=momint)
        for i, (ccc, b, m) in enumerate(zip(counts, bin_pix, moments)):
            print('\t', i, 'out of', len(counts))
            v = sampling.sample_velocity(tracer, m, int(ccc))
            idx = np.where(bin_list==b)
            fake['vR'][idx] = v['vR']
            fake['vZ'][idx] = v['vZ']
            fake['vf'][idx] = v['vf']
        vx = mycoords.cylin_to_aligned3D_vels_phiarray(fake['vR'],
                									fake['vf'],
                									fake['vZ'],
                									fake['phi'])
        vxyz = matrix_product(M, vx)
        fake['V_los'] = vxyz[2]
    else:
    	moments = momcalc.velocity_moments(fake['R'],
    									fake['Z'],
    									tracer,
    									potential,
    									momint=momint)
    	for i, gc0 in enumerate(fake):
    		if (i%1e4==0):
    			print('\t', i, 'out of', n)
    		v_los = 1000*u.km/u.s
    		while np.abs(v_los)>700.*u.km/u.s:
    			v = sampling.sample_velocity(tracer, moments[i], 1)
    			vx = mycoords.cylin_to_aligned3D_vels(v['vR'],
    												v['vf'],
    												v['vZ'],
    												fake['phi'][i])
    			vxyz = matrix_product(M, vx)
    			v_los = vxyz[2][0]
    		fake['vR'][i] = v['vR']
    		fake['vZ'][i] = v['vZ']
    		fake['vf'][i] = v['vf']
    		fake['V_los'][i] = v_los

    fake['p_red'] = p_red
    fake['p_blue'] = p_blue

    if iom:
    	fake['pot'] = potential.potential(fake['R'],
    									fake['Z'],
    									use_potint=use_potint)
    	tmp = 1.*fake['pot']
    	tmp += 0.5*(fake['vR']**2. + fake['vf']**2. + fake['vZ']**2.)
    	fake['E'] = tmp.to((u.km/u.s)**2)
    	tmp = fake['R'] * fake['vf']
    	fake['Lz'] = tmp.to(u.kpc*u.km/u.s)

    return fake

def plot_dataset_ling_fig4(dat, qav=0.85):

	fig, ax = plt.subplots(3, 1, figsize=(6,5), sharex='col')

	Rdash = np.sign(dat['x']) * np.sqrt(dat['x']**2. + (dat['y']/qav)**2.)
	ax[0].scatter(Rdash, dat['g-i'], c=dat['g-i'], vmin=0.6, vmax=1.5, s=5, cmap='coolwarm')
	ax[1].scatter(Rdash, dat['V_los'], c=dat['g-i'], vmin=0.6, vmax=1.5, s=5, cmap='coolwarm')
	ax[2].scatter(Rdash, dat['dV_los'], c=dat['g-i'], vmin=0.6, vmax=1.5, s=5, cmap='coolwarm')

	ax[0].set_ylabel('g-i [mag]')
	ax[1].set_ylabel('$v_\mathrm{los}$ [km/s]')
	ax[2].set_ylabel('$\delta v_\mathrm{los}$ [km/s]')
	ax[2].set_xlabel('R [arcsec]')

	fig.tight_layout()
	fig.subplots_adjust(hspace=0., top=0.8, right=0.8)

	axh1 = fig.add_subplot(3,3,1)
	tmp = ax[0].get_position()
	axh1.set_position([tmp.x0, tmp.y0+tmp.height, tmp.width, 0.15])
	axh1.hist(Rdash[dat['p_red']>0.5], range=ax[0].get_xlim(), bins=25, histtype='step', color='r')
	axh1.hist(Rdash[dat['p_blue']>0.5], range=(-600,600), bins=25, histtype='step', color='b')

	axh2 = fig.add_subplot(3,3,2)
	tmp = ax[0].get_position()
	axh2.set_position([tmp.x0+tmp.width, tmp.y0, 0.15, tmp.height])
	axh2.hist(dat['g-i'][dat['p_red']>0.5], range=ax[0].get_ylim(), bins=25, histtype='step', color='r', orientation='horizontal')
	axh2.hist(dat['g-i'][dat['p_blue']>0.5], range=ax[0].get_ylim(), bins=25, histtype='step', color='b', orientation='horizontal')
	axh2.set_ylim(ax[0].get_ylim())

	axh3 = fig.add_subplot(3,3,3)
	tmp = ax[1].get_position()
	axh3.set_position([tmp.x0+tmp.width, tmp.y0, 0.15, tmp.height])
	axh3.hist(dat['V_los'][dat['p_red']>0.5], range=ax[1].get_ylim(), bins=25, histtype='step', color='r', orientation='horizontal')
	axh3.hist(dat['V_los'][dat['p_blue']>0.5], range=ax[1].get_ylim(), bins=25, histtype='step', color='b', orientation='horizontal')
	axh3.set_ylim(ax[1].get_ylim())

	axh4 = fig.add_subplot(3,3,4)
	tmp = ax[2].get_position()
	axh4.set_position([tmp.x0+tmp.width, tmp.y0, 0.15, tmp.height])
	axh4.hist(dat['dV_los'][dat['p_red']>0.5], range=ax[2].get_ylim(), bins=25, histtype='step', color='r', orientation='horizontal')
	axh4.hist(dat['dV_los'][dat['p_blue']>0.5], range=ax[2].get_ylim(), bins=25, histtype='step', color='b', orientation='horizontal')
	axh4.set_ylim(ax[2].get_ylim())

	for ax0 in [axh1, axh2, axh3, axh4]:
	    ax0.set_xticklabels([])
	    ax0.set_yticklabels([])

	plt.show()

	return fig

def get_random_background(X, n, fd,
						free=None,
						constrained=False,	# whether or not to constrain
						sigr_true=None,		# true value of sigma_r
						sigv_true=None,		# true value of sigma_v
						rtol=None):			# relative tolerance from truth

	# list of free points
	if free is None:
		free = np.full(X.shape[0], True, dtype=bool)
	free_nbr = np.where(free)
	free_nbr = free_nbr[0]

	# get background
	if constrained:
		sigr_check, sigv_check = False, False
		while not (sigr_check & sigv_check):
			bkgnd = np.random.choice(free_nbr, n, replace=False)
			sigr = np.std(fd[bkgnd]['r'])
			sigv = np.std(fd[bkgnd]['V_los'])
			sigr_check = (np.abs(sigr/sigr_true-1.) < rtol)
			sigv_check = (np.abs(sigv/sigv_true-1.) < rtol)
	else:
		bkgnd = np.random.choice(free_nbr, n, replace=False)

	# update list of free points
	free[bkgnd] = False

	return bkgnd, free

def get_cluster(X,
			fd,
			idx_c,
			rad,
			n,
			kdtree=None,
			free=None,
			constrained=False,	# whether or not to constrain cluster properties
			sigr_true=None,		# true value of sigma_r
			sigv_true=None,		# true value of sigma_v
			mu_r_true=None,		# true value of mu_r
			rtol=None):			# relative tolerance from truth

	# list of free points
	if free is None:
	    free = np.full(X.shape[0], True, dtype=bool)

	# get all free nbrs
	if kdtree is None:
	    dist = np.sqrt((X[:,0] - X[idx_c,0])**2. + (X[:,1] - X[idx_c,1])**2.)
	    free_nbr = np.where((dist < rad) & free)
	    free_nbr = free_nbr[0]
	else:
	    nbr = kdtree.query_ball_point(X[idx_c,:], rad)
	    nbr = np.array(nbr)
	    idx_free_nbr = np.where(free[nbr])
	    free_nbr = nbr[idx_free_nbr]

	# get cluster members if there are enough
	if np.size(free_nbr)>=n:
		cluster = np.random.choice(free_nbr, n, replace=False)
		success = True
		if constrained:
			mur = np.mean(fd[cluster]['r'])
			sigr = np.std(fd[cluster]['r'])
			sigv = np.std(fd[cluster]['V_los'])
			sigr_check = (np.abs(sigr/sigr_true-1.) < rtol)
			sigv_check = (np.abs(sigv/sigv_true-1.) < rtol)
			mur_check = (np.abs(mur/mu_r_true-1.) < rtol)
			success = sigv_check & mur_check
	else:
	    success = False
	if not success:
		cluster = np.array([], dtype='int64')

	# update list of free points
	free[cluster] = False

	return success, cluster, free

def find_nearest_unique(a, a_argsort, v):

	# return unique indices of a into which entries of v should be inserted...
	# ... to maintain ordering

	idx0 = np.searchsorted(a[a_argsort], v)
	unq0, cnts0 = np.unique(idx0, return_counts=True)

	idx = np.searchsorted(a[a_argsort], v)

	# deal with cases where 2 of entries v have same a-index
	unq, cnts = np.unique(idx, return_counts=True)
	while len(idx)!=len(unq):
		idx0 = np.where(cnts>1)
		idx0 = idx0[0][0]
		idx1 = np.where(idx==unq[idx0])
		offset = np.array([i for i in range(len(idx1[0]))])
		idx[idx1] = idx[idx1] + offset
		unq, cnts = np.unique(idx, return_counts=True)

	# fix cases where we have offset the indices beyond allowed limit
	max_idx = len(a)
	idx_fix = np.where(idx > max_idx)
	n_fix = len(idx_fix[0])
	if n_fix>0:
		unused_idx = np.setdiff1d(np.arange(max_idx), idx)
		idx[idx_fix] = unused_idx[-n_fix:]

	# fix cases where min(v) < min(a) i.e. where indices = 0
	idx_fix = np.where(idx < 1)
	n_fix = len(idx_fix[0])
	if n_fix>0:
		unused_idx = np.setdiff1d(np.arange(max_idx), idx)
		idx[idx_fix] = unused_idx[0:n_fix]

	try:
		retval = a_argsort[idx-1]
	except IndexError:
		print('Re-write this because it has screwed up in a million ways')

	return retval

def get_background_constrained_r(X,
								n,
								fd,
								fdcluster,
								free=None,
								update_free=True):

	# list of free points
	if free is None:
		free = np.full(X.shape[0], True, dtype=bool)
	free_nbr = np.where(free)
	free_nbr = free_nbr[0]

	# get r dist of cluster
	pdf, e = np.histogram(fdcluster['r'].value, bins=10000)
	cdf = np.cumsum(pdf)/np.sum(pdf)

	xr = np.interp(fd['r'].value, e[1::], cdf)
	x = np.random.uniform(0, 1, n)
	xr_argsort = np.argsort(xr[free_nbr])
	idx_bkgnd = find_nearest_unique(xr[free_nbr], xr_argsort, x)
	bkgnd = free_nbr[idx_bkgnd]

	if update_free:
		# update list of free points
		free[bkgnd] = False

	return bkgnd, free

def get_background_constrained_v(X,
								n,
								fd,
								fdcluster,
								free=None,
								update_free=True):
	# list of free points
	if free is None:
		free = np.full(X.shape[0], True, dtype=bool)
	free_nbr = np.where(free)
	free_nbr = free_nbr[0]

	# get v dist of cluster
	pdf, e = np.histogram(fdcluster['V_los'].value, bins=10000)
	cdf = np.cumsum(pdf)/np.sum(pdf)

	xv = np.interp(fd['V_los'].value, e[1::], cdf)
	x = np.random.uniform(0, 1, n)
	xv_argsort = np.argsort(xv[free_nbr])
	idx_bkgnd = find_nearest_unique(xv[free_nbr], xv_argsort, x)
	bkgnd = free_nbr[idx_bkgnd]

	if update_free:
		# update list of free points
		free[bkgnd] = False

	return bkgnd, free

def get_background_constrained_r_and_v(X, n, fd, fdcluster, free=None):

	# list of free points
	if free is None:
		free = np.full(X.shape[0], True, dtype=bool)
	free_nbr = np.where(free)
	free_nbr = free_nbr[0]

	bkgnd_r, _ = get_background_constrained_r(X,
											int(100*n),
											fd,
											fdcluster,
											free=free,
											update_free=False)

	bkgnd_v, free_v = get_background_constrained_v(X[bkgnd_r],
											n,
											fd[bkgnd_r],
											fdcluster,
											free=None,
											update_free=True)

	# update list of free points
	bkgnd = bkgnd_r[bkgnd_v]
	free[bkgnd] = False

	return bkgnd, free

def get_dataset(fd,
			X=None,
			free=None,
            kdtree=None,
            N=50,         		# number of points in cluster
            clu_rad=0.03,       # radius of cluster in dimensionless (E,Lz)
            max_attempts=1000,	# maximum number of attempts to pick cluster
			constrained=False,
			sigr_true=None,
			sigv_true=None,
			mur_true=None,
			rtol=None):

	if X is None:
		X = np.array([fd['circ_hat'].value, fd['eps_hat'].value]).T
	if free is None:
		free = np.full(X.shape[0], True, dtype=bool)
	if kdtree is None:
		kdtree = KDTree(X, 100)

	N = int(N)

	attempts = 0
	success = False
	while (success == False) & (attempts < max_attempts):
		rnd = np.random.choice(np.where(free)[0], 1, replace=False)[0]
		success, cluster, free = get_cluster(X, fd, rnd, clu_rad, N,
		                                    kdtree=kdtree,
		                                    free=free,
											constrained=constrained,
											sigr_true=sigr_true,
											sigv_true=sigv_true,
											mu_r_true=mur_true,
											rtol=rtol)
		attempts += 1
	if (success == False):
		# raise ValueError('Max iterations reached. No cluster found.')
		return success, [], [], free
	else:
		random_bkg, free = get_background_constrained_r_and_v(X,
															N,
															fd,
															fd[cluster],
															free=free)
		# random_bkg, free = get_random_background(X, N, fd,
		# 										free=free,
		# 										constrained=constrained,
		# 										sigr_true=sigr_true,
		# 										sigv_true=sigv_true,
		# 										rtol=rtol)
		indices = np.concatenate((cluster, random_bkg))
		in_cluster = [1 for i in range(N)]
		in_cluster += [0 for i in range(N)]
		in_cluster = np.array(in_cluster)

		if len(indices)!=len(np.unique(indices)):
			raise ValueError('an entry has been used twice')

		return success, indices, in_cluster, free

##############################
# plotting
##############################

def plot_clusters(fd, X, clusters):

	fig, ax = plt.subplots(1, 2, figsize=(8,4))

	kw_draw_contours = {}
	kw_draw_contours['step_fact'] = 2.
	kw_draw_contours['min_cnt'] = 10
	kw_draw_contours['kw_contour'] = {'colors':'k', 'linewidths':0.5}

	plotting.draw_countours(X[:,0], X[:,1],
						axis=ax[0],
						**kw_draw_contours)
	for c in clusters:
	    ax[0].plot(X[c,0], X[c,1], '.', ms=4)

	plotting.draw_countours(fd['r'].value, fd['V_los'].value,
							axis=ax[1],
							**kw_draw_contours)
	for c in clusters:
	    ax[1].plot(fd['r'][c], fd['V_los'][c], '.', ms=4)

	ax[0].set_xlabel('$\\lambda$')
	ax[0].set_ylabel('$E/\\Phi_0$')
	ax[1].set_xlabel('$r$ [arcsec]')
	ax[1].set_ylabel('$V_\mathrm{LOS}$ [km/s]')

	fig.tight_layout()

	return fig

def get_circ_boundary(fd, potential_mge, plot=False):

	# scale E by central potential and Lz by Lz_circ(E)
	pot0 = potential_mge.potential([0.]*u.kpc, [0.]*u.kpc)[0]
	enrg = (fd['E']/pot0).value
	circ = (fd['Lz']/potential_mge.Lz_max(fd['E'])).value

	# get eps bins
	nbins = 1000
	ntot = len(fd)
	tmp = np.linspace(0, ntot, nbins+1)
	h, e = np.histogram(enrg, bins=1000)
	h = np.cumsum(h)
	h = np.concatenate(([0], h))
	enrg_bins = np.interp(tmp, h, e)
	enrg_bins_c = (enrg_bins[0:-1] + enrg_bins[1::])/2.

	# get sigma_circ in eps bins
	mean_c = np.zeros(nbins)
	for ibin in range(nbins):
	    idx = np.where((enrg>=enrg_bins[ibin]) & (enrg<enrg_bins[ibin+1]))
	    mean_c[ibin] = np.sqrt(np.sum(circ[idx]**2.)/idx[0].size)

	# firt sigmoid to sigma_circ(eps)
	def sigmoid(p, x):
	    a, b, alpha, xT = p
	    return a + (b-a)/(1.+np.exp(-alpha*(x-xT)))
	def chi2(p, x, y):
	    return np.sum(1.*(y-sigmoid(p, x))**2.)
	import scipy.optimize as op
	p0 = (0.13, 0.4, 30.0, 0.25)
	p = op.minimize(chi2, p0, args=(enrg_bins_c, mean_c))

	def circ_boundary(eps):
		return sigmoid(p.x, eps)

	if plot:

		fig, ax = plt.subplots(1, 3, figsize=(9, 3))

		kw_draw_contours={}
		kw_draw_contours['step_fact'] = 2.
		kw_draw_contours['min_cnt'] = 10
		kw_draw_contours['kw_contour'] = {'colors':'k', 'linewidths':0.5}

		plotting.draw_countours(fd['Lz'].value/1e4, fd['E'].value/1e5,
							axis=ax[0],
							**kw_draw_contours)
		E = np.linspace(0, -1e6, 100)*(u.km/u.s)**2.
		ax[0].plot(potential_mge.Lz_max(E)/1e4, E/1e5, ':r')
		ax[0].plot(-potential_mge.Lz_max(E)/1e4, E/1e5, ':r')

		plotting.draw_countours(circ, enrg, axis=ax[1], extent=(-1,1,-0.2,1), **kw_draw_contours)
		enrg_arr = np.linspace(-0.2, 1.0, 100)
		ax[1].plot(mean_c, enrg_bins_c, '.m')
		ax[1].plot(circ_boundary(enrg_arr), enrg_arr, ':m')
		ax[1].plot(-circ_boundary(enrg_arr), enrg_arr, ':m')

		x = circ/circ_boundary(enrg)
		plotting.draw_countours(x, enrg, axis=ax[2], extent=(-3,3,-0.2,1), **kw_draw_contours)
		# get sigma_circ in eps bins
		mean_c = np.zeros(nbins)
		cmap = plt.cm.viridis
		for ibin in range(nbins):
		    idx = np.where((enrg>=enrg_bins[ibin]) & (enrg<enrg_bins[ibin+1]))
		    mean_c[ibin] = np.sqrt(np.sum(x[idx]**2.)/idx[0].size)
		ax[2].plot(mean_c, enrg_bins_c, '.m')

		ax[0].set_xlabel('$L_z$ [$10^5$ kpc km/s]')
		ax[0].set_ylabel('$E$ [$10^4$ (km/s)$^2$]')
		ax[0].set_xlim(-5,5)
		ax[0].set_ylim(2,-10)

		ax[1].set_xlabel('$\\lambda$')
		ax[1].set_ylabel('$\\epsilon$')
		ax[1].set_xlim(-1,1)
		ax[1].set_ylim(2e5/pot0.value, -1e6/pot0.value)

		ax[2].set_xlabel('$\\hat{\\lambda}$')
		ax[2].set_ylabel('$\\epsilon$')
		ax[2].set_xlim(-3,3)
		ax[2].set_ylim(2e5/pot0.value, -1e6/pot0.value)

		fig.tight_layout()

	return circ_boundary
