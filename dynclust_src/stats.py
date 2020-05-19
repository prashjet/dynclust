import numpy as np

from scipy.optimize import minimize
from scipy.spatial import distance
from scipy import special
from scipy.stats import norm

from astropy import io
import astropy.units as u
from astropy import table

from sklearn import mixture
from sklearn import metrics
from sklearn.neighbors.kde import KernelDensity

import emcee

from . import sampling

from pathos.multiprocessing import ProcessingPool as Pool

from IPython.core.debugger import Tracer

def get_two_gmm_datstruc(n, n_gmm):

    dtype = [
        ('means_circ', '({0},2)float64'.format(n_gmm)),
        ('weights_circ', '{0}float64'.format(n_gmm)),
        ('covars_circ', '({0},2,2)float64'.format(n_gmm)),
        ('prec_circ', '({0},2,2)float64'.format(n_gmm)),
        ('prec_chol_circ', '({0},2,2)float64'.format(n_gmm)),
        ('means_chat', '({0},2)float64'.format(n_gmm)),
        ('weights_chat', '{0}float64'.format(n_gmm)),
        ('covars_chat', '({0},2,2)float64'.format(n_gmm)),
        ('prec_chat', '({0},2,2)float64'.format(n_gmm)),
        ('prec_chol_chat', '({0},2,2)float64'.format(n_gmm)),
        ('E_smp_mu', 'float64'),
        ('Lz_smp_mu', 'float64'),
        ('E_smp_std', 'float64'),
        ('Lz_smp_std', 'float64')
        ]
    gmmfit = np.zeros(n, dtype=dtype)

    return gmmfit

def get_gmm_two_fits(gc0,
                potential,
                tracer,
                n_gmm,
                circ_boundary,
                max_r,
                momint=None,
                savefile=None,
                ):

    gmmfit = get_two_gmm_datstruc(1, n_gmm)

    gmm = mixture.GaussianMixture(n_components=n_gmm,
                                covariance_type='full',
                                max_iter=100)

    smp = sampling.sample_E_Lz(gc0, potential, tracer,
        n_dsmp=300,
        rmax=max_r,
        n_vsmp=300,
        momint=momint)

    gmmfit['E_smp_mu'][0] = np.mean(smp['E']).value/1e5
    gmmfit['Lz_smp_mu'][0] = np.mean(smp['Lz']).value/1e4
    gmmfit['E_smp_std'][0] = np.std(smp['E']).value/1e5
    gmmfit['Lz_smp_std'][0] = np.std(smp['Lz']).value/1e4

    # normalise 1st way
    pot0 = potential.potential([0.]*u.kpc, [0.]*u.kpc)[0]
    eps = smp['E']/pot0
    circ = smp['Lz']/potential.Lz_max(smp['E'])

    # fit GMM
    X = np.array([eps.value, circ.value]).T
    gmm.fit(X)
    gmmfit['means_circ'][0] = gmm.means_
    gmmfit['weights_circ'][0] = gmm.weights_
    gmmfit['covars_circ'][0] = gmm.covariances_
    gmmfit['prec_circ'][0] = gmm.precisions_
    gmmfit['prec_chol_circ'][0] = gmm.precisions_cholesky_

    # normalise 2nd way
    chat = circ/circ_boundary(eps)

    # fit GMM
    X = np.array([eps.value, chat.value]).T
    gmm.fit(X)
    gmmfit['means_chat'][0] = gmm.means_
    gmmfit['weights_chat'][0] = gmm.weights_
    gmmfit['covars_chat'][0] = gmm.covariances_
    gmmfit['prec_chat'][0] = gmm.precisions_
    gmmfit['prec_chol_chat'][0] = gmm.precisions_cholesky_

    return gmmfit

def get_gmm_datstruc(n, n_gmm):

    dtype = [
        ('means', '({0},2)float64'.format(n_gmm)),
        ('weights', '{0}float64'.format(n_gmm)),
        ('covars', '({0},2,2)float64'.format(n_gmm)),
        ('prec', '({0},2,2)float64'.format(n_gmm)),
        ('prec_chol', '({0},2,2)float64'.format(n_gmm)),
        ('E_smp_mu', 'float64'),
        ('Lz_smp_mu', 'float64'),
        ('E_smp_std', 'float64'),
        ('Lz_smp_std', 'float64'),
        ('logL_true', 'float64'),
        ('label', str)
        ]
    gmmfit = np.zeros(n, dtype=dtype)

    return gmmfit

def get_gmm_fit(gc0,
                potential,
                tracer,
                n_gmm,
                max_r,
                momint=None,
                savefile=None,
                ):

    gmmfit = get_gmm_datstruc(1, n_gmm)

    gmm = mixture.GaussianMixture(n_components=n_gmm,
                                covariance_type='full',
                                max_iter=100)

    smp = sampling.sample_E_Lz(gc0, potential, tracer,
        n_dsmp=300,
        rmax=max_r,
        n_vsmp=300,
        momint=momint)

    gmmfit['E_smp_mu'][0] = np.mean(smp['E']).value/1e5
    gmmfit['Lz_smp_mu'][0] = np.mean(smp['Lz']).value/1e4
    gmmfit['E_smp_std'][0] = np.std(smp['E']).value/1e5
    gmmfit['Lz_smp_std'][0] = np.std(smp['Lz']).value/1e4

    # normalise 1st way
    pot0 = potential.potential([0.]*u.kpc, [0.]*u.kpc)[0]
    eps = smp['E']/pot0
    circ = smp['Lz']/potential.Lz_max(smp['E'])

    # fit GMM
    X = np.array([eps.value, circ.value]).T
    gmm.fit(X)
    gmmfit['means'][0] = gmm.means_
    gmmfit['weights'][0] = gmm.weights_
    gmmfit['covars'][0] = gmm.covariances_
    gmmfit['prec'][0] = gmm.precisions_
    gmmfit['prec_chol'][0] = gmm.precisions_cholesky_

    # get pdf at true value
    if 'E' in gc0.columns:
        eps0 = (gc0['E']/pot0).value
        circ0 = (gc0['Lz']/potential.Lz_max(gc0['E'])).value
        logL0 = gmm.score_samples(np.array([[eps0], [circ0]]).T)
        gmmfit['logL_true'][0] = logL0[0]

    return gmmfit

def gmm_jsd(gmm_p, gmm_q, n_samples=10**5):
    X, _ = gmm_p.sample(n_samples)
    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)
    log_mix_X = np.logaddexp(log_p_X, log_q_X)

    Y, _ = gmm_q.sample(n_samples)
    log_p_Y = gmm_p.score_samples(Y)
    log_q_Y = gmm_q.score_samples(Y)
    log_mix_Y = np.logaddexp(log_p_Y, log_q_Y)

    js_divergence = (log_p_X.mean() - (log_mix_X.mean() - np.log(2))
            + log_q_Y.mean() - (log_mix_Y.mean() - np.log(2))) / 2
    js_distance = np.sqrt(js_divergence)

    return js_distance

def get_jsd_two_gmm(gmmfit, savefile=None):

    n_idx = len(gmmfit)
    idx = np.arange(n_idx)
    labels = gmmfit['label']
    i_list, i_label = [], []
    j_list, j_label = [], []
    for i in range(n_idx):
        i_list = i_list + list(np.repeat(int(idx[i]), n_idx-i-1))
        i_label = i_label + list(np.repeat(labels[i], n_idx-i-1))
        if i<n_idx:
            j_list = j_list + list(idx[i+1::])
            j_label = j_label + list(labels[i+1::])
    n_pairs = len(i_list)

    gc_pairs = table.Table()
    gc_pairs['i'] = i_label
    gc_pairs['j'] = j_label
    gc_pairs['js_circ'] = np.zeros(len(gc_pairs), dtype='float64')
    gc_pairs['js_chat'] = np.zeros(len(gc_pairs), dtype='float64')

    n_gmm = len(gmmfit['weights_circ'][0])
    gmm_i = mixture.GaussianMixture(n_components=n_gmm)
    gmm_j = mixture.GaussianMixture(n_components=n_gmm)

    for idx in range(n_pairs):

        i, j = i_list[idx], j_list[idx]

        gmm_i.means_ = gmmfit['means_circ'][i]
        gmm_i.weights_ = gmmfit['weights_circ'][i]
        gmm_i.covariances_ = gmmfit['covars_circ'][i]
        gmm_i.precisions_ = gmmfit['prec_circ'][i]
        gmm_i.precisions_cholesky_ = gmmfit['prec_chol_circ'][i]

        gmm_j.means_ = gmmfit['means_circ'][j]
        gmm_j.weights_ = gmmfit['weights_circ'][j]
        gmm_j.covariances_ = gmmfit['covars_circ'][j]
        gmm_j.precisions_ = gmmfit['prec_circ'][j]
        gmm_j.precisions_cholesky_ = gmmfit['prec_chol_circ'][j]

        gc_pairs['js_circ'][idx] = gmm_jsd(gmm_i, gmm_j)

        gmm_i.means_ = gmmfit['means_chat'][i]
        gmm_i.weights_ = gmmfit['weights_chat'][i]
        gmm_i.covariances_ = gmmfit['covars_chat'][i]
        gmm_i.precisions_ = gmmfit['prec_chat'][i]
        gmm_i.precisions_cholesky_ = gmmfit['prec_chol_chat'][i]

        gmm_j.means_ = gmmfit['means_chat'][j]
        gmm_j.weights_ = gmmfit['weights_chat'][j]
        gmm_j.covariances_ = gmmfit['covars_chat'][j]
        gmm_j.precisions_ = gmmfit['prec_chat'][j]
        gmm_j.precisions_cholesky_ = gmmfit['prec_chol_chat'][j]

        gc_pairs['js_chat'][idx] = gmm_jsd(gmm_i, gmm_j)

    if savefile is not None:

        io.ascii.write(gc_pairs, savefile)

    return gc_pairs

def get_jsd_gmm(gmmfit, savefile=None, multiprocessing=False, n_pool=10):

    n_idx = len(gmmfit)
    idx = np.arange(n_idx)
    labels = gmmfit['label']
    i_list, i_label = [], []
    j_list, j_label = [], []
    for i in range(n_idx):
        i_list = i_list + list(np.repeat(int(idx[i]), n_idx-i-1))
        i_label = i_label + list(np.repeat(labels[i], n_idx-i-1))
        if i<n_idx:
            j_list = j_list + list(idx[i+1::])
            j_label = j_label + list(labels[i+1::])
    n_pairs = len(i_list)

    gc_pairs = table.Table()
    gc_pairs['i'] = i_label
    gc_pairs['j'] = j_label
    gc_pairs['jsd'] = np.zeros(len(gc_pairs), dtype='float64')

    n_gmm = len(gmmfit['weights'][0])
    gmm_i = mixture.GaussianMixture(n_components=n_gmm)
    gmm_j = mixture.GaussianMixture(n_components=n_gmm)

    def wrapper(idx):
        i, j = i_list[idx], j_list[idx]
        gmm_i.means_ = gmmfit['means'][i]
        gmm_i.weights_ = gmmfit['weights'][i]
        gmm_i.covariances_ = gmmfit['covars'][i]
        gmm_i.precisions_ = gmmfit['prec'][i]
        gmm_i.precisions_cholesky_ = gmmfit['prec_chol'][i]
        gmm_j.means_ = gmmfit['means'][j]
        gmm_j.weights_ = gmmfit['weights'][j]
        gmm_j.covariances_ = gmmfit['covars'][j]
        gmm_j.precisions_ = gmmfit['prec'][j]
        gmm_j.precisions_cholesky_ = gmmfit['prec_chol'][j]
        jsd = gmm_jsd(gmm_i, gmm_j)
        gc_pairs['jsd'][idx] = jsd
        return jsd

    if multiprocessing:
        pool = Pool(n_pool)
        jsd = pool.map(wrapper, range(n_pairs))
        pool.close()
        gc_pairs['jsd'] = jsd

    else:
        for idx in range(n_pairs):
            wrapper(idx)

    if savefile is not None:

        io.ascii.write(gc_pairs, savefile)

    return gc_pairs

def get_jsd_crossterms_gmm(gmmfit1, gmmfit2, savefile=None):

    n_idx1, n_idx2 = len(gmmfit1), len(gmmfit2)
    idx1, idx2 = np.arange(n_idx1), np.arange(n_idx2)
    n_idx = n_idx1 * n_idx2
    labels1, labels2 = gmmfit1['label'], gmmfit2['label']

    i_list, i_label = [], []
    j_list, j_label = [], []
    for i in range(n_idx1):
        i_list = i_list + list(np.repeat(int(idx1[i]), n_idx2))
        i_label = i_label + list(np.repeat(labels1[i], n_idx2))
        j_list = j_list + list(idx2)
        j_label = j_label + list(labels2)

    n_pairs = len(i_list)

    gc_pairs = table.Table()
    gc_pairs['i'] = i_label
    gc_pairs['j'] = j_label
    gc_pairs['jsd'] = np.zeros(len(gc_pairs), dtype='float64')

    n_gmm = len(gmmfit1['weights'][0])
    gmm_i = mixture.GaussianMixture(n_components=n_gmm)
    gmm_j = mixture.GaussianMixture(n_components=n_gmm)

    for idx in range(n_pairs):

        i, j = i_list[idx], j_list[idx]

        gmm_i.means_ = gmmfit1['means'][i]
        gmm_i.weights_ = gmmfit1['weights'][i]
        gmm_i.covariances_ = gmmfit1['covars'][i]
        gmm_i.precisions_ = gmmfit1['prec'][i]
        gmm_i.precisions_cholesky_ = gmmfit1['prec_chol'][i]

        gmm_j.means_ = gmmfit2['means'][j]
        gmm_j.weights_ = gmmfit2['weights'][j]
        gmm_j.covariances_ = gmmfit2['covars'][j]
        gmm_j.precisions_ = gmmfit2['prec'][j]
        gmm_j.precisions_cholesky_ = gmmfit2['prec_chol'][j]

        gc_pairs['jsd'][idx] = gmm_jsd(gmm_i, gmm_j)

    if savefile is not None:

        io.ascii.write(gc_pairs, savefile)

    return gc_pairs

def get_cdf(x, rng=(0,1.), bins=1000):
    cdf, e = np.histogram(x, range=rng, bins=bins-1)
    cdf = 1.*np.cumsum(cdf)/np.sum(cdf)
    cdf = np.concatenate((np.zeros(1), cdf))
    return e, cdf

def get_cdf_cis(cdfs, cis=[68., 95.], median=True):

    idx_not_nan = np.where(np.all(cdfs==cdfs, 1))
    cdfs = cdfs[idx_not_nan[0], :]

    bins = cdfs.shape[1]
    cdf_cis = np.zeros([5, bins])
    pcs = [50.-x/2. for x in cis[::-1]]
    if median:
        pcs += [50.]
    pcs += [50.+x/2. for x in cis]
    for i in range(bins):
        for j, pc in enumerate(pcs):
            cdf_cis[j, i] = np.percentile(cdfs[:, i], pc)
    return cdf_cis

def get_cdfs_and_cis(x, rng=(0,1.), bins=1000, cis=[68., 95.], median=True):
    n = x.shape[0]
    cdfs = np.zeros([n, bins])
    for i in range(n):
        e, cdf = get_cdf(x[i,:], rng=rng, bins=bins)
        cdfs[i, :] = cdf
    cdf_cis = get_cdf_cis(cdfs, cis=cis, median=median)
    return e, cdfs, cdf_cis

def get_distance_matrix(j00,
                        ilab = 'i',
                        jlab = 'j',
                        metric = 'jsd',
                        return_nodes=False):
    nodes = np.concatenate((j00[ilab], j00[jlab]))
    nodes = np.unique(nodes)    # removes duplicates and returns sorted
    n_nodes = len(nodes)
    X = np.zeros((n_nodes,n_nodes))
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            tmp1 = (j00[ilab]==nodes[i]) & (j00[jlab]==nodes[j])
            tmp2 = (j00[jlab]==nodes[i]) & (j00[ilab]==nodes[j])
            tmp3 = np.where(tmp1 | tmp2)
            X[i,j] = j00[tmp3][metric]
            X[j,i] = j00[tmp3][metric]
    if return_nodes:
        return X, nodes
    else:
        return X

def plot_cdf_cis(e, cdf_cis, ax, fill=False, hatch=False, median=False, c='r'):

    if fill:
        kw_fb = {'facecolor':c,
            'linewidth':1,
            'alpha':0.3,
            'edgecolor':c
            }
        ax.fill_between(e, cdf_cis[0,:], cdf_cis[4,:], **kw_fb)
        kw_fb['alpha'] = 0.6
        ax.fill_between(e, cdf_cis[1,:], cdf_cis[3,:], **kw_fb)

    if hatch:
        kw_fb = {'facecolor':'none',
            'linewidth':1,
            'hatch':'///',
            'edgecolor':c
            }
        ax.fill_between(e, cdf_cis[0,:], cdf_cis[4,:], **kw_fb)
        kw_fb['hatch'] = '\\\\\\'
        ax.fill_between(e, cdf_cis[1,:], cdf_cis[3,:], **kw_fb)

    if median:
        ax.plot(e, cdf_cis[2,:], '-', lw=2, color=c)

    return

def get_kde_jsd(x, y, kw_kde={}):

    kde = KernelDensity(**kw_kde)
    kde.fit(x)
    log_p_x = kde.score_samples(x)
    log_p_y = kde.score_samples(y)
    kde.fit(y)
    log_q_x = kde.score_samples(x)
    log_q_y = kde.score_samples(y)
    log_mix_x = np.logaddexp(log_p_x, log_q_x)
    log_mix_y = np.logaddexp(log_p_y, log_q_y)
    kl_p_m = log_p_x.mean() - (log_mix_x.mean() - np.log(2))
    kl_q_m = log_q_y.mean() - (log_mix_y.mean() - np.log(2))
    js_divergence = (kl_p_m + kl_q_m) / 2.
    js_distance = np.sqrt(js_divergence)

    return js_distance

def get_dist_pdf(x, y, rng, bins, nsplit=1):

    # get pdf
    n_smp = x.shape[0] * y.shape[0]
    xsplit = np.array_split(x, nsplit)
    pdf = np.zeros(bins)
    for x0 in xsplit:
        dist = metrics.pairwise.pairwise_distances(x0, y).ravel()
        h, e = np.histogram(dist, range=rng, bins=bins)
        pdf += h
    de = e[1] - e[0]
    pdf /= (n_smp*de)

    # get maxL
    idx = np.where(pdf==np.max(pdf))
    maxL = np.mean(e[idx]) + de/2.

    # get median
    cdf = np.cumsum(pdf)*de
    median = np.interp(0.5, cdf, e[1::])

    return pdf, maxL, median

def get_run_statistics(runs, rd, fd):

    N = len(runs)

    x = table.Table()
    x['run'] = rd['run']
    x['in_group'] = rd['in_group']
    x['r'] = fd[rd['pool_idx']]['r']
    x['V_los'] = fd[rd['pool_idx']]['V_los']
    x = x.to_pandas()
    x = x.groupby(['run', 'in_group'])

    y = x.std()
    sigr0 = y.loc[(slice(0,N), 0), 'r'].values
    sigr1 = y.loc[(slice(0,N), 1), 'r'].values
    sigv0 = y.loc[(slice(0,N), 0), 'V_los'].values
    sigv1 = y.loc[(slice(0,N), 1), 'V_los'].values

    y = x.mean()
    mur0 = y.loc[(slice(0,N), 0), 'r'].values
    mur1 = y.loc[(slice(0,N), 1), 'r'].values
    muv0 = y.loc[(slice(0,N), 0), 'V_los'].values
    muv1 = y.loc[(slice(0,N), 1), 'V_los'].values

    runs['sig_r_in'] = sigr1
    runs['sig_r_out'] = sigr0
    runs['sig_v_in'] = sigv1
    runs['sig_v_out'] = sigv0
    runs['mu_r_in'] = mur1
    runs['mu_r_out'] = mur0
    runs['mu_v_in'] = muv1
    runs['mu_v_out'] = muv0

    x = table.Table()
    x['run'] = rd['run']
    x['in_group'] = rd['in_group']
    x['eps_hat'] = fd[rd['pool_idx']]['eps_hat']
    x['circ_hat'] = fd[rd['pool_idx']]['circ_hat']
    x = x.to_pandas()
    x = x.groupby(['run', 'in_group'])

    y = x.mean()
    muepsh0 = y.loc[(slice(0,N), 0), 'eps_hat'].values
    muepsh1 = y.loc[(slice(0,N), 1), 'eps_hat'].values
    mucirch0 = y.loc[(slice(0,N), 0), 'circ_hat'].values
    mucirch1 = y.loc[(slice(0,N), 1), 'circ_hat'].values
    runs['mu_eps_hat_in'] = muepsh1
    runs['mu_eps_hat_out'] = muepsh0
    runs['mu_circ_hat_in'] = mucirch1
    runs['mu_circ_hat_out'] = mucirch0

    y = x.std()
    muepsh0 = y.loc[(slice(0,N), 0), 'eps_hat'].values
    muepsh1 = y.loc[(slice(0,N), 1), 'eps_hat'].values
    mucirch0 = y.loc[(slice(0,N), 0), 'circ_hat'].values
    mucirch1 = y.loc[(slice(0,N), 1), 'circ_hat'].values
    runs['sig_eps_hat_in'] = muepsh1
    runs['sig_eps_hat_out'] = muepsh0
    runs['sig_circ_hat_in'] = mucirch1
    runs['sig_circ_hat_out'] = mucirch0
    runs['sig_iom_hat_in'] = np.sqrt(muepsh1**2 + mucirch1**2.)

    x = table.Table()
    x['run'] = rd['run']
    x['in_group'] = rd['in_group']
    x['E'] = fd[rd['pool_idx']]['E']
    x['Lz'] = fd[rd['pool_idx']]['Lz']
    x = x.to_pandas()
    x = x.groupby(['run', 'in_group'])

    y = x.mean()
    muE0 = y.loc[(slice(0,N), 0), 'E'].values
    muE1 = y.loc[(slice(0,N), 1), 'E'].values
    muLz0 = y.loc[(slice(0,N), 0), 'Lz'].values
    muLz1 = y.loc[(slice(0,N), 1), 'Lz'].values
    runs['mu_E_in'] = muE1
    runs['mu_E_out'] = muE0
    runs['mu_Lz_in'] = muLz1
    runs['mu_Lz_out'] = muLz0

    y = x.std()
    muE0 = y.loc[(slice(0,N), 0), 'E'].values
    muE1 = y.loc[(slice(0,N), 1), 'E'].values
    muLz0 = y.loc[(slice(0,N), 0), 'Lz'].values
    muLz1 = y.loc[(slice(0,N), 1), 'Lz'].values
    runs['sig_E_in'] = muE1
    runs['sig_E_out'] = muE0
    runs['sig_Lz_in'] = muLz1
    runs['sig_Lz_out'] = muLz0

    return runs

def get_run_statistics_notpaired(runs, rd, fd):

    N = len(runs)

    x = table.Table()
    x['run'] = rd['run']
    x['r'] = fd[rd['pool_idx']]['r']
    x['V_los'] = fd[rd['pool_idx']]['V_los']
    x['eps_hat'] = fd[rd['pool_idx']]['eps_hat']
    x['circ_hat'] = fd[rd['pool_idx']]['circ_hat']
    x['E'] = fd[rd['pool_idx']]['E']
    x['Lz'] = fd[rd['pool_idx']]['Lz']
    x = x.to_pandas()
    x = x.groupby(['run'])

    ystd = x.std()
    ymu = x.mean()
    for lab in ['r', 'V_los', 'eps_hat', 'circ_hat', 'E', 'Lz']:
        sig = ystd.loc[slice(0,N), lab].values
        mu = ymu.loc[slice(0,N), lab].values
        runs['sig_{0}'.format(lab)] = sig
        runs['mu_{0}'.format(lab)] = mu

    runs['sig_iom_hat'] = np.sqrt(
                        runs['sig_eps_hat']**2 +
                        runs['sig_circ_hat']**2.
                        )
    return runs

def get_pairwise_distances(tab1,
                        tab2=None,
                        cols=['mu_eps_hat', 'mu_circ_hat'],
                        flatten=True):
    n1 = len(tab1)
    x = tab1[cols[0]]
    for col in cols[1::]:
        x = np.vstack((x, [tab1[col]]))
    x = x.T

    if tab2 is None:
        dist = metrics.pairwise.pairwise_distances(x)
    else:
        n2 = len(tab2)
        y = tab2[cols[0]]
        for col in cols[1::]:
            y = np.vstack((y, [tab2[col]]))
        y = y.T
        dist = metrics.pairwise.pairwise_distances(x, y)

    if flatten:
        if tab2 is None:
            idx = np.triu_indices(n1, 1)
            dist = dist[idx]
        else:
            dist = np.ravel(dist)

    return dist

def get_pairwise_distance_percentiles(runs, rd, pc=[5, 25, 50, 75, 95]):
    Nruns = len(runs)
    for pc0 in pc:
        runs['d{0}_in'.format(pc0)] = -1.
        runs['d{0}_out'.format(pc0)] = -1.
    for irun, run in enumerate(runs):
        bb = rd[rd['run']==irun]
        for (in_grp, lab) in zip([0,1], ['out', 'in']):
            cc = bb[bb['in_group']==in_grp]
            dist = get_pairwise_distances(cc)
            dist_pc = np.percentile(dist, pc)
            for pc0, dist_pc0 in zip(pc, dist_pc):
                runs[irun]['d{0}_{1}'.format(pc0, lab)] = dist_pc0
    for pc0 in pc:
        x, y = runs['d{0}_in'.format(pc0)], runs['d{0}_out'.format(pc0)]
        runs['Del_d{0}'.format(pc0)] = x - y
    return runs

def get_pairwise_distance_percentiles_notpaired(runs,
                                            rd,
                                            pc=[5, 25, 50, 75, 95],
                                            cols=['mu_eps_hat', 'mu_circ_hat'],
                                            distance_lab='d'):
    Nruns = len(runs)
    for pc0 in pc:
        runs['{0}{1}'.format(distance_lab, pc0)] = -1.
    for irun, run in enumerate(runs):
        bb = rd[rd['run']==run['id']]
        dist = get_pairwise_distances(bb, cols=cols)
        dist_pc = np.percentile(dist, pc)
        for pc0, dist_pc0 in zip(pc, dist_pc):
            runs[irun]['{0}{1}'.format(distance_lab, pc0)] = dist_pc0
    return runs

def get_my_genlogistic_q(a, c):
    return ((a-1.)/a)**(1./c)-1

def my_genlogistic(x, pars):
    a, b, c = pars
    q = get_my_genlogistic_q(a, c)
    return a + (1.-a) * (1. + q*np.exp(-b*x))**-c

def get_genlogistic_startpars(xmed, a0=-0.1, c0=4):
    b0_set = False
    while b0_set is False:
        q0 = (((a0-1.)/a0)**(1./c0)-1)
        tmp0 = 1./q0 * (((1.-a0)/(0.5-a0))**(1./c0) - 1.)
        if tmp0>0:
            b0 = -1./xmed * np.log(tmp0)
            b0_set = True
        else:
            c0 += 0.1
    return a0, b0, c0

def ddx_my_genlogistic(x, pars):
    a, b, c = pars
    q = get_my_genlogistic_q(a, c)
    exp_mbx = np.exp(-b*x)
    return (1.-a)*b*c*q * (1. + q*exp_mbx)**-(c+1) * exp_mbx

def log_ddx_my_genlogistic(x, pars):
    a, b, c = pars
    q = (((a-1.)/a)**(1./c)-1)
    exp_mbx = np.exp(-b*x)
    return np.sum(np.log((1.-a)*b*c*q) - (c+1.)*np.log(1. + q*exp_mbx) - b*x)

def fit_my_genlogistic(dist):
    xmed = np.median(dist)
    success = False
    bounds = ((None,-1e-6), (1e-6,None), (1e-6,None))
    while success==False:
        p0 = get_genlogistic_startpars(xmed,
                                    a0=np.random.uniform(-2,-0.1,1)[0],
                                    c0=np.random.uniform(0.01,10,1)[0])
        r = minimize(lambda p:-np.sum(log_ddx_my_genlogistic(dist, p)),
                    p0,
                    bounds=bounds)
        success = r.success
    return r.x, success

def get_pairwise_distance_distribution_fits(runs, rd):
    Nruns = len(runs)
    runs['pars_in'] = np.zeros((Nruns, 3))
    runs['pars_out'] = np.zeros((Nruns, 3))
    runs['success_in'] = np.zeros(Nruns)
    runs['success_out'] = np.zeros(Nruns)
    for irun, run in enumerate(runs):
        print(irun)
        bb = rd[rd['run']==irun]
        for (in_grp, lab) in zip([0,1], ['out', 'in']):
            cc = bb[bb['in_group']==in_grp]
            dist = get_pairwise_distances(cc)
            pars, success = fit_my_genlogistic(dist)
            runs[irun]['pars_{0}'.format(lab)] = pars
            runs[irun]['success_{0}'.format(lab)] = success
    return runs

def get_pairwise_distance_distribution_fits_notpaired(runs, rd):
    Nruns = len(runs)
    runs['pars'] = np.zeros((Nruns, 3))
    runs['success'] = np.zeros(Nruns)
    for irun, run in enumerate(runs):
        print(irun)
        bb = rd[rd['run']==irun]
        dist = get_pairwise_distances(bb)
        pars, success = fit_my_genlogistic(dist)
        runs[irun]['pars'] = pars
        runs[irun]['success'] = success
    return runs

def get_f_posterior(dist, pars_in, pars_out):

    def lnlike(f, dist, pars_in, pars_out):
        L = f*ddx_my_genlogistic(dist, pars_in)
        L += (1.-f)*ddx_my_genlogistic(dist, pars_out)
        return np.sum(np.log(L))

    def lnprior(f):
        if -0 < f < 1:
            return 0.0
        return -np.inf

    def lnprob(f, dist, pars_in, pars_out):
        lp = lnprior(f)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike(f, dist, pars_in, pars_out)

    ndim, nwalkers = 1, 20
    pos = np.random.uniform(size=nwalkers)
    pos = [[pos0] for pos0 in pos]

    sampler = emcee.EnsembleSampler(nwalkers,
                                    ndim,
                                    lnprob,
                                    args=(dist, pars_in, pars_out))
    sampler.run_mcmc(pos, 200)

    return sampler

def get_flims_mrg_over_runs(dist, runs_ungrouped, runs_grouped,
                                        nwalkers=20,
                                        nsteps=500,
                                        pc=[50, 95]):

    def lnlike(f, pdf_grp, pdf_ung):
        L = f * pdf_grp
        L += (1.-f) * pdf_ung
        return np.sum(np.log(L))

    def lnprior(f):
        if -0 < f < 1:
            return 0.0
        return -np.inf

    def lnprob(f, pdf_grp, pdf_ung):
        lp = lnprior(f)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike(f, pdf_grp, pdf_ung)

    # get model likelihoods for f=1
    L_grp = np.zeros((len(dist), len(runs_grouped)))
    for idx0, x in enumerate(runs_grouped):
        L_grp[:, idx0] = ddx_my_genlogistic(dist, x['pars'])
    pdf_grp = np.mean(L_grp, 1)

    # get model likelihoods for f=0
    L_ung = np.zeros((len(dist), len(runs_ungrouped)))
    for idx0, x in enumerate(runs_ungrouped):
        L_ung[:, idx0] = ddx_my_genlogistic(dist, x['pars'])
    pdf_ung = np.mean(L_ung, 1)

    ndim = 1
    pos = np.random.uniform(size=nwalkers)
    pos = [[pos0] for pos0 in pos]
    sampler = emcee.EnsembleSampler(nwalkers,
                                    ndim,
                                    lnprob,
                                    args=(pdf_grp, pdf_ung))
    sampler.run_mcmc(pos, nsteps)
    chain = sampler.chain

    # remove burn in and flatten
    chain = np.ravel(chain[:, 100::])

    # return requested percentiles
    return sampler, np.percentile(chain, pc)

def get_likelihoods_per_run(runs, dist):
    L = table.Table()
    L['id'] = runs['id']
    L['grouped'] = runs['grouped']
    nruns, ndist = len(runs), len(dist)
    L.add_index('id')
    L['L'] = np.zeros((nruns, ndist))
    for i, x in enumerate(runs):
        L['L'][i] = ddx_my_genlogistic(dist, x['pars'])
    return L

def marginalise(x):
    return np.mean(x, 0)

def get_par_grid(grp, nbins_eps, nbins_iom):

    pc_bins = np.linspace(0, 100, nbins_eps+1)
    eps_bins = np.percentile(grp['mu_eps_hat'], pc_bins, interpolation='higher')
    # increase upper limit to include last point
    eps_bins[-1] += 1e-3

    pc_bins = np.linspace(0, 100, nbins_iom+1)
    iom_bins = np.zeros([nbins_eps, nbins_iom+1])
    for i in range(nbins_eps):
        idx = np.where(
                (grp['mu_eps_hat']>=eps_bins[i]) &
                (grp['mu_eps_hat']<eps_bins[i+1])
                )
        iom_bins[i, :] = np.percentile(grp['sig_iom_hat'][idx],
                                    pc_bins,
                                    interpolation='higher')
    # increase upper limit to include last point
    iom_bins[:, -1] += 1e-3

    tmp = table.Table()
    tmp['id'] = grp['id']
    tmp['grouped'] = grp['grouped']
    tmp.add_index('id')
    tmp['mu_eps_hat_bin'] = np.digitize(grp['mu_eps_hat'], eps_bins) - 1
    tmp['sig_iom_bin'] = -1
    for i in range(nbins_eps):
        idx = np.where(tmp['mu_eps_hat_bin']==i)
        x = grp[idx]['sig_iom_hat']
        xbins = iom_bins[i, :]
        tmp['sig_iom_bin'][idx] = np.digitize(x, xbins) - 1

    return tmp, eps_bins, iom_bins

def f_pairs(f, N=50):
    return special.comb(f*N, 2) / special.comb(N, 2)

def get_constraints_on_frac(runs,
                            dist_gcs,
                            Ngcs,
                            nbins_eps=5,
                            nbins_iom=10,
                            pc=[0.5, 0.95]):

    nbins_tot = nbins_eps * nbins_iom

    Ltab = get_likelihoods_per_run(runs, dist_gcs)
    pdf_ung = marginalise(Ltab[Ltab['grouped']==False]['L'])

    grp = runs[runs['grouped']]
    grididx, eps_bins, iom_bins = get_par_grid(grp, nbins_eps, nbins_iom)
    eps_cent = (eps_bins[0:-1] + eps_bins[1::])/2.
    iom_cent = (iom_bins[:, 0:-1] + iom_bins[:, 1::])/2.

    grididx = table.join(grididx, Ltab, 'id')
    grididx = grididx.group_by(['mu_eps_hat_bin', 'sig_iom_bin'])
    pdf_grp = grididx.groups.keys
    pdf_grp['L'] = grididx['L'].groups.aggregate(marginalise)

    # increase resolution of fbins HERE !!!!!
    f1 = np.array([0])
    f2 = np.logspace(-4, -1, 199)
    f3 = np.linspace(0.1, 1, 800)
    frac_edg = np.concatenate((f1, f2, f3[1::]))
    frac = (frac_edg[0:-1] + frac_edg[1::])/2.
    dfrac = frac[1]-frac[0]
    ftmp = frac[:, np.newaxis, np.newaxis]
    L = ftmp * pdf_grp['L'] + (1.-ftmp) * pdf_ung
    logL = np.sum(np.log(L), 2)
    logL -= np.max(logL)
    L = np.exp(logL)
    idx_maxL = np.argmax(L, axis=0)
    frac_cdf = np.cumsum(L, 0)/np.sum(L, 0)
    frac_cdf = np.vstack((np.zeros((1,nbins_tot)), frac_cdf))

    # cdf_pc_arr = np.zeros((nbins_tot, len(pc), 2))
    # for iii in range(nbins_tot):
    #     idmxL0 = idx_maxL[iii]
    #     cdf0 = frac_cdf[idmxL0+1, iii]
    #     for jjj, pc0 in enumerate(pc):
    #         if (cdf0 >= pc0/2.) & (cdf0 <= 0.5+pc0/2.):
    #             cdf_lo = cdf0 - pc0/2.
    #             cdf_hi = cdf0 + pc0/2.
    #         elif (cdf0 < pc0/2.):
    #             cdf_lo = 0.
    #             cdf_hi = pc0
    #         else:
    #             cdf_lo = 1. - pc0
    #             cdf_hi = 1.
    #         cdf_pc_arr[iii, jjj, 0] = cdf_lo
    #         cdf_pc_arr[iii, jjj, 1] = cdf_hi
    #
    # f_pair_lims = np.zeros((nbins_tot, len(pc), 2))
    # for iii in range(nbins_tot):
    #     f_pair_lims[iii, :, :] = np.interp(cdf_pc_arr[iii, :, :],
    #                                     frac_cdf[:, iii],
    #                                     frac_edg)

    # transform from fraction of pairs to fraction of members
    f2 = np.logspace(-4, -1, 199)
    f3 = np.linspace(0.1, 1, 800)
    f_arr = np.concatenate((f1, f2, f3[1::]))
    f_pairs_arr = f_pairs(f_arr, Ngcs)
    idx = np.where(f_pairs_arr==0)
    # trim excess 0's so interpolation works
    f_pairs_arr = np.delete(f_pairs_arr, idx[0][1::])
    f_arr = np.delete(f_arr, idx[0][1::])
    # f_lims = np.interp(f_pair_lims, f_pairs_arr, f_arr)

    # store results
    zzz = table.Table()

    zzz['mu_eps_hat_bin'] = pdf_grp['mu_eps_hat_bin']
    zzz['sig_iom_bin'] = pdf_grp['sig_iom_bin']

    zzz['mu_eps_hat'] = eps_cent[pdf_grp['mu_eps_hat_bin']]
    zzz['mu_eps_hat_lo'] = eps_bins[pdf_grp['mu_eps_hat_bin']]
    zzz['mu_eps_hat_hi'] = eps_bins[pdf_grp['mu_eps_hat_bin']+1]

    tmp = iom_cent[pdf_grp['mu_eps_hat_bin'], pdf_grp['sig_iom_bin']]
    zzz['sig_iom'] = tmp
    tmp = iom_bins[pdf_grp['mu_eps_hat_bin'], pdf_grp['sig_iom_bin']]
    zzz['sig_iom_lo'] = tmp
    tmp = iom_bins[pdf_grp['mu_eps_hat_bin'], pdf_grp['sig_iom_bin']+1]
    zzz['sig_iom_hi'] = tmp

    # for iii, pc0 in enumerate(pc):
    #     for jjj, lab in enumerate(['lo', 'hi']):
    #         tmp = 'flim_{0}_{1}'.format(int(100*pc0), lab)
    #         zzz[tmp] = f_lims[:, iii, jjj]

    zzz['L'] = pdf_grp['L']
    zzz['L_mrg_eps_iom'] = np.sum(L, 0)/np.sum(L)

    mrg_frac_pdf = np.sum(L, 1)
    yyy = table.Table()
    yyy['f'] = np.interp(frac, f_pairs_arr, f_arr)
    f_edg = np.interp(frac_edg, f_pairs_arr, f_arr)
    yyy['df'] = f_edg[1::] - f_edg[0:-1]
    pdf = mrg_frac_pdf
    pdf /= np.sum(pdf*yyy['df'])
    yyy['pdf'] = pdf

    return zzz, yyy, pdf_ung, Ltab

def get_confidence_limits(pdf, f, df, pc=[0.50, 0.95], maxL=True):

    argmaxL = np.argmax(pdf)
    cdf = np.cumsum(pdf*df)
    cdf_maxL = cdf[argmaxL]

    con_lims = np.zeros((len(pc), 2))
    for jjj, pc0 in enumerate(pc):
        if (cdf_maxL >= pc0/2.) & (cdf_maxL <= 0.5+pc0/2.):
            cdf_lo = cdf_maxL - pc0/2.
            cdf_hi = cdf_maxL + pc0/2.
        elif (cdf_maxL < pc0/2.):
            cdf_lo = 0.
            cdf_hi = pc0
        else:
            cdf_lo = 1. - pc0
            cdf_hi = 1.
        con_lims[jjj, 0] = cdf_lo
        con_lims[jjj, 1] = cdf_hi

    cdf = np.concatenate(([0], cdf))
    f_edg = np.concatenate(([0], f+df/2.))
    f_lims = np.interp(con_lims, cdf, f_edg)

    if maxL:
        return f_lims, f[argmaxL]
    else:
        return f_lims

def compare_two_1d_gmms(wX, muX, sigX, wY, muY, sigY, eps, umax=5):

    def Phi(x):
        # CDF of Normal(0,1^2)
        return 0.5 * (1. + special.erf(x/np.sqrt(2.)))

    def I(a, b, umax=5):
        # sqrt(pi)/2 * int_{-inf}^{inf} exp(-t^2) erf(at+b) dt
        # http://alumnus.caltech.edu/~amir/bivariate.pdf
        # n.b. need a<1 for convergence
        uarr = np.arange(umax+1)
        I1 = np.pi * (0.5 - Phi(-np.sqrt(2)*b))
        H = np.array([special.hermite(2*u+1)(b) for u in uarr])
        # vectorize
        if hasattr(a, 'shape'):
            uarr = np.broadcast_to(uarr, a.shape[::-1] + (umax+1,)).T
        summand = (a/2.)**(2*uarr+2) / special.gamma(uarr+2) * H
        I2 = np.sqrt(np.pi) * np.exp(-b**2.) * np.sum(summand, axis=0)
        return I1 - I2

    def f(muX, sigX, muY, sigY, eps, umax=5):
        # P(|x-y|<eps) where X~Norm(muX,sigX^2) and Y~Norm(muY,sigY^2)
        # vectorize
        muX, sigX = np.atleast_1d(muX), np.atleast_1d(sigX)
        muY, sigY = np.atleast_1d(muY), np.atleast_1d(sigY)
        # need sig1 < sig2 for series in I(a,b) to convergence
        mu1, sig1 = 1.*muX, 1.*sigX
        mu2, sig2 = 1.*muY, 1.*sigY
        idx = np.where(sigY < sigX)
        mu1[idx], sig1[idx] = muY[idx], sigY[idx]
        mu2[idx], sig2[idx] = muX[idx], sigX[idx]
        # calculate
        a = sig1/sig2
        b1 = (mu1 - mu2 + eps)/np.sqrt(2)/sig2
        b2 = (mu1 - mu2 - eps)/np.sqrt(2)/sig2
        prob = (I(a, b1, umax=umax) - I(a, b2, umax=umax)) / np.pi
        if prob.shape==(1,):
            return prob[0]
        return prob

    def F(wX, muX, sigX, wY, muY, sigY, eps, umax=5):
        # P(|x-y|<eps) for gaussian mixtures, i.e.
        # where X ~ sum_i wXi * Norm(muXi, sigXi^2), Y similar
        f_ij = f(muX, sigX, muY, sigY, eps, umax=umax)
        w_ij = wX * wY
        return np.sum(w_ij * f_ij)

    return F(wX, muX, sigX, wY, muY, sigY, eps, umax=umax)

class chinese_restaurant_process:

    def __init__(self, sizes, counts):

        self.sizes = [np.array(s0) for s0 in sizes]
        self.counts = [np.array(c0) for c0 in counts]
        self.n = np.sum(np.array(counts[0]) * np.array(sizes[0]))
        self.k = np.array([np.sum(c0) for c0 in counts])
        ks = np.array([len(s) for s in sizes])
        max_ks = np.max(ks)
        tmp = [np.pad(s0, (0, max_ks-len(s0)), 'constant') for s0 in sizes]
        tmp = np.ma.masked_array(tmp, mask=(tmp==0))
        self.sarr = tmp
        tmp = [np.pad(c0, (0, max_ks-len(c0)), 'constant') for c0 in counts]
        tmp = np.ma.masked_array(tmp, mask=(tmp==0))
        self.carr = tmp

    def log_pochhammer_k(self, a, b, c):
        if c==0:
            return b*np.log(a)
        else:
            return b*np.log(c) + special.gammaln(a/c+b) - special.gammaln(a/c)

    def logL(self, p, sizes, counts, k, n, return_arr=False):
        alpha, theta = p
        f = self.log_pochhammer_k(theta+alpha, k-1., alpha)
        f -= self.log_pochhammer_k(theta+1., n-1., 1.)
        summand = counts * self.log_pochhammer_k(1.-alpha, sizes-1., 1.)
        logL_i = (f + np.ma.sum(summand, 1).T).T
        if return_arr:
            return logL_i
        return np.sum(logL_i)

    def logprior(self, p):
        alpha, theta = p
        if alpha<=0:
            return -np.inf
        if alpha>=1:
            return -np.inf
        if theta<-alpha:
            return -np.inf
        return 1./(1+theta)**2.

    def logprob(self, p, sizes, counts, k, n):
        logpri = self.logprior(p)
        if logpri == -np.inf:
            return -np.inf
        else:
            return logpri + self.logL(p, sizes, counts, k, n)

    def get_maxL(self, max_attempts=1000, nc=10):
        # find maximum likelihood starting from random parameters
        # accept result when nc succusful minimizations dont lead to improvement
        nsslml = 0  # Number of Successes Since Last Max likelihood
        attempt = 0
        maxl0 = -np.inf
        while (nsslml < nc) and (attempt < max_attempts):
            alpha0 = np.random.uniform(0, 1, 1)
            theta0 = np.random.uniform(-alpha0, 1, 1)
            p0 = [alpha0, theta0]
            p = minimize(lambda p :-self.logL(p,
                                              self.sarr,
                                              self.carr,
                                              self.k,
                                              self.n),
                         p0,
                         constraints = [{'type':'ineq',
                                         'fun':lambda p: p[0]+p[1]}],
                         bounds = ((0,1), (None, None)))
            if p.success:
                if p.fun < -maxl0:
                    maxl0 = -p.fun
                    nsslml = 0
                    pbest = p
                else:
                    nsslml += 1
            attempt += 1
        if nsslml == nc:
            self.maxLalpha = pbest.x[0]
            self.maxLtheta = pbest.x[1]
            return pbest
        else:
            raise ValueError('No successful fit found')

    def run_emcee(self, nwalkers=100):
        self.nwalkers = nwalkers
        p00 = self.maxLalpha + np.random.normal(0, 0.01, self.nwalkers)
        p01 = self.maxLtheta + np.random.normal(0, 0.01, self.nwalkers)
        p = np.vstack((p00, p01)).T
        # run emcee
        ndim = 2
        sampler = emcee.EnsembleSampler(nwalkers,
                                        ndim,
                                        self.logprob,
                                        args=[self.Barr, self.k, self.n])
        sampler.run_mcmc(p, 500)
        # get maxL pars
        idx = np.argmax(sampler.lnprobability)
        idx = np.unravel_index(idx, sampler.lnprobability.shape)
        pmaxprb = sampler.chain[idx[0], idx[1], :]
        self.maxLpars_emcee = pmaxprb

    def sample(self, ntrials, alpha=None, theta=None):
        if alpha==None:
            alpha = self.maxLalpha
        if theta==None:
            theta = self.maxLtheta
        def sample_one():
            i = 0
            table_sizes = np.array([1])
            n0 = 1
            for i in range(1, self.n):
                n_tables0 = len(table_sizes)
                p_old_tables = (table_sizes - alpha)/(n0 + theta)
                p_new_table = (theta + n_tables0*alpha)/(n0 + theta)
                pmf = np.concatenate(([0], p_old_tables, [p_new_table]))
                cmf = np.cumsum(pmf)
                rnd = np.random.uniform(0, 1, 1)
                t0 = np.digitize(rnd, cmf)
                if t0 > n_tables0:
                    table_sizes = np.concatenate((table_sizes, [1]))
                else:
                    table_sizes[t0-1] += 1
                n0 += 1
            return table_sizes
        sizes, counts, Blist = [], [], []
        for i in range(ntrials):
            B0 = sample_one()
            siz, cnt = np.unique(B0, return_counts=True)
            Blist += [B0]
            sizes += [siz]
            counts += [cnt]
        return sizes, counts, Blist


class gaussian_support_lt0:
    def __init__(self, x):
        self.x = x
    def pdf(self, p, x, return_arr=True):
        mu, sig = p
        n = len(x)
        gaussian = norm(loc=mu, scale=sig)
        nrm = gaussian.cdf(0)
        likelihhod = gaussian.pdf(x)/nrm
        if return_arr:
            return likelihhod
        else:
            return np.product(likelihhod)
    def cdf(self, p, x):
        mu, sig = p
        gaussian = norm(loc=mu, scale=sig)
        nrm = gaussian.cdf(0)
        return gaussian.cdf(x)/nrm
    def logL(self, p, x):
        mu, sig = p
        n = len(x)
        gaussian = norm(loc=mu, scale=sig)
        nrm = gaussian.logcdf(0)
        logL = -n*nrm + np.sum(gaussian.logpdf(x))
        return logL
    def get_maxL(self, max_attempts=1000, nc=10):
        # find maximum likelihood starting from random parameters
        # accept result when nc succusful minimizations dont lead to improvement
        nsslml = 0  # Number of Successes Since Last Max likelihood
        attempt = 0
        maxl0 = -np.inf
        mu0, sig0 = np.mean(self.x), np.std(self.x)
        logsig0 = np.log10(sig0)
        while (nsslml < nc) and (attempt < max_attempts):
            mu00 = np.random.uniform(mu0-2*sig0, mu0+2*sig0)
            sig00 = 10.**np.random.uniform(logsig0-0.5, logsig0+0.5, 1)
            p0 = [mu00, sig00]
            p = minimize(lambda p :-self.logL(p, self.x),
                         p0,
                         bounds = ((None, None), (0, None)))
            if p.success:
                if p.fun < -maxl0:
                    maxl0 = -p.fun
                    nsslml = 0
                    pbest = p
                else:
                    nsslml += 1
            attempt += 1
        if nsslml == nc:
            self.mu = pbest.x[0]
            self.sig = pbest.x[1]
        else:
            raise ValueError('No successful fit found')


 # END
