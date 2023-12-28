import numpy as np
import statsmodels.api as sm
from scipy.spatial import distance

def AR_series(p,T):
    # generate stationary AR(p) series
    stationary = False
    while not stationary:
        arparams1 = np.random.uniform(0.2, 0.9)
        arparams1 = arparams1 * np.random.choice([1, -1])
        arparams1 = np.array([arparams1])
        arparams = np.random.randn(p-1)*0.2
        # check whether all roots are outside the unit circle (if the polynomial is in decreasing order, 
        # then the roots should be inside the unit circle)
        stationary = abs(np.roots(np.concatenate([np.array([1]),-arparams1, -arparams]))).max() < 1
    
    maparams = np.array([0])
    ar = np.r_[1, -arparams1, -arparams] # add zero-lag and negate
    ma = np.r_[1, maparams] # add zero-lag
    amp = sm.tsa.arma_generate_sample(ar, ma, T, burnin=5*T)
    return amp

def triangle(length, amplitude):
    section = length // 4
    for direction in (1, -1):
        for i in range(section):
            yield i * (amplitude / section) * direction
        for i in range(section):
            yield (amplitude - (i * (amplitude / section))) * direction

def cal_statistcs(est_sig, I_hat, fwd, true_loc, cal_rsquared = False,
                  Xtrue=None, SST_Xtrue=None, Y_true_demean=None, SST_Y=None, G_true_demean=None):
    S = est_sig.shape[0]
    O = est_sig.shape[1]
    T = est_sig.shape[2]
    est_sig_norm = np.linalg.norm(est_sig, axis = (1,2))

    maxsig = est_sig_norm[I_hat]
    peak_ix1_idx = maxsig.argmax()
    peak_ix1 = I_hat[peak_ix1_idx]
    est_loc = np.array([fwd['source_rr'][peak_ix1,:]])

    source_distances = distance.cdist(est_loc, true_loc, 'euclidean')[0,:]*1e3
    if peak_ix1 < S//2:
        sel_true_loc = 0
        not_sel_true_loc = 1
    else:
        sel_true_loc = 1
        not_sel_true_loc = 0
    true_distance = source_distances[sel_true_loc]

    # peak two most significant sources seperately in two hemispheres
    if peak_ix1 < S//2:
        source_two_idx = np.where(I_hat >= S//2)[0]
    else:
        source_two_idx = np.where(I_hat < S//2)[0]
    if source_two_idx.size == 0:
        source_two_idx = np.setdiff1d(np.arange(maxsig.size), peak_ix1_idx)
    peak_ix2_idx = source_two_idx[maxsig[source_two_idx].argmax()]
    peak_ix2 = I_hat[peak_ix2_idx]
    est_loc = np.array([fwd['source_rr'][peak_ix2,:]])
    source_distances = distance.cdist(est_loc, true_loc, 'euclidean')[0,:]*1e3
    true_distance = (true_distance + source_distances[not_sel_true_loc]) / 2

    rsquared_Y = np.nan
    rsquared_X1 = np.nan
    rsquared_X2 = np.nan
    estimated_locs = np.array([peak_ix1, peak_ix2])
    if cal_rsquared:
        Xhat = est_sig[peak_ix1,:,:]
        Xtrue_sel = Xtrue[sel_true_loc,:,:]
        Xres = Xhat - Xtrue_sel
        rsquared_X1 = 1 - np.linalg.norm(Xres)**2 / SST_Xtrue[sel_true_loc]
        source_id_list_full = np.concatenate([np.arange(start*O, start*O+O, dtype = 'int') for start in I_hat])

        Yhat = np.dot(G_true_demean[:,source_id_list_full], est_sig[I_hat,:,:].reshape(I_hat.shape[0]*O,T))
        Yres = Yhat - Y_true_demean
        rsquared_Y = 1 - np.linalg.norm(Yres)**2 / SST_Y

        Xhat = est_sig[peak_ix2,:,:]
        Xtrue_sel = Xtrue[not_sel_true_loc,:,:]
        Xres = Xhat - Xtrue_sel
        rsquared_X2 = 1 - np.linalg.norm(Xres)**2 / SST_Xtrue[not_sel_true_loc]
    return true_distance, rsquared_X1, rsquared_X2, rsquared_Y, estimated_locs