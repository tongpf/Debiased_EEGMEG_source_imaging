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

def cal_statistics(est_sig, I_hat, fwd, true_loc, cal_rsquared = False,
                  Xtrue=None, SST_Xtrue=None, Y_true_demean=None, SST_Y=None, G_true_demean=None):
    # This function is designed to calculate the most significant sources separately in two hemispheres, which matches
    # the data generation process as listed in Table II of the paper.
    # If the sources were generated sequentially with no guarantee of the hemisphere, then the function cal_statistics_TableIII
    # should be used.
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
        
        Yhat = np.dot(G_true_demean, est_sig.reshape(S*O,T))
        Yres = Yhat - Y_true_demean
        rsquared_Y = 1 - np.linalg.norm(Yres)**2 / SST_Y

        Xhat = est_sig[peak_ix2,:,:]
        Xtrue_sel = Xtrue[not_sel_true_loc,:,:]
        Xres = Xhat - Xtrue_sel
        rsquared_X2 = 1 - np.linalg.norm(Xres)**2 / SST_Xtrue[not_sel_true_loc]
    return true_distance, rsquared_X1, rsquared_X2, rsquared_Y, estimated_locs


def cal_statistics_TableIII(est_sig, I_hat, fwd, true_loc, dipole_num=2, cal_rsquared = False,
                  Xtrue=None, SST_Xtrue=None, Y_true_demean=None, SST_Y=None, G_true_demean=None):
    # This function is designed to calculate the most significant sources sequentially, where any two adjacent sources
    # are separated by several distance. The number of sources is determined by dipole_num. 
    # This matches the data generation process as listed in Table III of the paper.
    S = est_sig.shape[0]
    O = est_sig.shape[1]
    T = est_sig.shape[2]
    est_sig_norm = np.linalg.norm(est_sig, axis = (1,2))

    maxsig = est_sig_norm[I_hat]
    peak_ix1_idx = maxsig.argmax()
    peak_ix1 = I_hat[peak_ix1_idx]
    est_loc = np.array([fwd['source_rr'][peak_ix1,:]])

    source_distances = distance.cdist(est_loc, true_loc, 'euclidean')[0,:]*1e3
    sel_loc1 = source_distances.argmin()
    true_distance1 = source_distances[sel_loc1]
    source_locs_all = fwd['source_rr']

    if dipole_num == 1:
        true_distance = true_distance1
        estimated_locs = np.array([peak_ix1])
    else:
        rm_locs1 = distance.cdist(est_loc, source_locs_all, 'euclidean')[0,:]*1e3
        rm_locs1 = np.where(rm_locs1 <= 20)[0]
        est_sig_norm[rm_locs1] = 0
        maxsig = est_sig_norm[I_hat]
        peak_ix2_idx = maxsig.argmax()
        peak_ix2 = I_hat[peak_ix2_idx]
        est_loc = np.array([fwd['source_rr'][peak_ix2,:]])

        source_distances = distance.cdist(est_loc, true_loc, 'euclidean')[0,:]*1e3
        source_distances[sel_loc1] = np.inf
        sel_loc2 = source_distances.argmin()
        true_distance2 = source_distances[sel_loc2]
        if dipole_num == 2:
            true_distance = (true_distance1 + true_distance2) / 2
            estimated_locs = np.array([peak_ix1, peak_ix2])
        else:
            rm_locs2 = distance.cdist(est_loc, source_locs_all, 'euclidean')[0,:]*1e3
            rm_locs2 = np.where(rm_locs2 <= 20)[0]
            est_sig_norm[rm_locs2] = 0
            maxsig = est_sig_norm[I_hat]
            peak_ix3_idx = maxsig.argmax()
            peak_ix3 = I_hat[peak_ix3_idx]
            est_loc = np.array([fwd['source_rr'][peak_ix3,:]])

            source_distances = distance.cdist(est_loc, true_loc, 'euclidean')[0,:]*1e3
            source_distances[sel_loc1] = np.inf
            source_distances[sel_loc2] = np.inf
            sel_loc3 = source_distances.argmin()
            true_distance3 = source_distances[sel_loc3]
            true_distance = (true_distance1 + true_distance2 + true_distance3) / 3
            estimated_locs = np.array([peak_ix1, peak_ix2, peak_ix3])

    rsquared_Y = np.nan
    rsquared_X1 = np.nan
    rsquared_X2 = np.nan
    rsquared_X3 = np.nan
    if cal_rsquared:
        Xhat = est_sig[peak_ix1,:,:]
        Xtrue_sel = Xtrue[0,:,:]
        Xres = Xhat - Xtrue_sel
        rsquared_X1 = 1 - np.linalg.norm(Xres)**2 / SST_Xtrue[0]
        
        Yhat = np.dot(G_true_demean, est_sig.reshape(S*O,T))
        Yres = Yhat - Y_true_demean
        rsquared_Y = 1 - np.linalg.norm(Yres)**2 / SST_Y

        if dipole_num > 1:
            Xhat = est_sig[peak_ix2,:,:]
            Xtrue_sel = Xtrue[1,:,:]
            Xres = Xhat - Xtrue_sel
            rsquared_X2 = 1 - np.linalg.norm(Xres)**2 / SST_Xtrue[1]

            if dipole_num > 2:
                Xhat = est_sig[peak_ix3,:,:]
                Xtrue_sel = Xtrue[2,:,:]
                Xres = Xhat - Xtrue_sel
                rsquared_X3 = 1 - np.linalg.norm(Xres)**2 / SST_Xtrue[2]

    return true_distance, rsquared_X1, rsquared_X2, rsquared_X3, rsquared_Y, estimated_locs
