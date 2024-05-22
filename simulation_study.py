import os.path as op
import pandas as pd
import numpy as np
import random
import math
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy
from scipy.spatial import distance
import mne
from mne.datasets import fetch_fsaverage
from ADMM_GroupLASSO import gl_ADMM_dual_joint, _cal_I_hat, gl_ADMM_dual_bias_correction, _Gmatrix_to_tensor, \
    _Gtensor_covariance
from data_utils import AR_series, cal_statistcs

# Set MNE data directory.
print(mne.get_config())
#mne.set_config("SUBJECTS_DIR", "your path to the subjects directory")

# Download fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)

# The files live in:
subject = "fsaverage"
trans = "fsaverage"  # MNE has a built-in fsaverage transformation
# create a source space with desired density been oct5 (1026 sources per hemisphere)
src = mne.setup_source_space(subject, spacing='oct5',
                             subjects_dir=subjects_dir,n_jobs=-1)
bem = op.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")

# create a dummy raw from zero
GSN_HydroCel_128_montage = mne.channels.make_standard_montage("GSN-HydroCel-128")
ch_names = GSN_HydroCel_128_montage.ch_names
data = np.zeros((len(ch_names), 700)) # 7 seconds long epochs
info = mne.create_info(ch_names, 100, "eeg") # sampling rate 100 Hz
raw = mne.io.RawArray(data, info)

# Read and set the EEG electrode locations, which are already in fsaverage's
# space (MNI space) for standard_1020:
raw.set_montage(GSN_HydroCel_128_montage)
raw.set_eeg_reference(projection=True)  # needed for inverse modeling

# Check that the locations of EEG electrodes is correct with respect to MRI
# mne.viz.plot_alignment(raw.info, src=src, eeg=["projected"], trans=trans, show_axes=True, mri_fiducials=True,
#     dig="fiducials", surfaces=("white", "outer_skin", "inner_skull", "outer_skull"))

# Compute forward solution
fwd = mne.make_forward_solution(
    raw.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0
)

N = raw._data.shape[0]
T = raw._data.shape[1]//7 # sampling rate 100 Hz, 7 seconds long epochs
G = fwd["sol"]["data"] # gain matrix
O = 3
S = G.shape[1]//O
dipole_num = 2

result_pd = pd.DataFrame(columns=['sim_num','method', 'lambda', 'snr_n', 'snr_s', 'Distance', 'R2_X1', 'R2_X2', 'R2_Y'])
row_num = 0

# number of simulations
sim_num = 0
for sim_num in range(10):
    X_raw = np.zeros((S*O, T*7))
    EX_raw = np.random.randn(S*O, T*7) # noise applied in source locations
    E_raw = np.random.randn(N, T*7) # noise applied in sensor locations
    select_loc1 = random.sample(range(S//2), 1)
    select_loc2 = random.sample(range(S//2), 1)
    selected_locs = [select_loc1[0],select_loc2[0]+S//2] # select two sources in each hemisphere
    # generate two active signals from 5s to 7s
    for start_idx in range(len(selected_locs)):
        start = selected_locs[start_idx]
        amp = AR_series(5,2*T) # generate AR(5) series as signal amplitude
        ori = np.random.randn(O)
        ori = ori/np.linalg.norm(ori) # generate random orientation
        X_raw[(start*O):(start*O+O), (5*T):] = np.outer(ori, amp)*1e-7 # in MNE, the data unit is volt for EEG
    selected_locs_full = np.concatenate([np.arange(start*O, start*O+O, dtype = 'int') for start in selected_locs])

    # generate eight background signals
    X_raw_backgroud = np.zeros((S*O, T*7))
    backgroud_locs = list(range(S))
    backgroud_locs = [loc for loc in backgroud_locs if loc not in selected_locs]
    backgroud_locs = random.sample(backgroud_locs, 8)
    backgroud_locs.sort()

    for start_idx in range(len(backgroud_locs)):
        start = backgroud_locs[start_idx]
        amp = AR_series(5,T*7)
        ori = np.random.randn(O)
        ori = ori/np.linalg.norm(ori)
        X_raw_backgroud[(start*O):(start*O+O), :] = np.outer(ori, amp)*1e-7

    # the minimum signal energy in the active sources
    X_raw_min = np.linalg.norm(X_raw[selected_locs_full,(5*T):].reshape(dipole_num,O,2*T), axis = (1,2))
    X_raw_min = X_raw_min.min()

    # the maximum signal energy in the background sources
    X_raw_backgroud_max = np.linalg.norm(X_raw_backgroud[:,(5*T):].reshape(S,O,2*T), axis = (1,2))
    X_raw_backgroud_max = X_raw_backgroud_max[backgroud_locs]
    X_raw_backgroud_max = X_raw_backgroud_max.max()

    # scale the background sources to half power level as the active sources
    X_raw_backgroud = X_raw_backgroud / X_raw_backgroud_max * X_raw_min / np.sqrt(2)
    X_raw = X_raw + X_raw_backgroud

    energy_X = np.linalg.norm(X_raw[:,(5*T):])**2 / (2*T * X_raw.shape[0])
    energy_EX = np.linalg.norm(EX_raw[:,(5*T):])**2 / (2*T * EX_raw.shape[0])

    snr_s = 20
    for snr_s in [20, 10, 0]:
        # scale the noise level
        energy_EX_target = energy_X / (10**(snr_s/10))
        EX_adj = np.sqrt(energy_EX_target / energy_EX)
        EX = EX_raw.copy() * EX_adj

        X = X_raw.copy() + EX
        
        #save the X values in active periods
        Xtrue = X[selected_locs_full,(5*T):].reshape((dipole_num, O, 2*T))
        #save the active source locations
        true_loc = fwd['source_rr'][selected_locs,:]

        # Y = G @ X + E
        Y_raw = np.dot(G, X)

        energy_Y = np.linalg.norm(Y_raw[:,(5*T):])**2 / (2*T * Y_raw.shape[0])
        energy_E = np.linalg.norm(E_raw[:,(5*T):])**2 / (2*T * E_raw.shape[0])

        snr_n = 20
        for snr_n in [20, 10, 0, -10]:
            # scale the noise level
            energy_E_target = energy_Y / (10**(snr_n/10))
            E_adj = np.sqrt(energy_E_target / energy_E)
            E = E_raw.copy() * E_adj

            raw._data = E.copy() + Y_raw.copy()
            # save the Y values in active periods
            # by applying montages, the data is demeaned across channels
            Y = raw._data[:, (5*T):].copy()

            # generate epochs
            # our AR process has no intercept, thus the mean of the data is zero
            epochs = mne.Epochs(raw, np.array([[0,0,0]]), event_id={'signal start':0}, baseline=[0,4.99],
                tmin=-0, tmax=6.99, preload=True)

            # compute noise covariance matrix
            cov = mne.compute_covariance(epochs, method='empirical', tmin=0, tmax=4.99)
            # compute noise+signal covariance matrix
            cov_all = mne.compute_covariance(epochs, method='empirical', tmin=5, tmax=6.99)

            # generate the evoked data
            evoked = epochs.average()

            # plot the evoked data
            evoked = evoked.crop(3,6.99)
            fig, axs = plt.subplots(1, 1, figsize=(4.6,2.4))
            evoked.plot(axes = axs,spatial_colors=True,highlight=np.array([5.0,6.99]),show = False,
                        ylim = {'eeg':[-150, 150]},
                        titles=rf'$SNR_n={snr_n}dB, SNR_s={snr_s}dB$')
            # remove the "N_ave" annotation
            for text in list(axs.texts):
                text.remove()
            fig.subplots_adjust(left=0.15, right=0.99, bottom=0.19, top=0.9, wspace=0.1, hspace=0.5)
            fig.savefig(f'img/simulated_evoked_snr_n={snr_n}dbsnr_s={snr_s}db.png',dpi=300)
            plt.clf()

            # only use the active periods
            evoked = evoked.crop(5,6.99)

            # save the true values
            # In multivariate regression, add intercept is equivalent to demean Y and G
            # thus, the true model can be written as Y = 1 @ a.T + G @ X + E with an intercept a
            Y_true_demean = evoked.data
            G_true_demean = fwd['sol']['data'] - np.mean(fwd['sol']['data'], axis=0, keepdims=True)

            Xtrue_mean = Xtrue.mean(axis = 2)
            Y_true_demean_mean = Y_true_demean.mean(axis = 1)

            SST_Xtrue = np.linalg.norm(Xtrue - Xtrue_mean[:,:,np.newaxis],axis = (1,2))**2
            SST_Y = np.linalg.norm(Y_true_demean - Y_true_demean_mean[:,np.newaxis])**2
            
            for mymethod in ['DeESI']:
                if mymethod == 'DeESI':
                    G_mat = fwd['sol']['data'] # gain matrix
                    Y_mat = evoked.data # measurements
                    G_mat = G_mat - np.mean(G_mat, axis=0, keepdims=True) # demean G
                    Y_mat = Y_mat - np.mean(Y_mat, axis=0, keepdims=True) # demean Y

                    # data whitening using noise covariance matrix
                    # first, rescale the G and Y to have unit scale in each element
                    G_mat_adj1 = np.linalg.norm(G_mat) / math.sqrt(G_mat.shape[1] * G_mat.shape[0])
                    Y_mat_adj1 = np.linalg.norm(Y_mat) / math.sqrt(Y_mat.shape[1] * Y_mat.shape[0])
                    G_mat = G_mat / G_mat_adj1
                    Y_mat = Y_mat / Y_mat_adj1

                    # covariance matrix should also be rescaled
                    cov_adj = cov['data'].copy() / Y_mat_adj1**2
                    # whitenning by cov_adj**(-1/2)
                    C_eigenvalues, C_eigenvectors = np.linalg.eigh(cov_adj)
                    C_eigenvalues[C_eigenvalues < 1e-4] = 0 # covariance matrix is estimated by 100 samples, 
                                                            # thus is not full rank
                    C_eigenvalues_sqrt_inv = C_eigenvalues.copy()
                    C_eigenvalues_sqrt_inv[C_eigenvalues_sqrt_inv>0] = 1 / np.sqrt(C_eigenvalues[C_eigenvalues>0])
                    C_sqrt_inv = C_eigenvectors @ np.diag(C_eigenvalues_sqrt_inv) @ C_eigenvectors.T # cov_adj**(-1/2)
                    Y_mat = C_sqrt_inv@Y_mat
                    G_mat = C_sqrt_inv@G_mat

                    # rescale the G and Y again after whitening
                    G_mat_adj2 = np.linalg.norm(G_mat) / math.sqrt(G_mat.shape[1] * G_mat.shape[0])
                    Y_mat_adj2 = np.linalg.norm(Y_mat) / math.sqrt(Y_mat.shape[1] * Y_mat.shape[0])
                    G_mat = G_mat / G_mat_adj2
                    Y_mat = Y_mat / Y_mat_adj2

                    G_mat_adj = G_mat_adj1 * G_mat_adj2
                    Y_mat_adj = Y_mat_adj1 * Y_mat_adj2

                    # convert G_mat to tensor with shape (N,O,S)
                    G_tensor = _Gmatrix_to_tensor(G_mat, O, block_mathod='consecutive')
                    # compute the O*O covariance matrix of G_tensor in each source location
                    sigma_G_sqrt, sigma_G_sqrt_inv = _Gtensor_covariance(G_tensor)
                    # W_s = sigma_G_sqrt_inv
                    wlist = [sigma_G_sqrt, sigma_G_sqrt_inv]

                    # use LCMV as prior, which may be regarded as a soft thresholded sure independence screening 
                    filters = mne.beamformer.make_lcmv(evoked.info, fwd, cov_all, pick_ori="vector", reg=0.5,
                                                       noise_cov=cov, rank=None,depth=1,verbose=False)
                    stc = mne.beamformer.apply_lcmv(evoked, filters,verbose=False)
                    wlist1d = np.linalg.norm(stc.data, axis = (1,2))
                    wlist1d = wlist1d.min() / wlist1d  # which makes the maximum value to be 1
                    wlist[0] = wlist[0] * wlist1d[np.newaxis,np.newaxis,:]
                    wlist[1] = wlist[1] / wlist1d[np.newaxis,np.newaxis,:]  # the panalty will be larger for unlikely sources

                    X0 = np.zeros((S*O, 2*T))

                    for lambda_i in [200,100,75,50,35,25,20,15,10]:
                        # calculate estimated X without bias correction
                        Xout, lamt, sigma_list = \
                            gl_ADMM_dual_joint(10,X0,G_mat,Y_mat,lambda_i,O,wlist = wlist,block_mathod='consecutive',
                                               tol = 1e-5,tol_norm=1e-5,max_iter = 4000, varing_rho = True)
                        # I_hat is the estimated active source locations
                        I_hat, G_tensor, Xt_tensor = _cal_I_hat(Xout, G_mat, O, block_mathod='consecutive')
                        estnum = I_hat.shape[0]
                        print(sigma_list)
                        print('estnum = %d' % estnum)
                        print('lambda = %f' % lambda_i)

                        X0 = Xout.copy() # the warm start for the next iteration

                        # if the estimated number of active sources less than 2, skip this lambda
                        if estnum<2:
                            continue
                        elif estnum>N/O:#if the estimated number of active sources is larger than the number of sensors,
                                          #stop the iteration
                            break

                        # calculate the debiased X with set A been I_hat
                        Xout_debias, Xout_debias_rotate, significance_list = \
                            gl_ADMM_dual_bias_correction(I_hat, Xt_tensor, G_tensor, Y_mat, lambda_i, O,wlist = 'auto', 
                                                         bias_correction_method = 'joint', clear_not_select=False,
                                                         block_mathod='consecutive', tol = 1e-5,tol_norm=1e-5,
                                                         max_iter = 4000,
                                                         varing_rho=True)

                        Xt_tensor = Xt_tensor * Y_mat_adj / G_mat_adj
                        Xout_debias = Xout_debias * Y_mat_adj / G_mat_adj

                        Xout_debias = Xout_debias.reshape((S,O,2*T))
                        Xout_debias_rotate = Xout_debias_rotate.reshape((S,O,2*T))

                        # calculate statistics for the uncorrected X
                        true_distance, rsquared_X1, rsquared_X2, rsquared_Y, estimated_locs_raw = \
                            cal_statistcs(Xt_tensor, I_hat, fwd, true_loc, cal_rsquared = True, Xtrue=Xtrue, 
                                          SST_Xtrue=SST_Xtrue, Y_true_demean=Y_true_demean, SST_Y=SST_Y, 
                                          G_true_demean=G_true_demean)
                        result_pd.loc[row_num] = [sim_num, 'DeESI_raw', lambda_i, snr_n, snr_s, true_distance, 
                                                  rsquared_X1, rsquared_X2, rsquared_Y]
                        row_num += 1
                        if row_num % 10 == 0:
                            print('row_num = %d' % row_num)
                        
                        # calculate statistics for the debiased X
                        true_distance, rsquared_X1, rsquared_X2, rsquared_Y, estimated_locs = \
                            cal_statistcs(Xout_debias, I_hat, fwd, true_loc, cal_rsquared = True, Xtrue=Xtrue, 
                                          SST_Xtrue=SST_Xtrue, Y_true_demean=Y_true_demean, SST_Y=SST_Y, 
                                          G_true_demean=G_true_demean)
                        result_pd.loc[row_num] = [sim_num, 'DeESI_debias', lambda_i, snr_n, snr_s, true_distance, 
                                                  rsquared_X1, rsquared_X2, rsquared_Y]
                        row_num += 1
                        if row_num % 10 == 0:
                            print('row_num = %d' % row_num)

                        # calculate statistics for the signifiance based location estimate
                        true_distance, rsquared_X1, rsquared_X2, rsquared_Y, estimated_locs = \
                            cal_statistcs(Xout_debias_rotate, I_hat, fwd, true_loc, cal_rsquared = False)
                        estimated_locs.sort()
                        Xt_tensor = Xt_tensor / Y_mat_adj * G_mat_adj
                        Xout_debias, Xout_debias_rotate, significance_list = \
                            gl_ADMM_dual_bias_correction(estimated_locs, Xt_tensor, G_tensor, Y_mat, lambda_i, 
                                                         O,wlist = 'auto', bias_correction_method = 'joint', 
                                                         clear_not_select=False, block_mathod='consecutive', tol = 1e-5,
                                                         tol_norm=1e-5, max_iter = 4000, varing_rho=True)
                        Xout_debias = Xout_debias * Y_mat_adj / G_mat_adj
                        Xout_debias = Xout_debias.reshape((S,O,2*T))

                        true_distance, rsquared_X1, rsquared_X2, rsquared_Y, estimated_locs = \
                            cal_statistcs(Xout_debias, estimated_locs, fwd, true_loc, cal_rsquared = True, Xtrue=Xtrue, 
                                          SST_Xtrue=SST_Xtrue, Y_true_demean=Y_true_demean, SST_Y=SST_Y, 
                                          G_true_demean=G_true_demean)
                        result_pd.loc[row_num] = [sim_num, 'DeESI_std', lambda_i, snr_n, snr_s, true_distance, 
                                                  rsquared_X1, rsquared_X2, rsquared_Y]
                        row_num += 1
                        if row_num % 10 == 0:
                            print('row_num = %d' % row_num)

                        # calculate the debiased X with set A been estimated_locs_raw, 
                        # the most significant locations in the uncorrected X
                        #Xt_tensor = Xt_tensor / Y_mat_adj * G_mat_adj
                        Xout_debias, Xout_debias_rotate, significance_list = \
                            gl_ADMM_dual_bias_correction(estimated_locs_raw, Xt_tensor, G_tensor, Y_mat, lambda_i, 
                                                         O,wlist = 'auto', bias_correction_method = 'joint', 
                                                         clear_not_select=False, block_mathod='consecutive', tol = 1e-5,
                                                         tol_norm=1e-5, max_iter = 4000, varing_rho=True)
                        Xout_debias = Xout_debias * Y_mat_adj / G_mat_adj
                        Xout_debias = Xout_debias.reshape((S,O,2*T))
                        # calculate statistics for the debiased X
                        true_distance, rsquared_X1, rsquared_X2, rsquared_Y, estimated_locs = \
                            cal_statistcs(Xout_debias, estimated_locs_raw, fwd, true_loc, cal_rsquared = True, 
                                          Xtrue=Xtrue, SST_Xtrue=SST_Xtrue, Y_true_demean=Y_true_demean, SST_Y=SST_Y, 
                                          G_true_demean=G_true_demean)
                        result_pd.loc[row_num] = [sim_num, 'DeESI_rawselected_debias', lambda_i, snr_n, snr_s, 
                                                  true_distance, rsquared_X1, rsquared_X2, rsquared_Y]
                        row_num += 1
                        if row_num % 10 == 0:
                            print('row_num = %d' % row_num)
                
            result_pd.to_csv('simulation.csv') # save the results after each simulation


