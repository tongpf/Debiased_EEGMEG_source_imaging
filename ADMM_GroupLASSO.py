import numpy as np
import random
import math
import scipy

def gl_ADMM_dual_joint(joint_est_num, X0, G, Y, lam, O, wlist=None, 
                       block_mathod='consecutive', tol=1e-8,tol_norm=1e-4, max_iter=1000, varing_rho=False):
    N = G.shape[0]
    S = int(G.shape[1]/O)
    T = Y.shape[1]
    if wlist is None:
        I_O = np.eye(O)
        sigma_G_sqrt = np.repeat(I_O[:, :, np.newaxis], S, axis=2)
        sigma_G_sqrt_inv = sigma_G_sqrt
        wlist = [sigma_G_sqrt, sigma_G_sqrt_inv]
    elif wlist == 'auto':
        G_tensor = _Gmatrix_to_tensor(G, O, block_mathod=block_mathod)
        sigma_G_sqrt, sigma_G_sqrt_inv = _Gtensor_covariance(G_tensor)
        wlist = [sigma_G_sqrt, sigma_G_sqrt_inv]
    
    sigma_list = []
    lam_new = lam * np.linalg.norm(Y - np.dot(G, X0)) / math.sqrt(N*T)
    Xt = X0.copy()
    sigma_hat_pre = math.inf
    
    for _ in range(joint_est_num):
        #lam_use = lam_new*(math.sqrt(O*T))*math.sqrt(T*N) #lambda propto T*sqrt(NO)
        lam_use = lam_new*(math.sqrt(T)) #lambda propto sqrt(T), it seems that lambda may depend on N
        Xt = gl_ADMM_dual(Xt,G,Y,lam_use,O,wlist,block_mathod=block_mathod,
                          tol = tol,tol_norm=tol_norm,max_iter = max_iter,varing_rho=varing_rho)
        sigma_hat = np.linalg.norm(Y - np.dot(G, Xt)) / math.sqrt(N*T)
        lam_new = lam * sigma_hat

        sigma_list.append(sigma_hat)

        if abs(sigma_hat - sigma_hat_pre) < tol:
            break
        else:
            sigma_hat_pre = sigma_hat
    return Xt, lam_new, sigma_list

def _Gmatrix_to_tensor(G, O, block_mathod='consecutive'):
    N = G.shape[0]
    S = int(G.shape[1]/O)
    if block_mathod == 'jump':
        G_tensor = np.reshape(G.copy(), (N, O, S))
    elif block_mathod == 'consecutive':
        G_tensor = np.reshape(G.copy(), (N, O, S),order='F')
    return G_tensor

def _Xmatrix_to_tensor(X, O, block_mathod='consecutive'):
    S = int(X.shape[0]/O)
    T = X.shape[1]
    if block_mathod == 'jump':
        X_tensor = np.reshape(X.copy(), (S, O, T),order='F')
    elif block_mathod == 'consecutive':
        X_tensor = np.reshape(X.copy(), (S, O, T))
    return X_tensor

def _cal_I_hat(Xt, G, O, block_mathod='consecutive', threshold=0):
    """
    Calculate the index of non-zero sources
    Reshape the data into tensor format
    """
    N = G.shape[0]
    S = int(G.shape[1]/O)
    T = Xt.shape[1]
    Xtnorm = np.linalg.norm(Xt, axis=1)

    # reshape
    if block_mathod == 'jump':
        G_tensor = np.reshape(G.copy(), (N, O, S))
        Xt_tensor = np.reshape(Xt.copy(), (S, O, T),order='F')
    elif block_mathod == 'consecutive':
        G_tensor = np.reshape(G.copy(), (N, O, S),order='F')
        Xt_tensor = np.reshape(Xt.copy(), (S, O, T))
    
    # calculate the norm of each block
    if O > 1:
        if block_mathod == 'jump':
            block_norm = np.sqrt(Xtnorm[0:S]**2+Xtnorm[(S):(2*S)]**2+Xtnorm[(2*S):(3*S)]**2)
        elif block_mathod == 'consecutive':
            block_norm = np.sqrt(Xtnorm[0::O]**2+Xtnorm[1::O]**2+Xtnorm[2::O]**2)
        I_hat = np.where(block_norm > threshold)[0] 
    else:
        I_hat = np.where(Xtnorm > threshold)[0]
    
    return I_hat, G_tensor, Xt_tensor
    
def _Gtensor_covariance(G_tensor,depth=1):
    sigma_G = np.einsum('kil, kjl->ijl',G_tensor,G_tensor) / G_tensor.shape[0]
    sigma_G_eigenvalues, sigma_G_eigenvectors = np.linalg.eigh(sigma_G.transpose(2, 0, 1))
    sigma_G_eigenvalues = sigma_G_eigenvalues**depth
    sigma_G_eigenvalues = sigma_G_eigenvalues.transpose(1, 0)
    sigma_G_eigenvectors = sigma_G_eigenvectors.transpose(1, 2, 0)
    sigma_G_eigenvalues_sqrt = np.sqrt(sigma_G_eigenvalues)
    sigma_G_eigenvalues_sqrt_inv = 1/sigma_G_eigenvalues_sqrt
    sigma_G_sqrt = np.einsum('ijl,jl,kjl->ikl',sigma_G_eigenvectors,sigma_G_eigenvalues_sqrt,sigma_G_eigenvectors)
    sigma_G_sqrt_inv = np.einsum('ijl,jl,kjl->ikl',sigma_G_eigenvectors,sigma_G_eigenvalues_sqrt_inv,sigma_G_eigenvectors)
    return sigma_G_sqrt, sigma_G_sqrt_inv

def gl_ADMM_dual_bias_correction(I_hat_all, Xt_tensor, G_tensor, Y, lam, O, wlist='auto', 
                                 bias_correction_method = 'joint', clear_not_select = False, block_mathod='consecutive', 
                                 tol=1e-8,tol_norm=1e-4, max_iter=1000,varing_rho=True):
    Xout_debias = Xt_tensor.copy()
    neg_I_hat = np.setdiff1d(np.arange(int(G_tensor.shape[-1]/1)), I_hat_all)
    significance_list = np.zeros((int(G_tensor.shape[-1]/1))) # the p-value
    S = int(G_tensor.shape[-1])
    T = Y.shape[1]

    if wlist == 'auto':
        sigma_G_sqrt, sigma_G_sqrt_inv = _Gtensor_covariance(G_tensor)
    else:
        sigma_G_sqrt = wlist[0]
        sigma_G_sqrt_inv = wlist[1]
    G_tilde = G_tensor.copy()
    G_tilde[:,:,neg_I_hat] = np.einsum('ikl, kjl->ijl',G_tensor[:,:,neg_I_hat], 
                                       sigma_G_sqrt_inv[:,:,neg_I_hat]) # only the G_{,neg_I_hat} need to be whitened

    if clear_not_select:
        Xout_debias[neg_I_hat, :] = 0
        Xout_debias_rotate = Xout_debias.copy()
        Xt_tilde = Xout_debias.copy()
        Xt_tilde[neg_I_hat,:,:] = np.einsum('ijl, ljk->lik', sigma_G_sqrt[:,:,neg_I_hat], Xout_debias[neg_I_hat,:,:])
    else:
        Xout_debias_rotate = Xout_debias.copy()
        Xout_debias_rotate[neg_I_hat, :] = 0
        Xt_tilde = Xt_tensor.copy()
        Xt_tilde[neg_I_hat,:,:] = np.einsum('ijl, ljk->lik', sigma_G_sqrt[:,:,neg_I_hat], Xt_tensor[neg_I_hat,:,:])
    
    if bias_correction_method == 'joint':
        I_hat = I_hat_all
        X_hat_debias_idx, X_hat_debias_rotate_idx, debias_flag_idx, effect_mat = \
            _gl_ADMM_dual_bias_correction(I_hat, Xt_tilde, G_tilde, Y, lam, O, 
                                          tol=tol,tol_norm=tol_norm, max_iter=max_iter,varing_rho=varing_rho)
        Xout_debias[I_hat, :, :] = np.reshape(X_hat_debias_idx.copy(), (I_hat.shape[0], O, T))
        Xout_debias_rotate[I_hat, :] = np.reshape(X_hat_debias_rotate_idx.copy(), (I_hat.shape[0], O, T))
        if debias_flag_idx:
            effect_mat = _Gmatrix_to_tensor(effect_mat, O)
            significance_list[I_hat] = np.linalg.norm(np.einsum('ijl, ljk->lik', effect_mat, Xout_debias[I_hat, :, :]),axis=(1,2))
            #np.linalg.norm(effect_mat @ X_hat_debias_idx)
    elif bias_correction_method == 'seperate':
        for I_hat in I_hat_all:
            I_hat = np.array([I_hat])
            X_hat_debias_idx, X_hat_debias_rotate_idx, debias_flag_idx, effect_mat = \
                _gl_ADMM_dual_bias_correction(I_hat, Xt_tilde, G_tilde, Y, lam, O, 
                                              tol=tol,tol_norm=tol_norm, max_iter=max_iter,varing_rho=varing_rho)
            Xout_debias[I_hat, :] = np.reshape(X_hat_debias_idx.copy(), (I_hat.shape[0], O, T))
            Xout_debias_rotate[I_hat, :] = np.reshape(X_hat_debias_rotate_idx.copy(), (I_hat.shape[0], O, T))
            if debias_flag_idx:
                significance_list[I_hat] = np.linalg.norm(effect_mat @ X_hat_debias_idx)

    # Xout_debias = np.einsum('ijl, ljk->lik', sigma_G_sqrt_inv, Xout_debias)
    # Xout_debias_rotate = np.einsum('ijl, ljk->lik', sigma_G_sqrt_inv, Xout_debias_rotate)
    if block_mathod == 'jump':
        Xout_debias = np.reshape(Xout_debias, (S*O,T), order='F')
        Xout_debias_rotate = np.reshape(Xout_debias_rotate, (S*O,T), order='F')
    elif block_mathod == 'consecutive':
        Xout_debias = np.reshape(Xout_debias, (S*O,T))
        Xout_debias_rotate = np.reshape(Xout_debias_rotate, (S*O,T))
    return Xout_debias, Xout_debias_rotate, significance_list

def _gl_ADMM_dual_bias_correction(I_hat, Xt_tilde, G_tilde, Y, lam, O,
                       tol=1e-8, tol_norm=1e-4, max_iter=1000,varing_rho=True):
    N = G_tilde.shape[0]
    S = int(G_tilde.shape[-1])
    T = Y.shape[1]

    if I_hat.shape[0] > 0 and I_hat.shape[0] < N/O:
        debias_flag = True
        neg_I_hat = np.setdiff1d(np.arange(S), I_hat)
        G_I = G_tilde[:, :, I_hat]
        G_neg_I = G_tilde[:, :, neg_I_hat]
        X_neg_I = Xt_tilde[neg_I_hat, :, :]
        G_neg_I_flat = np.reshape(G_neg_I, (N, (S-I_hat.shape[0])*O),order='F')
        G_I_flat = np.reshape(G_I, (N, I_hat.shape[0]*O),order='F')
        X_neg_I_flat = np.reshape(X_neg_I, ((S-I_hat.shape[0])*O, T))
    else:
        debias_flag = False

    if debias_flag:
        B0 = np.zeros(((S-I_hat.shape[0])*O, I_hat.shape[0]*O))

        lam_res = lam*(math.sqrt((I_hat.shape[0]+1)*O)+math.sqrt(2*math.log(S))) # sqrt((A+1)*O)+sqrt(2*log(S)), silimar to sqrt(T)+sqrt(2*log(S))
        Bt = gl_ADMM_dual(B0,G_neg_I_flat,G_I_flat,lam_res,O,wlist=None,block_mathod='consecutive',
                          tol = tol, tol_norm=tol_norm,max_iter = max_iter, varing_rho=varing_rho)
        if (Bt==0).all():
            print('The penalty parameter is too large, the debiasing procedure is equivalent to the OLS regression.')
        Z_I = G_I_flat - np.dot(G_neg_I_flat, Bt)
        P_proj = Z_I @ np.linalg.inv(Z_I.T @ Z_I) @ Z_I.T
        effect_mat = P_proj @ G_I_flat

        if np.linalg.matrix_rank(effect_mat) == effect_mat.shape[1]:
            midterm = P_proj @ (Y - np.dot(G_neg_I_flat, X_neg_I_flat))
            effect_mat_inv = scipy.linalg.pinv(effect_mat)
            X_hat_debias = effect_mat_inv @ midterm
            # import statsmodels.api as sm
            # OLSmodel = sm.OLS(Y, G_I_flat)
            # OLSresult = OLSmodel.fit()
            # print(OLSresult.tvalues)
            debias_var = effect_mat_inv @ P_proj @ effect_mat_inv.T
            debias_var_inv_square = np.diag(debias_var)
            debias_var_inv_square = np.diag(1/np.sqrt(debias_var_inv_square))
            X_hat_debias_rotate = debias_var_inv_square @ X_hat_debias
        else:
            print('The Q_A*G_A is not full rank, the debiasing is not performed.')
            X_I = Xt_tilde[I_hat, :, :]
            X_I_flat = np.reshape(X_I, (I_hat.shape[0]*O, T))
            debias_flag = False
            X_hat_debias = X_I_flat.copy()
            X_hat_debias_rotate = X_I_flat.copy()
            effect_mat = 0
    else:
        X_I = Xt_tilde[I_hat, :, :]
        X_I_flat = np.reshape(X_I, (I_hat.shape[0]*O, T))
        X_hat_debias = X_I_flat.copy()
        X_hat_debias_rotate = X_I_flat.copy()
        effect_mat = 0

    return X_hat_debias, X_hat_debias_rotate, debias_flag, effect_mat

def gl_ADMM_dual(X0, G, Y, lam, O, wlist='auto', block_mathod='consecutive', 
                 tol=1e-8, tol_norm=1e-4, max_iter=1000, varing_rho=True):
    # Initialize
    N = G.shape[0]
    S = int(G.shape[1]/O)
    T = Y.shape[1]

    G_tensor = _Gmatrix_to_tensor(G, O, block_mathod=block_mathod)
    X0_tensor = _Xmatrix_to_tensor(X0, O, block_mathod=block_mathod)
    if wlist is None:
        I_O = np.eye(O)
        sigma_G_sqrt = np.repeat(I_O[:, :, np.newaxis], S, axis=2)
        sigma_G_sqrt_inv = sigma_G_sqrt
    elif wlist == 'auto':
        sigma_G_sqrt, sigma_G_sqrt_inv = _Gtensor_covariance(G_tensor)
    else:
        sigma_G_sqrt = wlist[0]
        sigma_G_sqrt_inv = wlist[1]

    G_tilde = np.einsum('ikl, kjl->ijl',G_tensor, sigma_G_sqrt_inv)
    G_tilde = np.reshape(G_tilde, (N, S*O),order='F')

    GGT = np.dot(G_tilde, G_tilde.T)
    # stepsize0 = math.sqrt(N*S*O)/np.linalg.norm(GGT, 2)
    # stepsize0 = 1/np.linalg.norm(GGT, 2)
    # stepsize0 = math.sqrt(N*S*O)/np.linalg.norm(GGT)
    # stepsize0 = math.sqrt(N)/np.linalg.norm(GGT)
    stepsize0 = np.array(1/S/O)
    max_stepsize_tol = stepsize0.copy()*math.sqrt(N*S*O)
    min_stepsize_tol = stepsize0.copy()/math.sqrt(N*S*O)
    stepsize = stepsize0.copy()
    coreinv = np.linalg.inv(np.eye(N)+stepsize*GGT)
    coreinvG = np.dot(coreinv, G_tilde)
    coreinvY = np.dot(coreinv, Y)
    fastcoeff = 1  #(1+math.sqrt(5))/2

    Xt = np.einsum('ijl, ljk->lik', sigma_G_sqrt, X0_tensor)
    Xt = np.reshape(Xt, (S*O, T))

    #Xt = scipy.sparse.csr_matrix(Xt)
    #Xt_all_zero = np.zeros((S*O, T))
    #Xt_all_zero = scipy.sparse.csr_matrix((S*O, T))
    utnorm = np.linalg.norm(Xt, axis=1)
    utmask = utnorm > 0
    Xt = Xt[utmask, :]

    Zt = Y - G_tilde[:,utmask] @ Xt#scipy.sparse.csr_matrix.dot(G_tilde, Xt)
    #Xt = Xt.toarray()

    Ut = np.dot(G_tilde.T, Zt)
    loss_p = math.inf
    loss_d = math.inf
    tol_p = 0
    tol_d = 0

    sqrtSOT = np.sqrt(S*O*T)
    sqrtNT = np.sqrt(N*T)
    tolsqrtSOT = tol*sqrtSOT
    tolsqrtNT = tol*sqrtNT

    iter_idx = 0
    print('iter_idx = ', iter_idx, 'loss_p = ', loss_p, 'loss_d = ',loss_d, 'lam = ', lam, 'stepsize = ', stepsize)

    while iter_idx < max_iter and (loss_p>tol_p or loss_d>tol_d):
        # Update Z
        mXtpUt = stepsize*Ut
        mXtpUt[utmask, :] = mXtpUt[utmask, :] - Xt
        Zt = coreinvY+np.dot(coreinvG,mXtpUt)
        #Zt = coreinvY+np.dot(coreinvG,(-Xt+stepsize*Ut))
        GTZ = np.dot(G_tilde.T, Zt)
        # Update U
        Utpre = Ut.copy()
        Ut = GTZ.copy()
        Ut[utmask, :] = GTZ[utmask, :] + Xt/stepsize
        if O == 1:
            utnorm = np.linalg.norm(Ut, axis=1)
        else:
            Utreshape = Ut.reshape((S,O,T))
            utnorm = np.linalg.norm(Utreshape, axis=(1,2))
            utnorm = np.repeat(utnorm, O)
        utmasknew = utnorm > lam
        Ut[utmasknew, :] = lam*Ut[utmasknew, :]/utnorm[utmasknew][:, None]
        # Update X
        Xtt = fastcoeff*stepsize*(GTZ[utmasknew, :]-Ut[utmasknew, :])
        utmask_and_utmasknew = np.logical_and(utmask, utmasknew)
        Xtt[utmask_and_utmasknew[utmasknew], :] = Xt[utmask_and_utmasknew[utmask], :] + Xtt[utmask_and_utmasknew[utmasknew], :]
        Xt = Xtt.copy()
        utmask = utmasknew.copy()
        #set_row_csr(Xtt, utmask, Xt[utmask, :] + fastcoeff*stepsize*(GTZ[utmask, :]-Ut[utmask, :]))
        #Xtt[utmask, :] = Xt[utmask, :] + fastcoeff*stepsize*(GTZ[utmask, :]-Ut[utmask, :])
        #Xt[~utmask, :] = 0

        GX = G_tilde[:,utmask] @ Xt#scipy.sparse.csr_matrix.dot(G_tilde, Xtt)
        #Xt = Xtt.toarray()

        loss_p = np.linalg.norm(GTZ-Ut)
        loss_d = np.linalg.norm(stepsize*G_tilde@(Ut-Utpre))

        tol_p = tolsqrtSOT + tol_norm*max(np.linalg.norm(GTZ), np.linalg.norm(Ut))
        tol_d = tolsqrtNT + tol_norm*np.linalg.norm(GX)

        if varing_rho and iter_idx < max_iter/10:
            if loss_d/sqrtNT > 10*loss_p/sqrtSOT:
                stepsize = max(stepsize/2,min_stepsize_tol)
                coreinv = np.linalg.inv(np.eye(N)+stepsize*GGT)
                coreinvG = np.dot(coreinv, G_tilde)
                coreinvY = np.dot(coreinv, Y)
            elif loss_p/sqrtSOT > 10*loss_d/sqrtNT:
                stepsize = min(stepsize*2,max_stepsize_tol)
                coreinv = np.linalg.inv(np.eye(N)+stepsize*GGT)
                coreinvG = np.dot(coreinv, G_tilde)
                coreinvY = np.dot(coreinv, Y)

        iter_idx = iter_idx + 1
        if iter_idx % 200 == 0:
            print('iter_idx = ', iter_idx, 'loss_p = ', loss_p, 'loss_d = ',loss_d, 'lam = ', lam, 'stepsize = ', stepsize)
    
    Xtt = np.zeros((S*O, T))
    Xtt[utmask, :] = Xt
    Xt = Xtt
    Xt_tensor = _Xmatrix_to_tensor(Xt, O, block_mathod=block_mathod)
    Xout = np.einsum('ijl, ljk->lik', sigma_G_sqrt_inv, Xt_tensor)

    if block_mathod == 'jump':
        Xout = np.reshape(Xout, (S*O,T), order='F')
    elif block_mathod == 'consecutive':
        Xout = np.reshape(Xout, (S*O,T))

    if iter_idx == max_iter:
        print('Warning: the ADMM algorithm does not converge to the desired tolerance.')
    return Xout

if __name__ == '__main__':
    N = 256 # number of sensors
    S = 1000 # number of sources
    O = 3 # number of orientations
    T = 100 # number of time points
    k = 3 # number of non-zero sources
    lam = 75 #0.3 * math.sqrt(3*100*256) # regularization parameter
    #O_org = 'jump'
    O_org = 'consecutive'

    # Generate random data
    G = np.random.randn(N, S*O)
    G_column_norm = np.random.uniform(0.1,10,(S))
    X = np.zeros((S*O, T))
    selected_locs = random.sample(range(S), k)
    if O_org == 'jump':
        for start in selected_locs:
            amp = np.random.randn(T)
            ori = np.random.randn(O)
            ori = ori/np.linalg.norm(ori)
            X[np.arange(start, S*O, S, dtype = 'int'), :] = np.outer(ori, amp)
        selected_locs_full = np.concatenate([np.arange(start, S*O, S, dtype = 'int') for start in selected_locs])
        G_column_norm = np.concatenate([G_column_norm for start in O])
    elif O_org == 'consecutive':
        for start in selected_locs:
            amp = np.random.randn(T)
            ori = np.random.randn(O)
            ori = ori/np.linalg.norm(ori)
            X[(start*O):(start*O+O), :] = np.outer(ori, amp)
        selected_locs_full = np.concatenate([np.arange(start*O, start*O+O, dtype = 'int') for start in selected_locs])
        G_column_norm = np.concatenate([np.array([start]*O) for start in G_column_norm])

    G = G * G_column_norm[None, :]

    E = np.random.randn(N, T) * 0.3
    Y = np.dot(G, X) + E  # Y = G@X + E

    X0 = np.zeros((S*O, T))
    G_tensor = _Gmatrix_to_tensor(G, O, block_mathod=O_org)

    sigma_G_sqrt, sigma_G_sqrt_inv = _Gtensor_covariance(G_tensor)
    # weighting matrix W_s = sigma_G_sqrt_inv
    wlist = [sigma_G_sqrt, sigma_G_sqrt_inv]

    # raw estimate without bias correction
    Xout, lamt, sigma_list = \
        gl_ADMM_dual_joint(50,X0,G,Y,lam,O,wlist,block_mathod=O_org,
                           tol = 1e-4, max_iter = 2000, varing_rho=True)
    
    I_hat, G_tensor, Xt_tensor = _cal_I_hat(Xout, G, O, block_mathod=O_org)
    estnum = I_hat.shape[0]
    print(estnum)

    # debiased estimate
    Xout_debias, Xout_debias_rotate, significance_list = \
        gl_ADMM_dual_bias_correction(I_hat, Xt_tensor, G_tensor, Y, lam, O, bias_correction_method = 'joint',
                                     clear_not_select=True, block_mathod=O_org, tol = 1e-4,max_iter = 4000,
                                     varing_rho=True)

    # check the ground truth and the estimated source activities
    print(X[selected_locs_full, 0])
    print(Xout[selected_locs_full, 0])
    print(Xout_debias[selected_locs_full, 0])
    # check the MSE
    print('group Lasso full loss', np.linalg.norm(X-Xout))
    print('debiased group Lasso full loss', np.linalg.norm(X-Xout_debias))
    print('group Lasso selected loss', np.linalg.norm(X[selected_locs_full, :]-Xout[selected_locs_full, :]))
    print('debiased group Lasso selected loss', np.linalg.norm(X[selected_locs_full, :]-Xout_debias[selected_locs_full, :]))
    # check the estimated sigma
    print(np.linalg.norm(Y - np.dot(G, X)) / math.sqrt(N*T))
    print(np.linalg.norm(Y - np.dot(G, Xout)) / math.sqrt(N*T))
    print(np.linalg.norm(Y - np.dot(G, Xout_debias)) / math.sqrt(N*T))
