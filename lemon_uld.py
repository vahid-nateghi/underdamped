import numpy as np
import time
from pathlib import Path
import h5py
import pickle
from sklearn.model_selection import train_test_split
import torch
 
from util.pcca import _pcca_connected_isa
from Lemon_Slice import LemonSlice
import RFF_uld as rf

from eff_dyn_learning import effective_diffusion, effective_drift
from util.scriptutil import load_config
from data_preparation import data_loader


def old_analysis(cg_data_full, friction, test_ratio, input):
    from RFF_old import M_RFF_Generator, eigen_decomposition

    C1_simp = 0.5952
    C2_simp = 0.5605
    ratio = C1_simp/C2_simp
    its = np.zeros((5,3))
    for i in range(5):
        Omega = rf.sample_rff_gauss_periodic(1, 100, np.pi, 1, 30)
        cg_data, _ = train_test_split(cg_data_full.T, train_size=test_ratio)

        M0, Ml = M_RFF_Generator(cg_data.T[0][None,:], Omega, a=ratio*2/friction/input.beta)
        M = M0.T @ M0
        eig_vals, _, _, _ = eigen_decomposition(M,Ml,tol=1e-8, r_max=Ml.shape[-1]) 
        its[i] = 1/eig_vals[1:4]

    np.save(input.data_path+f'/lemon_old_gedmd_beta_{input.beta:.2f}.npy', its)


def vamp_score_analysis(X_full, cg_data_full, test_lemonslice, test_ratio, input):
    mode_list = ['feat_num','gamma'] 
    for mode in mode_list:
        if mode == 'feat_num':            
            p_list = [50,100,200,500]
            sigma = 1
        elif mode == 'gamma':
            p_list = [0.1, 0.3, 0.7, 1, 1.3, 5]
            p = 300
        ntest = 5
        nev = 4
        d_mat = np.zeros((nev, len(p_list), ntest)) 
        dtest_mat = np.zeros((nev, len(p_list), ntest))

        X,_, cg_data,_= train_test_split(X_full.T, cg_data_full.T, train_size=test_ratio)
        X = X.T
        cg_data = cg_data.T 
        
        grad_V = test_lemonslice.gradient(X)
        cg_map_grad = test_lemonslice.cg_map_grad(X) 

        for i in range(len(p_list)):
            if mode == 'feat_num':
                Omega1 = rf.sample_rff_gauss_periodic(1, p_list[i], np.pi, sigma, 30)
                Omega2 = rf.sample_rff_gaussian(1, p_list[i], sigma)
            elif mode == 'gamma':
                Omega1 = rf.sample_rff_gauss_periodic(1, p, np.pi, p_list[i], 30)
                Omega2 = rf.sample_rff_gaussian(1, p, p_list[i])

            Omega = np.concatenate((Omega1, Omega2), axis=0)

            d, dtest = rf.cv_generator_rff(X[2:], Omega, input.beta, test_lemonslice.gamma, grad_V, 
                                            rtrain=0.5, ntest=ntest, nev=nev, tol=1e-4, batch_size=2000,
                                            cg_data=cg_data, cg_map_grad=cg_map_grad)
            d_mat[:,i] = np.real(d.T[::-1])
            dtest_mat[:,i] = np.real(dtest.T[::-1])

        np.savez(input.data_path+f'/lemon_vamp_{mode}_beta_{input.beta:.2f}.npz',
                                     d_mat=d_mat, dtest_mat=dtest_mat)

def gedmd_analysis(X_full, cg_data_full, test_lemonslice, test_ratio, input):
    nev = 4
    p = 400
    ntests = 5
    sigma = 0.8

    d_zeta = int(cg_data_full.shape[0]/2)
    eigs = np.zeros((ntests,3))
    Wlist = []
    chi_list = []
    Xlist = []
    cg_datalist = []
    Omega_list = []
    transformation_mat_list = []
    for i in range(ntests):
        Omega1 = rf.sample_rff_gauss_periodic(d_zeta, p, np.pi, sigma, 30)
        Omega2 = rf.sample_rff_gaussian(d_zeta, p, sigma)
        Omega = np.concatenate((Omega1, Omega2), axis=0)
        Omega_list.append(Omega)
        if input.beta >= 1.5:
            weights_full = np.load(input.data_path+f"/weights_epoch75_beta_{input.beta:.2f}.npy")[:,None]  
        else:
            weights_full = np.ones((X_full.shape[1],1))

        X,_, cg_data,_, weights,_= train_test_split(X_full.T, cg_data_full.T, weights_full,
                                                     train_size=test_ratio)
        X = X.T
        cg_data = cg_data.T 
        weights = weights.T

        cg_map_grad = test_lemonslice.cg_map_grad(X) 
        grad_V = test_lemonslice.gradient(X)

        di, W, U, M0 = rf.spectral_analysis_rff_generator_coarse_grained(cg_data, grad_V, cg_map_grad, X[2:], 
                                    Omega, nev, input.beta, test_lemonslice.gamma, tol=1e-4, 
                                    batch_size=4000, weights=weights)
        transformation_mat_list.append(U)
        eigs[i] = -np.real(di[-4:-1][::-1])

        W = M0 @ W

        for ii in range(4):
            theta_opt = 0.0
            vi = W[:, ii]
            vmax = np.max(np.abs(np.imag(vi)))
            for theta in np.arange(0, 2*np.pi, 0.05):
                if np.max(np.abs(np.imag(np.exp(1j * theta)*vi))) < vmax:
                    theta_opt = theta
                    vmax = np.max(np.abs(np.imag(np.exp(1j * theta)*vi)))
            W[:, ii] *= np.exp(1j * theta_opt)

        W = np.real(W[:, ::-1])
        W[:, 0] = np.mean(W[:, 0])

        chi, _ = _pcca_connected_isa(W[:, :4], n_clusters=4)
       
        Wlist.append(W)
        chi_list.append(chi)
        Xlist.append(X)
        cg_datalist.append(cg_data)

    with h5py.File(input.data_path+f'/lemon_{input.data_mode}_gedmd_beta_{input.beta:.2f}.h5', 'w') as f:
        f.create_dataset('Wlist', data=Wlist)
        f.create_dataset('Xlist', data=Xlist)
        f.create_dataset('cg_datalist', data=cg_datalist)
        f.create_dataset('eigs', data=eigs)
        f.create_dataset('Omega_list', data=Omega_list)
        mylist = f.create_group('transformation_mat_list')
        for i, arr in enumerate(transformation_mat_list):
            mylist.create_dataset(f'transformation_mat_{i}', data=arr)

def eff_dyn_learning(test_lemonslice, input):
 
    with h5py.File(input.data_path+f'/lemon_{input.data_mode}_gedmd_beta_{input.beta:.2f}.h5', 'r') as f:
        Xlist = f['Xlist'][()]
        cg_datalist = f['cg_datalist'][()]
        Omega_list = f['Omega_list'][()]
        mylist = f['transformation_mat_list']
        transformation_mat_list = [mylist[name][...] for name in mylist]

    X = Xlist[0]
    cg_data = cg_datalist[0]
    if input.beta == 2:
        Omega = Omega_list[3]
        U =transformation_mat_list[3] 
    elif input.beta == 1:
        Omega = Omega_list[0] 
        U = transformation_mat_list[0] 

    arg = np.dot(cg_data.T, Omega)
    M0 = np.hstack((np.cos(arg),np.sin(arg))) 
    M0_red = M0 @ np.real(U)
    cg_map_grad = test_lemonslice.cg_map_grad(X) 

    a_loc, alpha_sigma, a_loc_nn, a_loc_x2_scaler, a_loc_y_scaler  = \
        effective_diffusion(cg_data, test_lemonslice, cg_map_grad, M0_red)

    b_loc, alpha_b, b_loc_nn, b_loc_x2_scaler, b_loc_y_scaler = \
        effective_drift(cg_data, X, test_lemonslice, cg_map_grad, M0_red)

    lemon_eff_dyn = {
        'a_loc_ref': a_loc,
        'b_loc_ref': b_loc,
        'alpha_a': alpha_sigma,
        'a_loc_nn': a_loc_nn,
        'a_loc_y_scaler': a_loc_y_scaler,
        'a_loc_x2_scaler': a_loc_x2_scaler,
        'alpha_b':alpha_b,
        'b_loc_nn': b_loc_nn,
        'b_loc_y_scaler': b_loc_y_scaler,
        'b_loc_x2_scaler': b_loc_x2_scaler
    }

    with open(input.data_path+f'/lemon_{input.data_mode}_eff_dyn_beta_{input.beta:.2f}.pkl', 'wb') as f:
        pickle.dump(lemon_eff_dyn, f)

def gedmd_analysis_learned(test_lemonslice, input):

    with h5py.File(input.data_path+f'/lemon_{input.data_mode}_gedmd_beta_{input.beta:.2f}.h5', 'r') as f:
        Xlist = f['Xlist'][()]
        cg_datalist = f['cg_datalist'][()]
        Omega_list = f['Omega_list'][()]

    ntests = 5
    its_learned = np.zeros((3,ntests))

    with open(input.data_path+f'/lemon_{input.data_mode}_eff_dyn_beta_{input.beta:.2f}.pkl', 'rb') as f:
        lemon_eff_dyn = pickle.load(f)
    b_loc_nn = lemon_eff_dyn['b_loc_nn']
    b_loc_y_scaler = lemon_eff_dyn['b_loc_y_scaler']
    b_loc_x2_scaler = lemon_eff_dyn['b_loc_x2_scaler']
    a_loc_nn = lemon_eff_dyn['a_loc_nn']
    a_loc_y_scaler = lemon_eff_dyn['a_loc_y_scaler']
    a_loc_x2_scaler = lemon_eff_dyn['a_loc_x2_scaler']

    if input.beta == 2:
        Omega = Omega_list[3]
    elif input.beta == 1:
        Omega = Omega_list[0] 

    for i in range(ntests):

        X = Xlist[i]
        cg_data = cg_datalist[i]
        
        ## Checking the timescales of the learned system
        grad_V = test_lemonslice.gradient(X)

        x1_sin = np.sin(cg_data[0])
        x1_cos = np.cos(cg_data[0])

        x2_scaled = a_loc_x2_scaler.transform(cg_data[1].reshape(-1,1))
        X_processed = np.concatenate((x1_sin[:,None], x1_cos[:,None], x2_scaled),axis=1)

        cg_data_tor = torch.tensor(X_processed, dtype = torch.float32)
        a_ed = a_loc_y_scaler.inverse_transform(a_loc_nn(cg_data_tor)[:,0].detach()[:,None])[:,0]
        
        sigma_ed = np.zeros((2,2,cg_data.shape[1]))
        sigma_ed[1,1] = np.sqrt(a_ed)

        x2_scaled = b_loc_x2_scaler.transform(cg_data[1].reshape(-1,1))
        x_processed = np.concatenate((x1_sin[:,None], x1_cos[:,None], x2_scaled),axis=1)
        grad_V = np.squeeze(b_loc_y_scaler.inverse_transform(\
            b_loc_nn(torch.tensor(x_processed, dtype = torch.float32)).detach().numpy()))

        grad_V = grad_V+test_lemonslice.gamma*cg_data[1]

        di, _, _, _ = rf.spectral_analysis_rff_generator(cg_data, Omega, 4, grad_V[None,:], 
                            input.beta, sigma_ed, tol=1e-4, batch_size=4000) 
        its_learned[:,i] = -1/np.real(di[-4:-1][::-1])

    np.save(input.data_path+f'/lemon_{input.data_mode}_learned_its_beta_{input.beta:.2f}.npy', 
            its_learned)

def integration(test_lemonslice, input):
    with h5py.File(input.data_path+f'/lemon_{input.data_mode}_gedmd_beta_{input.beta:.2f}.h5', 'r') as f:
        Omega_list = f['Omega_list'][()]
        mylist = f['transformation_mat_list']
        transformation_mat_list = [mylist[name][...] for name in mylist]

    with open(input.data_path+f'/lemon_{input.data_mode}_eff_dyn_beta_{input.beta:.2f}.pkl', 'rb') as f:
        lemon_eff_dyn = pickle.load(f)
        
    alpha_b = lemon_eff_dyn['alpha_b']
    b_loc_nn = lemon_eff_dyn['b_loc_nn']
    b_loc_y_scaler = lemon_eff_dyn['b_loc_y_scaler']
    b_loc_x2_scaler = lemon_eff_dyn['b_loc_x2_scaler']
    alpha_sigma = lemon_eff_dyn['alpha_a']
    a_loc_nn = lemon_eff_dyn['a_loc_nn']
    a_loc_y_scaler = lemon_eff_dyn['a_loc_y_scaler']
    a_loc_x2_scaler = lemon_eff_dyn['a_loc_x2_scaler']

    nn_data = {
        'a_loc_nn': a_loc_nn,
        'a_loc_y_scaler': a_loc_y_scaler,
        'a_loc_x2_scaler': a_loc_x2_scaler,
        'b_loc_nn': b_loc_nn,
        'b_loc_y_scaler': b_loc_y_scaler,
        'b_loc_x2_scaler': b_loc_x2_scaler
    }

    x_proplist = []
    b_proplist = []
    a_proplist = []

    if input.beta == 1:
        Omega = Omega_list[0]
        U = np.real(transformation_mat_list[0])
        ntests = 2
    elif input.beta == 2:
        Omega = Omega_list[3] 
        U = np.real(transformation_mat_list[3])
        ntests = 5

    for i in range(ntests):

        x_prop, b_prop, a_prop = test_lemonslice.Simulate_CG(alpha_b, nn_data, alpha_sigma, Omega,
                                                U, [1,0], input.integration_length, input.dt) 
        x_proplist.append(x_prop)
        b_proplist.append(b_prop)
        a_proplist.append(a_prop)

    with h5py.File(input.data_path+f'/lemon_prop_beta_{input.beta:.2f}.h5', 'w') as f:
        f.create_dataset('x_prop', data=x_proplist)
        f.create_dataset('b_prop', data=b_proplist)
        f.create_dataset('a_prop', data=a_proplist)
        f.attrs['dt'] = input.dt

def integ_analysis(test_lemonslice, input):

    with h5py.File(input.data_path+f'/lemon_{input.data_mode}_gedmd_beta_{input.beta:.2f}.h5', 'r') as f:
        Omegalist = f['Omega_list'][()]

    with h5py.File(input.data_path+f'/lemon_prop_beta_{input.beta:.2f}.h5', 'r') as f:
        x_prop_list = f['x_prop'][()]
        a_prop_list = f['a_prop'][()]
        b_prop_list = f['b_prop'][()]

    ### gEDMD
    if input.beta == 1:
        Omega = Omegalist[0]
        delta = 10 if input.integration_length >= 1e+6 else 1
        data_size = min(40000, input.integration_length-1)  
    elif input.beta == 2:
        Omega = Omegalist[3] 
        delta = 10 if input.integration_length >= 1e+6 else 1
        data_size = min(15000, input.integration_length-1)

    ntests = 5
    its_list = np.zeros((ntests,3))
    x_proplist = []
    sigma_edlist = []
    grad_Vlist = []
    Wlist = []
    chilist = []

    for ii in range(ntests):
        x_prop_concat = []
        sigma_ed_concat = []
        grad_V = []

        for i in range(len(x_prop_list)):
            x_prop_samp, _,a_prop_samp,_,b_prop_samp,_ = \
                train_test_split(x_prop_list[i][:,:-1][:,::delta].T,
                                a_prop_list[i][0,1:][::delta],b_prop_list[i][0,1:][::delta],
                                train_size=data_size/x_prop_list[i].shape[1]*delta)
            x_prop_samp = x_prop_samp.T
            x_prop_concat.append(x_prop_samp) 
            sigma_ed_concat.append(a_prop_samp)
            grad_V.append(b_prop_samp+test_lemonslice.gamma*x_prop_samp[1])

        x_prop_samp = np.concatenate(x_prop_concat,axis=1)
        x_proplist.append(x_prop_samp)
        sigma_ed = np.zeros((2,2,x_prop_samp.shape[1]))
        sigma_ed[1,1] = np.concatenate(sigma_ed_concat)
        sigma_edlist.append(sigma_ed)
        grad_V = -np.squeeze(np.concatenate(grad_V)) 
        grad_Vlist.append(grad_V)

        di, W, _, M0 = rf.spectral_analysis_rff_generator(x_prop_samp, Omega, 4, grad_V[None,:], 
                                    input.beta, np.sqrt(sigma_ed), tol=1e-4, batch_size=4000) 
        W = M0 @ W

        its_prop = -1/np.real(di[-4:-1][::-1])

        its_list[ii] = its_prop
        for j in range(4):
            theta_opt = 0.0
            vi = W[:, j]
            vmax = np.max(np.abs(np.imag(vi)))
            for theta in np.arange(0, 2*np.pi, 0.05):
                if np.max(np.abs(np.imag(np.exp(1j * theta)*vi))) < vmax:
                    theta_opt = theta
                    vmax = np.max(np.abs(np.imag(np.exp(1j * theta)*vi)))
            W[:, j] *= np.exp(1j * theta_opt)

        W = np.real(W[:, ::-1])
        W[:, 0] = np.mean(W[:, 0])
        Wlist.append(W)
        chi, _ = _pcca_connected_isa(W, n_clusters=4)
        chilist.append(chi)

    with h5py.File(input.data_path+f'/lemon_prop_its_beta_{input.beta:.2f}.h5', 'w') as f:
        f.create_dataset('its', data=its_list)
        f.create_dataset('x_proplist', data=x_proplist)
        f.create_dataset('sigma_edlist', data=sigma_edlist)
        f.create_dataset('grad_Vlist', data=grad_Vlist)
        f.create_dataset('Wlist', data=Wlist)
        f.create_dataset('chilist', data=chilist)

def EDMD_analysis(data, delta, lag, nev, input, analysis):
    assert analysis in ('integ', 'uld'), "analysis must be 'integ' or 'uld'"

    if input.beta >= 1:
        train_ratio = 0.4
        delta_ = 10
    else:
        train_ratio = 0.2
        delta_ = 1

    its_edmd = np.zeros((3,5))

    data_sliced = [x[:,::delta_] for x in data] 

    for i in range(its_edmd.shape[1]): 
        Omega1 = rf.sample_rff_gauss_periodic(1, 300, np.pi, 1, 30)
        Omega2 = rf.sample_rff_gaussian(1, 300, 1)
        Omega = np.concatenate((Omega1, Omega2), axis=0)

        d, _, _ = rf.spectral_analysis_edmd(data_sliced, Omega, lag, nev, train_ratio, tol=1e-6)
        t_i = -delta * delta_ * input.dt * lag / np.log(np.real(d[:-1]))

        its_edmd[:,i] = t_i[::-1]

    np.save(input.data_path+f'/lemon_edmd_{analysis}_beta_{input.beta:.2f}.npy', its_edmd)


if __name__ == '__main__':


    cwd = Path.cwd()
    Path(cwd/".figures/").mkdir(parents=True, exist_ok=True)
    Path(cwd/".results/").mkdir(parents=True, exist_ok=True)

    input = load_config("", "setting_gedmd.json")

    if input.mass != "equalmass":
        raise ValueError(f"This script only handles 'equalmass' case, for which the 'mass' \
                         in the config file 'setting_gedmd.json' should be 'equalmass'. In order to \
                         analyze the 'diffmass' case, run the script called 'lemon_uld_diffmass.py'.")

    gamma = 5

    test_lemonslice = LemonSlice(k=4, beta=input.beta, gamma=gamma, multiplier=1.2)
    
    # Generating or loading data 
    X_full, cg_data_full, test_ratio = data_loader(test_lemonslice, input)

    # EDMD in full space
    if input.EDMD_fullspace: 
        print('EDMD in full-space dataset is started...')
        assert input.data_mode in ('uld'), "EDMD works only if data_mode in the config file is 'uld'"
        EDMD_analysis([cg_data_full], delta=5, lag=10, nev=4, input=input, analysis='uld')
    # CG generator approximation induced by full-space data
    if input.gedmd_analysis:
        print('gEDMD analysis is started...')
        gedmd_analysis(X_full, cg_data_full, test_lemonslice, test_ratio, input)

    # CG dynamics learning
    if input.eff_dyn_learning:
        print('Effective dynamics learning is started...')
        eff_dyn_learning(test_lemonslice, input)

    # CG generator approximation induced by full-space data according to the CG dynamics
    if input.gedmd_analysis_learned:
        print('gEDMD based on learned drift and diffusion is started...')
        gedmd_analysis_learned(test_lemonslice, input)

    # CG dynamics integration
    if input.integration:
        print('Integration of the learned SDE is started...')
        integration(test_lemonslice, input)

    # EDMD in CG space
    if input.EDMD_CGspace: 
        print('EDMD in integrated dataset is started...')
        with h5py.File(input.data_path+f'/lemon_prop_beta_{input.beta:.2f}.h5', 'r') as f:
            data = f['x_prop'][()]
        EDMD_analysis(data, delta=1, lag=10, nev=4, input=input, analysis='integ')
    # CG generator approximation induced by CG data
    if input.integ_analysis:
        print('gEDMD in integrated dataset is started...')
        integ_analysis(test_lemonslice, input)

    # CG generator approximation according to the overdamped Langevin assumption
    if input.old_analysis:
        print('Overdamped gEDMD in full-space dataset is started...')
        old_analysis(cg_data_full, gamma, test_ratio, input)

    # RFF Parameters' optimization
    if input.vamp_score_analysis:
        print('VAMP-score analysis is started...')
        vamp_score_analysis(X_full, cg_data_full, test_lemonslice, test_ratio, input)
