import numpy as np
from pathlib import Path
import h5py
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl

from Lemon_Slice import LemonSlice
import RFF_uld as rf
from util.scriptutil import load_config

def gedmd_analysis(X_full, cg_data_full, test_lemonslice, test_ratio, input, mass, mode, tol, sigma):
    nev = 4
    p = 400
    ntests = 5

    d_zeta = int(cg_data_full.shape[0]/2)

    its = np.zeros((nev-1,ntests))

    for i in range(ntests):
        Omega1 = rf.sample_rff_gauss_periodic(d_zeta, p, np.pi, 1, 30)
        Omega2 = rf.sample_rff_gaussian(d_zeta, p, sigma)
        Omega = np.concatenate((Omega1, Omega2), axis=0)

        X,_, cg_data,_= train_test_split(X_full.T, cg_data_full.T, train_size=test_ratio)
        X = X.T
        cg_data = cg_data.T 

        cg_map_grad = test_lemonslice.cg_map_grad(X) 
        grad_V = test_lemonslice.gradient(X)
        print(f"CG data shapes: X={cg_data.shape}") if i==0 else None

        di, _, _,_ = rf.spectral_analysis_rff_generator_coarse_grained(cg_data, grad_V, cg_map_grad, X[2:], 
                                    Omega, nev, input.beta, test_lemonslice.gamma, tol=tol,  
                                    batch_size=2000, mass=mass)
        its[:,i] = -1/np.real(di[-4:-1][::-1])
        
    np.save(input.data_path+f'/lemon_gedmd_diff_mass_analysis_{mode}.npy', its)


def EDMD_analysis(data, delta, lag, nev, dt, train_ratio):
    ntests = 5
    its_edmd = np.zeros((nev-1,ntests))

    for i in range(ntests): 

        Omega1 = rf.sample_rff_gauss_periodic(1, 300, np.pi, 1, 30)
        Omega2 = rf.sample_rff_gaussian(1, 300, 1)
        Omega = np.concatenate((Omega1, Omega2), axis=0)
        d, _, _ = rf.spectral_analysis_edmd(data, Omega, lag, nev, train_ratio, tol=1e-4)
        t_i = -delta * dt * lag / np.log(np.real(d[:-1]))

        its_edmd[:,i] = t_i[::-1]

    np.save(input.data_path+f'/lemon_edmd_diff_mass_analysis_{mode}.npy', its_edmd)

if __name__ == '__main__':

    cwd = Path.cwd()
    Path(cwd/".figures/").mkdir(parents=True, exist_ok=True)
    Path(cwd/".results/").mkdir(parents=True, exist_ok=True)

    input = load_config("", "setting_gedmd.json")

    friction = 5
    delta = 50
    mode = input.mass # diffmass  equalmass
    if mode == 'equalmass':
        mass = np.array([1,1])
        test_lemonslice = LemonSlice(k=4, beta=input.beta, gamma=friction, multiplier=1.2, mass=mass)
        with h5py.File(input.data_path+f'/lemon_data_uld_beta_{input.beta:.2f}.h5', 'r') as f:
            X_full = f['x_fs'][()][:,::delta] 
            cg_data_full = f['x_cg'][()][:,::delta] 

    else:
        mass = np.array([1.,0.25])
        test_lemonslice = LemonSlice(k=4, beta=input.beta, gamma=friction, multiplier=1.2, mass=mass)

        if input.new_data:

            x0 = np.array([1,1,0,0])
            X = test_lemonslice.Simulate(x0, 10000000, input.dt)
            cg_data = np.array(test_lemonslice.coarse_graining(X))

            with h5py.File(input.data_path+f'/lemon_data_uld_test_diffmass.h5', 'w') as f:
                f.create_dataset('x_fs', data=X)
                f.create_dataset('x_cg', data=cg_data)
            X_full = X[:,::delta] 
            cg_data_full = cg_data[:,::delta] 
        else:
            with h5py.File(input.data_path+f'/lemon_data_uld_test_diffmass.h5', 'r') as f:
                X_full = f['x_fs'][()][:,::delta] 
                cg_data_full = f['x_cg'][()][:,::delta] 

    # EDMD 
    if input.EDMD_CGspace:
        print('EDMD analysis is started...')
        assert mode in ('equalmass', 'diffmass'), "analysis must be 'equalmass' or 'diffmass'"

        data = [cg_data_full] 
        EDMD_analysis(data, delta=delta, lag=10, nev=4, dt=input.dt, train_ratio=0.4)

    # CG generator approximation induced by full-space data
    if input.gedmd_analysis:
        print('gEDMD analysis is started...')
        if mode == 'equalmass':
            tol = 1e-4
            sigma = 1
            test_ratio = 0.2
        else:
            tol = 1e-4
            sigma = 2
            test_ratio = 0.4
        gedmd_analysis(X_full, cg_data_full, test_lemonslice, test_ratio, input, mass, mode, tol=tol, sigma=sigma) 
    

