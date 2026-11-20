import numpy as np
import h5py
import json

from rejection_sampling import rejection_sampling

def data_loader(lemonslice, config):
    if config.data_mode == 'uld':
        if config.beta >= 1:
            delta = 5 
        else:
            delta = 5
        if config.new_data:
            m = 20_000_000
            x0 = np.array([1,1,0,0])
            X = lemonslice.Simulate(x0, m, config.dt)
            V = lemonslice.potential(X)
            V -= V.min()
            cg_data = np.array(lemonslice.coarse_graining(X))

            with h5py.File(config.data_path+f'/lemon_data_uld_beta_{config.beta:.2f}.h5', 'w') as f:
                f.create_dataset('x_fs', data=X)
                f.create_dataset('x_cg', data=cg_data)
                f.create_dataset('potential', data=V)
            X_full = X[:,::delta] 
            cg_data_full = cg_data[:,::delta] 
        else:    
            with h5py.File(config.data_path+f'/lemon_data_uld_beta_{config.beta:.2f}.h5', 'r') as f:
                X_full = f['x_fs'][()][:,::delta] 
                cg_data_full = f['x_cg'][()][:,::delta] 

        if config.beta >= 1.5:
            datasize = 40000
            test_ratio = datasize/X_full.shape[1]
        else:
            datasize = 40000
            test_ratio = datasize/X_full.shape[1]

    elif config.data_mode == 'genai':
        if config.new_data:
            with open("setting_ti.json", "r") as f:
                config_ti = json.load(f)
            X_full = np.load(config.data_path+f"/samples_epoch{config_ti['sampling_epoch']}_beta_{config.beta:.2f}.npy")[-1].T
            momenta = 1/np.sqrt(config.beta) * np.random.randn(2, X_full.shape[1])
            X_full = np.concatenate((X_full, momenta), axis=0)

            V = lemonslice.potential(X_full)

            cg_data_full = np.array(lemonslice.coarse_graining(X_full))

            with h5py.File(config.data_path+f'/lemon_data_genai_beta_{config.beta:.2f}.h5', 'w') as f:
                f.create_dataset('x_fs', data=X_full)
                f.create_dataset('x_cg', data=cg_data_full)
                f.create_dataset('potential', data=V)
        else:
            with h5py.File(config.data_path+f'/lemon_data_genai_beta_{config.beta:.2f}.h5', 'r') as f:
                X_full = f['x_fs'][()]
                cg_data_full = f['x_cg'][()]

        if config.beta >= 1.5:
            datasize = 50000
            test_ratio = datasize/X_full.shape[1]
        else:
            datasize = 30000
            test_ratio = datasize/X_full.shape[1]

    elif config.data_mode == 'rs':
        if config.new_data:
            X_full = rejection_sampling(lemonslice,size=1000_000)
            momenta = np.sqrt(1/config.beta) * np.random.randn(2, X_full.shape[1])
            X_full = np.concatenate((X_full, momenta), axis=0)

            V = lemonslice.potential(X_full)

            cg_data_full = np.array(lemonslice.coarse_graining(X_full))

            with h5py.File(config.data_path+f'/lemon_data_rs_beta_{config.beta:.2f}.h5', 'w') as f:
                f.create_dataset('x_fs', data=X_full)
                f.create_dataset('x_cg', data=cg_data_full)
                f.create_dataset('potential', data=V)

        else:
            with h5py.File(config.data_path+f'/lemon_data_rs_beta_{config.beta:.2f}.h5', 'r') as f:
                X_full = f['x_fs'][()]
                cg_data_full = f['x_cg'][()]

        if config.beta >= 1.5:
            datasize = 60000
            test_ratio = datasize/X_full.shape[1]
        else:
            datasize = 60000
            test_ratio = datasize/X_full.shape[1]

    return X_full, cg_data_full, test_ratio