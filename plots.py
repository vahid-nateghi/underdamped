import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from util.scriptutil import load_config

def _plots():

    input = load_config("", "setting_gedmd.json")

    # Setting for poster
    # plot_settings = {"axes.labelsize": 25, 'xtick.labelsize':25, 'ytick.labelsize':25,  
    #         'legend.fontsize':20, 'legend.loc':'upper right', 'savefig.dpi':300,
    #         'axes.titlesize':30,'axes.titlepad': 10} 
    
    # Setting for paper
    plot_settings = {"axes.labelsize": 20, 'xtick.labelsize':20, 'ytick.labelsize':20,  
            'legend.fontsize':15, 'legend.loc':'upper right', 'savefig.dpi':300,
            'axes.titlesize':20, 'axes.titlepad':10, 'font.family':'Latin Modern Math',
            'mathtext.fontset':'stix'}  
    mpl.rcParams.update(plot_settings)
    betas = [0.25, 0.5, 0.75, 1, 1.5, 2]

    plt.figure(figsize=[8, 6])
    for i, beta in enumerate(betas):
        with h5py.File(input.data_path+f'/lemon_uld_gedmd_beta_{beta:.2f}.h5', 'r') as f:
            eigs_uld_gedmd = f['eigs'][()]
        its_uld_gedmd = 1/eigs_uld_gedmd
        plt.errorbar(np.arange(1,4), np.mean(its_uld_gedmd,axis=0), yerr=np.std(its_uld_gedmd,axis=0),
                    fmt='o-', capsize=6, label=f'$\\beta$: {beta:.2f}')
    for i,beta in enumerate(betas):
        with h5py.File(input.data_path+f'/lemon_genai_gedmd_beta_{beta:.2f}.h5', 'r') as f:
            eigs_genai_gedmd = f['eigs'][()]
        its_genai_gedmd = 1/eigs_genai_gedmd
        plt.errorbar(np.arange(1,4), np.mean(its_genai_gedmd,axis=0), yerr=np.std(its_genai_gedmd,axis=0),
                    linestyle=(0, (3, 1, 1, 1, 1, 1)),  c=plt.rcParams['axes.prop_cycle'].by_key()['color'][i], capsize=6)
    plt.yscale('log')
    plt.xticks([1,2,3])
    plt.ylabel('Timescale')
    plt.xlabel('Index')
    plt.legend(bbox_to_anchor=(1.04, 0.75), loc="upper left")
    plt.tight_layout()
    plt.savefig(f'.figures/lemon_its_comparison_genai.png')
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4*1.8, 4.8*1.1))
    for i, beta in enumerate(betas):
        with h5py.File(input.data_path+f'/lemon_rs_gedmd_beta_{beta:.2f}.h5', 'r') as f:
            eigs_rs_gedmd = f['eigs'][()]
        its_rs_gedmd = 1/eigs_rs_gedmd
        ax1.errorbar(np.arange(1,4), np.mean(its_rs_gedmd,axis=0), yerr=np.std(its_rs_gedmd,axis=0),
                    fmt='o--', c=plt.rcParams['axes.prop_cycle'].by_key()['color'][i], capsize=6, label=f'$\\beta$: {beta:.2f}')
    for beta in betas:
        with h5py.File(input.data_path+f'/lemon_uld_gedmd_beta_{beta:.2f}.h5', 'r') as f:
            eigs_uld_gedmd = f['eigs'][()]
        its_uld_gedmd = 1/eigs_uld_gedmd
        uld_line = ax1.errorbar(np.arange(1,4), np.mean(its_uld_gedmd,axis=0), yerr=np.std(its_uld_gedmd,axis=0),
                    fmt='o-', capsize=6)
    for i, beta in enumerate(betas):
        its_uld_edmd = np.load(input.data_path+f'/lemon_edmd_uld_beta_{beta:.2f}.npy')
        edmd_line = ax1.errorbar(np.arange(1,4), np.mean(its_uld_edmd,axis=1), yerr=np.std(its_uld_edmd,axis=1),
                    fmt='o-.',  c='k',  capsize=6)
    for i, beta in enumerate(betas):
        with h5py.File(input.data_path+f'/lemon_uld_gedmd_beta_{beta:.2f}.h5', 'r') as f:
            eigs_uld_gedmd = f['eigs'][()]
        its_uld_gedmd = 1/eigs_uld_gedmd
        ax2.errorbar(np.arange(1,4), np.mean(its_uld_gedmd,axis=0), yerr=np.std(its_uld_gedmd,axis=0),
                    fmt='o-',  c=plt.rcParams['axes.prop_cycle'].by_key()['color'][i],  capsize=6, label=f'$\\beta$:{beta:.2f}')
        its_old_gedmd = np.load(input.data_path+f'/lemon_old_gedmd_beta_{beta:.2f}.npy')
        old_line = ax2.errorbar(np.arange(1,4), np.mean(its_old_gedmd,axis=0), yerr=np.std(its_old_gedmd,axis=0),
                    fmt='o:',  c=plt.rcParams['axes.prop_cycle'].by_key()['color'][i],  capsize=6) 
        
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax1.set_xticks([1,2,3])
    ax2.set_xticks([1,2,3])
    ax1.set_yticks([0.3,1,10,100])
    ax2.set_yticks([])
    ax1.set_ylim([0.3,100])
    ax2.set_ylim([0.3,100])
    ax1.set_ylabel('Timescale')
    ax1.set_xlabel('Index')
    ax2.set_xlabel('Index')
    ax1.set_title("A) Full vs CG Timescales")
    ax2.set_title('B) ULD vs OLD')
    
    uld_line = ax2.errorbar([], [], yerr=[], color='black', fmt='o-', capsize=6)
    old_line = ax2.errorbar([], [], yerr=[], color='black', fmt='o:', capsize=6)
    rs_line = ax2.errorbar([], [], yerr=[], color='black', fmt='o--', capsize=6)
    edmd_line = ax2.errorbar([], [], yerr=[], color='black', fmt='o-.', capsize=6)  
    legend2 = ax2.legend([uld_line, old_line, edmd_line, rs_line],
                                ['ULD','OLD', 'EDMD', 'RS'], 
                                bbox_to_anchor=(1.04, 1.0), loc='upper left')
    ax2.legend(bbox_to_anchor=(1.04, 0.62), loc="upper left")
    plt.gca().add_artist(legend2)

    fig.tight_layout()
    plt.savefig(f'.figures/lemon_its_comparison_merged.png')
    plt.close()

    with h5py.File(input.data_path+f'/lemon_data_rs_beta_1.00.h5', 'r') as f:
        X = f['x_fs'][()][:,::10]


    plot_settings_cg_coord = {"axes.labelsize": 15, 'xtick.labelsize':15, 'ytick.labelsize':15,  
            'savefig.dpi':300, 'axes.titlesize':15, 'axes.titlepad': 10} 
    mpl.rcParams.update(plot_settings_cg_coord)
    with np.load(input.data_path+f"/lemon_potential_uniform.npz") as f:
        X_uni = f['X']
        V_uni = f['V']

    fig, ax = plt.subplots()
    se = plt.contourf(X_uni[0],X_uni[1],V_uni, levels=50)
    plt.colorbar(se)
    plt.xlabel('$q_1$')
    plt.ylabel('$q_2$', rotation=0)
    plt.xlim([-2,2])
    plt.ylim([-2,2])
    ax.set_aspect('equal')
    plt.tight_layout(pad=0.1)
    plt.savefig(f'.figures/lemon_potential_position.png')
    plt.close()

    fig, [ax1,ax2] = plt.subplots(1,2, figsize=(6, 3))

    ax1.contourf(X_uni[0],X_uni[1],V_uni, levels=20)
    arc_radius = 1
    ax1.plot([-2, 2], [0, 0], color='k', lw=.5)
    ax1.plot([0, 0], [-2, 2], color='k', lw=.5)
    ax1.plot([0, 2], [0, 2], color='b', lw=2)
    ax1.arrow(0.72, 0.72, -0.2, 0.2, color='w', width=.005, head_width=0.1)
    ax1.text(0.6, 0.9, r'$\nabla\theta$', color='w', fontsize=15)
    circle = plt.Circle((0, 0), arc_radius, color='w', fill=False)
    ax1.add_patch(circle)
    ax1.text(0.25, 0.05, r'$\theta_0$', color='w', fontsize=15)
    xedges = np.linspace(-2,2,50)
    yedges = np.linspace(-2,2,50)
    H, _, _ = np.histogram2d(X[2], X[3], bins=(xedges, yedges)) 
    p_fs = -np.log(np.where(H == 0, 1e-15, H)).T 
    p_fs -= p_fs.min()
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    Xmesh, Ymesh = np.meshgrid(xcenters,ycenters)
    levels = np.arange(0., 4, 0.01) 
    ax2.contourf(Xmesh, Ymesh, p_fs, levels=levels) 
    ax2.plot([-2, 2], [0, 0], color='k', lw=.5)
    ax2.plot([0, 0], [-2, 2], color='k', lw=.5)
    ax2.plot([-2, 2], [-2, 2], color='b', lw=2)
    ax2.arrow(0.72, 0.72, -0.2, 0.2, color='w', width=.002, head_width=0.1)
    ax2.text(0.5, 0.9, r'$\nabla\theta$', color='w', fontsize=15)
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax1.set_xlabel('$q_1$')
    ax1.set_ylabel('$q_2$',rotation=0, labelpad=15)    
    ax2.set_xlabel('$p_1$')
    ax2.set_ylabel('$p_2$',rotation=0, labelpad=15)   
    ax2.set_xlim([-2,2])
    ax2.set_ylim([-2,2])
    ax1.set_xlim([-2,2])
    ax1.set_ylim([-2,2])
    ax1.set_xticks([0])
    ax1.set_yticks([0])
    ax2.set_xticks([0])
    ax2.set_yticks([0])
    ax1.set_title('A) Position space')
    ax2.set_title('B) Momentum space')
    plt.tight_layout(pad=1.5, w_pad=0, h_pad=1.0)
    plt.savefig(f'.figures/lemon_potential_phase.png')
    plt.close()

    mpl.rcParams.update(plot_settings)
    for beta in [1.,2.]:
        its_integ_edmd = np.load(input.data_path+f'/lemon_edmd_integ_beta_{beta:.2f}.npy')
        with h5py.File(input.data_path+f'/lemon_uld_gedmd_beta_{beta:.2f}.h5', 'r') as f:
            eigs_uld_gedmd = f['eigs'][()]
        its_uld_gedmd = 1/eigs_uld_gedmd
        its_rs_gedmd_learned = np.load(input.data_path+f'/lemon_rs_learned_its_beta_{beta:.2f}.npy')

        with h5py.File(input.data_path+f'/lemon_prop_its_beta_{beta:.2f}.h5', 'r') as f:
            its_integ_gedmd = f['its'][()]

        plt.figure()
        plt.errorbar(np.arange(1,4), np.mean(its_uld_gedmd,axis=0), yerr=np.std(its_uld_gedmd,axis=0),
                    fmt='o-', capsize=6, label='Orig. data')
        plt.errorbar(np.arange(1,4), np.mean(its_rs_gedmd_learned,axis=1), yerr=np.std(its_rs_gedmd_learned,axis=1),
                    fmt='o-', capsize=6, label='Orig. data + learned diff.')
        plt.errorbar(np.arange(1,4), np.mean(its_integ_gedmd,axis=0), yerr=np.std(its_integ_gedmd,axis=0),
                    fmt='o-', capsize=6, label='Integ. data')
        plt.errorbar(np.arange(1,4), np.mean(its_integ_edmd,axis=1), yerr=np.std(its_integ_edmd,axis=1),
                    fmt='o-', capsize=6, label='EDMD (Integ. data)')

        plt.xticks([1,2,3])
        plt.ylabel('$t_i$', rotation=0)
        plt.xlabel('Index')
        plt.legend(loc=3)
        plt.title('D) Timescales')
        plt.tight_layout()
        plt.savefig(f'.figures/lemon_its_comparison_prop_beta_{beta:.2f}.png')
        plt.close()

    its_gedmd = np.load(input.data_path+f'/lemon_gedmd_diff_mass_analysis_equalmass.npy')
    its_edmd = np.load(input.data_path+f'/lemon_edmd_diff_mass_analysis_equalmass.npy')

    plt.figure()
    plt.errorbar(np.arange(1,4), np.mean(its_gedmd, axis=1), yerr=np.std(its_gedmd, axis=1),
        fmt='o-', capsize=6, label=f'gEDMD')
    plt.errorbar(np.arange(1,4), np.mean(its_edmd, axis=1), yerr=np.std(its_edmd, axis=1),
        fmt='o--', capsize=6, label=f'EDMD')
    plt.xlabel('Index')
    plt.xticks(np.arange(1, 4))
    plt.yticks(np.arange(2,16,2))
    plt.ylabel('Timescale')
    plt.legend()
    plt.title('C) Equal masses')
    plt.tight_layout()
    plt.savefig(f'.figures/lemon_its_equalmass.png')
    plt.close()

    its_gedmd = np.load(input.data_path+f'/lemon_gedmd_diff_mass_analysis_diffmass.npy')
    its_edmd = np.load(input.data_path+f'/lemon_edmd_diff_mass_analysis_diffmass.npy')

    plt.figure()
    plt.errorbar(np.arange(1,4), np.mean(its_gedmd, axis=1), yerr=np.std(its_gedmd, axis=1),
        fmt='o-', capsize=6, label=f'gEDMD')
    plt.errorbar(np.arange(1,4), np.mean(its_edmd, axis=1), yerr=np.std(its_edmd, axis=1),
        fmt='o--', capsize=6, label=f'EDMD')
    plt.xlabel('Index')
    plt.xticks(np.arange(1, 4))
    plt.yticks(np.arange(2,16,2))
    plt.ylabel('Timescale')
    plt.legend()
    plt.title('D) Unequal masses')
    plt.tight_layout()
    plt.savefig(f'.figures/lemon_its_diffmass.png')
    plt.close()

    with h5py.File(input.data_path+f'/lemon_data_uld_test_diffmass.h5', 'r') as f:
        cg_data_full = f['x_cg'][()][:,::50] 
    xedges = np.linspace(-np.pi,np.pi,50)
    yedges = np.linspace(-6,6,50)
    H, _, _ = np.histogram2d(*cg_data_full[:2], bins=(xedges, yedges), density=True)  
    V_cg = -np.log(np.where(H == 0, 1e-15, H)).T 
    V_cg -= V_cg.min()
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2

    Xmesh, Ymesh = np.meshgrid(xcenters,ycenters)
    
    plt.figure()
    levels = np.arange(0., 11, 0.5) 
    cc = plt.contourf(Xmesh, Ymesh, V_cg, levels=levels) 
    plt.ylim([-6,6])
    plt.xticks([-3/4*np.pi, -np.pi/4, np.pi/4, 3*np.pi/4],
               ['$-3\\pi/4$','$-\\pi/4$','$\\pi/4$','$3\\pi/4$'])
    plt.xlabel('z')
    plt.ylabel('v', rotation=0)
    plt.colorbar(cc, ticks=np.arange(0,11,1))
    plt.title('B) Free Energy - unequal masses')
    plt.tight_layout()
    plt.savefig(f'.figures/lemon_ph_diffmass.png')
    plt.close()

    with h5py.File(input.data_path+f'/lemon_data_{input.data_mode}_beta_1.00.h5', 'r') as f:
        cg_data_full = f['x_cg'][()][:,::50] 

    H, _, _ = np.histogram2d(*cg_data_full[:2], bins=(xedges, yedges), density=True)  
    V_cg = -np.log(np.where(H == 0, 1e-15, H)).T 
    V_cg -= V_cg.min()
    plt.figure()
    levels = np.arange(0., 11, 0.5) 
    cc = plt.contourf(Xmesh, Ymesh, V_cg, levels=levels) 
    plt.ylim([-6,6])
    plt.xticks([-3/4*np.pi, -np.pi/4, np.pi/4, 3*np.pi/4],
               ['$-3\\pi/4$','$-\\pi/4$','$\\pi/4$','$3\\pi/4$'])
    plt.xlabel('z')
    plt.ylabel('v', rotation=0)
    plt.colorbar(cc, ticks=np.arange(0,11,1))
    plt.title('A) Free Energy - equal masses')
    plt.tight_layout()
    plt.savefig(f'.figures/lemon_ph_equalmass.png')
    plt.close()

    fig, ax = plt.subplots(1,2, figsize=(10,5))
    gamma_list = [0.1, 0.3, 0.7, 1, 1.3, 5]
    p_list = [50, 100, 200, 500]
    for i, beta in enumerate(betas):
        dtest_mat = np.load(input.data_path+f'/lemon_vamp_gamma_beta_{beta:.2f}.npz')['dtest_mat']
        ax[0].errorbar(np.arange(len(gamma_list)),np.mean(-np.sum(np.real(dtest_mat),axis=0),axis=1),
            yerr=np.std(-np.sum(np.real(dtest_mat), axis=0),axis=1), fmt='o-', capsize=6)
    for i, beta in enumerate(betas):
        dtest_mat = np.load(input.data_path+f'/lemon_vamp_feat_num_beta_{beta:.2f}.npz')['dtest_mat']
        ax[1].errorbar(np.arange(len(p_list)),np.mean(-np.sum(np.real(dtest_mat),axis=0),axis=1),
            yerr=np.std(-np.sum(np.real(dtest_mat), axis=0),axis=1), fmt='o-', capsize=6, label=f'$\\beta= {i:.2f}$')
    ax[0].set_title('A) Kernel Bandwidth')
    ax[1].set_title('B) Number of Features')
    ax[0].set_ylabel('Score')
    ax[0].set_xlabel('$\\gamma$')
    ax[1].set_xlabel('p')
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[0].set_ylim([8e-2,1e+2])
    ax[1].set_ylim([8e-2,1e+2])
    ax[0].set_xticks(np.arange(len(gamma_list)), [str(p) for p in gamma_list])
    ax[1].set_xticks(np.arange(len(p_list)), [str(p) for p in p_list])
    ax[1].set_yticks([])
    plt.legend(bbox_to_anchor=(1.04, 0.75), loc="upper left")
    plt.tight_layout()
    plt.savefig('.figures/lemon_vamp.png')
    plt.close()

    for beta in [1., 2.]:
        with h5py.File(input.data_path+f'/lemon_rs_gedmd_beta_{beta:.2f}.h5', 'r') as f:
            cg_datalist = [*f['cg_datalist'][()]]
            Omegalist = f['Omega_list'][()]
            mylist = f['transformation_mat_list']
            transformation_mat_list = [mylist[name][...] for name in mylist]
            
        if beta == 2:
            Omega = Omegalist[3]
            U = transformation_mat_list[3] 
        elif beta == 1:
            Omega = Omegalist[0] 
            U = transformation_mat_list[0] 

        with open(input.data_path+f'/lemon_rs_eff_dyn_beta_{beta:.2f}.pkl', 'rb') as f:
            lemon_eff_dyn = pickle.load(f)
        alpha_b = lemon_eff_dyn['alpha_b']
        b_loc_nn = lemon_eff_dyn['b_loc_nn']
        b_loc_y_scaler = lemon_eff_dyn['b_loc_y_scaler']
        b_loc_x2_scaler = lemon_eff_dyn['b_loc_x2_scaler']
        alpha_sigma = lemon_eff_dyn['alpha_a']
        a_loc_nn = lemon_eff_dyn['a_loc_nn']
        a_loc_y_scaler = lemon_eff_dyn['a_loc_y_scaler']
        a_loc_x2_scaler = lemon_eff_dyn['a_loc_x2_scaler']
        a_loc_ref = lemon_eff_dyn['a_loc_ref']
        b_loc_ref = lemon_eff_dyn['b_loc_ref']

        yedge= 3/np.sqrt(beta) + 3

        y = np.mgrid[-np.pi:np.pi:80j, -yedge:yedge:80j]
        y = y.reshape((2,6400))
        arg = np.dot(y.T, Omega)
        M0 =  np.hstack((np.cos(arg),np.sin(arg)))
        M0_red = M0 @ np.real(U)
        a_ed = abs(alpha_sigma @ M0_red.T)
        
        x1_sin = np.sin(y[0])
        x1_cos = np.cos(y[0])
        x2_scaled = a_loc_x2_scaler.transform(y[1].reshape(-1,1))
        X_processed = np.concatenate((x1_sin[:,None], x1_cos[:,None], x2_scaled),axis=1)

        cg_data_tor = torch.tensor(X_processed, dtype = torch.float32)
        a_ed_nn = np.squeeze(a_loc_y_scaler.inverse_transform(a_loc_nn(cg_data_tor).detach()))

        mask = np.abs(y[1]) < 2/np.sqrt(beta)
        a_ed_total = a_ed_nn.copy()
        a_ed_total[mask] = a_ed[mask]

        f_ed = (alpha_b[None, :] @ M0_red.T)[0]

        x2_scaled = b_loc_x2_scaler.transform(y[1].reshape(-1,1))
        X_processed = np.concatenate((x1_sin[:,None], x1_cos[:,None], x2_scaled),axis=1)
        b_app =  np.squeeze(b_loc_y_scaler.inverse_transform(b_loc_nn(torch.tensor(X_processed, dtype = torch.float32)).detach().numpy()))

        mask = np.abs(y[1]) < 2/np.sqrt(beta)
        b_ed_total = b_app.copy()
        b_ed_total[mask] = f_ed[mask]

        fig, ax = plt.subplots(2,2, figsize=(12, 10)) 
        se = ax[0,0].scatter(cg_datalist[0][0], cg_datalist[0][1], c= a_loc_ref, vmax=8)
        ax[0,1].scatter(y[0], y[1], c= a_ed_total, vmax=8)
        se_d = ax[1,0].scatter(cg_datalist[0][0], cg_datalist[0][1], c= b_loc_ref, vmax=10, vmin=-10)
        ax[1,1].scatter(y[0], y[1], c= b_ed_total, vmax=10, vmin=-10) 
        ax[1,0].set_xlabel('z')
        ax[1,1].set_xlabel('z')
        ax[0,0].set_ylabel('v', rotation=0)
        ax[1,0].set_ylabel('v', rotation=0)
        ax[0,0].set_xlim([-np.pi,np.pi])
        ax[0,0].set_ylim([-yedge,yedge])
        ax[0,1].set_xlim([-np.pi,np.pi])
        ax[0,1].set_ylim([-yedge,yedge])
        ax[1,0].set_xlim([-np.pi,np.pi])
        ax[1,0].set_ylim([-yedge,yedge])
        ax[1,1].set_xlim([-np.pi,np.pi])
        ax[1,1].set_ylim([-yedge,yedge])
        ax[1,0].set_xticks([-3/4*np.pi, -np.pi/4, np.pi/4, 3*np.pi/4],
                ['$-3\\pi/4$','$-\\pi/4$','$\\pi/4$','$3\\pi/4$'])
        ax[1,1].set_xticks([-3/4*np.pi, -np.pi/4, np.pi/4, 3*np.pi/4],
                ['$-3\\pi/4$','$-\\pi/4$','$\\pi/4$','$3\\pi/4$'])
        ax[0,0].set_xticks([])
        ax[0,1].set_xticks([])
        fig.colorbar(se, ax=ax[0,0]) 
        fig.colorbar(se, ax=ax[0,1]) 
        fig.colorbar(se_d, ax=ax[1,0]) 
        fig.colorbar(se_d, ax=ax[1,1]) 
        ax[0,0].set_title('A) Local Diffusion')
        ax[0,1].set_title('B) Learned Diffusion')
        ax[1,0].set_title('C) Local Force')
        ax[1,1].set_title('D) Learned Force')
        plt.tight_layout()
        plt.savefig(f'.figures/lemon_learning_beta_{beta:.2f}.png')
        plt.close()

        with h5py.File(input.data_path+f'/lemon_prop_beta_{beta:.2f}.h5', 'r') as f:
            x_prop_list = [*f['x_prop'][()]]

        n_traj = len(x_prop_list)

        fig, ax = plt.subplots(n_traj)

        for i in range(n_traj):
            ax[i].plot(x_prop_list[i][0,::1])
            ax[i].axhline(y=np.pi/4, color='r', linestyle='--')
            ax[i].axhline(y=3*np.pi/4, color='r', linestyle='--')
            ax[i].axhline(y=-np.pi/4, color='r', linestyle='--')
            ax[i].axhline(y=-3*np.pi/4, color='r', linestyle='--')
            ax[i].set_yticks([-np.pi,0,np.pi], ['$-\\pi$','0','$\\pi$'])
            ax[i].set_xticks([])
            ax[i].set_ylabel('z',rotation=0)
        ax[-1].set_xticks(np.linspace(0,x_prop_list[i].shape[1],6))
        ax[-1].set_xlabel('t')
        ax[0].set_title('A) CG Position')
        plt.tight_layout()
        plt.savefig(f'.figures/x_prop_list_beta_{beta:.2f}.png')
        plt.close()

        xedges = np.linspace(-np.pi,np.pi,50)
        yedges = np.linspace(-6,6,50)
        H, _, _ = np.histogram2d(*np.concatenate(x_prop_list,axis=1), bins=(xedges, yedges), density=True)  
        V_cg = -np.log(np.where(H == 0, 1e-15, H)).T 
        V_cg -= V_cg.min()
        xcenters = (xedges[:-1] + xedges[1:]) / 2
        ycenters = (yedges[:-1] + yedges[1:]) / 2
        Xmesh, Ymesh = np.meshgrid(xcenters,ycenters)
        plt.figure()
        levels = np.arange(0., 11, 0.5) 
        cc = plt.contourf(Xmesh, Ymesh, V_cg, levels=levels) # , s=1
        plt.xlim([-np.pi,np.pi])
        plt.ylim([-3/np.sqrt(beta) - 3,3/np.sqrt(beta) + 3])
        plt.xticks([-3/4*np.pi, -np.pi/4, np.pi/4, 3*np.pi/4],
                ['$-3\\pi/4$','$-\\pi/4$','$\\pi/4$','$3\\pi/4$'])
        plt.xlabel('z')
        plt.ylabel('v', rotation=0)
        plt.colorbar(cc, ticks=np.arange(0,11,1))
        plt.title('B) Free Energy')
        plt.tight_layout()
        plt.savefig(f'.figures/x_prop_ph_beta_{beta:.2f}.png')
        plt.close()

        with h5py.File(input.data_path+f'/lemon_prop_its_beta_{beta:.2f}.h5', 'r') as f:
            chilist = [*f['chilist'][()]]
            x_proplist = [*f['x_proplist'][()]]
        plt.figure()
        marker = ['o', '^', '*', 's']
        for ii in range(4):
            ind = np.where(chilist[0][:,ii] >= 0.7)[0]
            plt.scatter(x_proplist[0][0,ind],x_proplist[0][1, ind], marker=marker[ii])     
        plt.ylim([-3/np.sqrt(beta) - 3,3/np.sqrt(beta) + 3])
        plt.xticks([-3/4*np.pi, -np.pi/4, np.pi/4, 3*np.pi/4],
                ['$-3\\pi/4$','$-\\pi/4$','$\\pi/4$','$3\\pi/4$'])
        plt.xlabel('z')
        plt.ylabel('v', rotation=0)
        plt.title('C) PCCA States')
        plt.tight_layout()
        plt.savefig(f'.figures/lemon_prop_pcca_beta_{beta:.2f}.png')
        plt.close()

    for beta in [1.5, 2]:
        samples = np.load(input.data_path+f"/samples_epoch75_beta_{beta:.2f}.npy")
        weights = np.load(input.data_path+f"/weights_epoch75_beta_{beta:.2f}.npy")

        df = pd.read_csv(".dataset/test.csv", header=[0, 1])

        cmap = "viridis"
        fig, axs = plt.subplots(1, 3, figsize=(10, 4), sharex=True, sharey=True, dpi=300)
        axs = axs.flatten()

        # plot reference
        ref_x = df[f"{beta:.2f}"]["x"].values[:len(samples[-1])]
        ref_y = df[f"{beta:.2f}"]["y"].values[:len(samples[-1])]

        n_bins = 100
        hist_ref = np.histogram2d(ref_x, ref_y, bins=n_bins, range=[[-1.5, 1.5], [-1.5, 1.5]], density=True)

        axs[0].imshow(hist_ref[0].T, origin="lower", cmap=cmap, extent=[-1.5, 1.5, -1.5, 1.5], vmin=0, vmax=1, aspect="auto")
        axs[0].set_title(rf"Reference")
        axs[0].set_xlabel('$q_1$')
        axs[0].set_ylabel('$q_2$', rotation=0)

        # plot samples
        model_x = samples[-1, :, 0]
        model_y = samples[-1, :, 1]
        hist = np.histogram2d(model_x, model_y, bins=n_bins, range=[[-1.5, 1.5], [-1.5, 1.5]], density=True)

        axs[1].imshow(hist[0].T, origin="lower", cmap=cmap, extent=[-1.5, 1.5, -1.5, 1.5], vmin=0, vmax=1, aspect="auto")
        axs[1].set_xlim(-1.5, 1.5)
        axs[1].set_ylim(-1.5, 1.5)
        axs[1].set_title(rf"Model")
        axs[1].set_xlabel('$q_1$')

        # plot weighted samples
        hist_weighted = np.histogram2d(model_x, model_y, bins=n_bins, range=[[-1.5, 1.5], [-1.5, 1.5]], density=True, weights=weights)
        axs[2].imshow(hist_weighted[0].T, origin="lower", cmap=cmap, extent=[-1.5, 1.5, -1.5, 1.5], vmin=0, vmax=1, aspect="auto")
        axs[2].set_xlim(-1.5, 1.5)
        axs[2].set_ylim(-1.5, 1.5)
        axs[2].set_title(rf"Model (weighted)")
        axs[2].set_xlabel('$q_1$')
        fig.suptitle(rf"$\beta$={beta}")    
        plt.tight_layout()
        plt.savefig(f'.figures/lemon_genai_histo_beta_{beta:.2f}.png')
        plt.close()

if __name__ == "__main__":
    cwd = Path.cwd()
    Path(cwd/".figures/").mkdir(parents=True, exist_ok=True)

    _plots()
