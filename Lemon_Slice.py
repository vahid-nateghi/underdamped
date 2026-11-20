import numpy as np
import torch
""" Implements 2d-Lemon Slice potential as model system: """

class LemonSlice:
    """
        Stochastic dynamics in two-dimensional Lemon Slice potential, given by:

        V(x, y) = mlt * (c * cos(k*phi) + 10*(r-1)**2)

        where r, phi are the polar coordinates of x, y.

        k, int:
            number of minima along the polar axis.
        beta, float:
            inverse temperature
        gamma, float:
            friction term.
        c, float:
            scaling factor for cosine part of the potential.
        multiplier, float:
            scaling factor for the potential.
        d, int (optional):
            dimension of the system. If not equal to two, the potential
            will just be harmonic along all additional dimensions.
        mass, ndarray:
            1-d array of the system mass
    """

    def __init__(self, k: int = 4, beta=1.0, c=1.0, d: int = 2, gamma=1.0, multiplier=1.0, mass=None):
        self.k = k
        self.beta = beta
        self.c = c
        self.d = d
        self.gamma = gamma
        self.multiplier = multiplier
        if mass is not None:
            self.mass = mass
        else:
            self.mass = np.ones((self.d,))

    def _polar_rep(self, x, y):
        """
            Compute polar coordinates from 2d Euclidean coordinates:

            x, y, nd-array (m):
                Arrays of two-dimensional Euclidean coordinates to be transformed.

            Returns:
            --------
            r, phi, nd-array (m):
                Arrays of polar coordinates corresponding to x and y.
        """
        r = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        
        return r, phi
    
    def cg_map_grad(self, x):
        """
        Gradient of the cg map.
        
        """
        r = np.sqrt(x[0] ** 2 + x[1] ** 2)
        cg_map_grad = np.zeros((2,4, x.shape[1]))
        cg_map_grad[0] = np.concatenate((np.array([-x[1]/r**2,x[0]/r**2]) ,np.zeros((2,x.shape[1]))), axis=0)
        cg_map_grad_aux = np.array([(2* x[1] * x[0] * x[2] + x[3] * (x[1]**2-x[0]**2))/ r**4/self.mass[0], ( -2* x[1] * x[0] * x[3] + x[2] * (x[1]**2-x[0]**2))/ r**4/self.mass[1]])
        cg_map_grad[1] = np.concatenate((cg_map_grad_aux, np.array([-x[1]/r**2/self.mass[0],x[0]/r**2/self.mass[1]])), axis=0)

        return cg_map_grad
    
    def potential(self, x):
        """
            Evaluate potential energy at Euclidean positions x

            x, nd-array (2d, m):
                Arrays of Euclidean coordinates as well as their generalized momenta.

            Returns:
            --------
            V, nd-array (m,):
                Values of the potential for all pairs of x-y-values.
        """
        # Transform first two dimensions to polar coordinates and compute "Lemon Slice"
        # part of the potential:
        r, phi = self._polar_rep(x[0, :], x[1, :])
        V = self.multiplier * (self.c * np.cos(self.k * phi) + 10 * (r - 1) ** 2) + self.multiplier

        return V

    def distribution(self,x):

        mu = np.exp(-self.beta * self.potential(x))

        return mu
        
    def gradient(self, x):
        """
            Evaluate the gradient of the potential
            x, nd-array (d, m):
                Arrays of Euclidean coordinates.

            Returns:
            --------
            grad_V, ndarray (d,m):
                The potential gradient for all m data points in x.

        """
        r, phi = self._polar_rep(x[0, :], x[1, :])  

        grad_V = np.zeros((2,x.shape[1]))
        grad_V[0] = self.multiplier * (+self.k * np.sin(self.k * phi) * (x[1] / r ** 2) + 20 * (r - 1) * (x[0] / r)) 
        grad_V[1] = self.multiplier * (-self.k * np.sin(self.k * phi) * (x[0] / r ** 2) + 20 * (r - 1) * (x[1] / r))

        return grad_V

    def drift(self, x):
        """
            Evaluate the drift for the SDE governing the underdamped Langevin dynamics 

            x, nd-array (2d, m):
                Arrays of Euclidean coordinates as well as their generalized momenta.

            Returns:
            --------
            dx, nd-array (2d, m)
                The drift for all m data points in x.
        """       
        dV = np.zeros((2 * self.d, x.shape[1]))
        # Transform first two dimensions to polar coordinates and compute "Lemon Slice"
        # part of the gradient:
        r, phi = self._polar_rep(x[0, :], x[1, :])  
        dV[0, :] = x[2]/self.mass[0]
        dV[1, :] = x[3]/self.mass[1]
        dV[2, :] = self.multiplier * (-self.c * self.k * np.sin(self.k * phi) * (x[1] / r ** 2) - 20 * (r - 1) * (x[0] / r)) - self.gamma * x[2]
        dV[3, :] = self.multiplier * (+self.c * self.k * np.sin(self.k * phi) * (x[0] / r ** 2) - 20 * (r - 1) * (x[1] / r)) - self.gamma * x[3]

        return dV

    def Simulate(self, x0, m, dt):
        """
            Generate trajectory of stochastic dynamics in Lemon Slice potential, using Euler scheme.

        x0, nd-array(2d):
            initial values for the simulation.
        m, int:
            number time steps to be returned (including initial value)
        dt, float:
            integration time step

        Returns:
        --------
        X, nd-array (2d, m):
            simulation trajectory
        """
        # Initialize:
        X = np.zeros((2*self.d, m))
        X_old = x0[:, None]

        #print(X_old)
        X[:, 0] = X_old[:, 0]
        # Run simulation:
        sigma = np.sqrt(2 * self.gamma / self.beta) * np.eye(self.d) @ np.diag(np.sqrt(self.mass))
        sigma_diff = np.block([[np.zeros((self.d, self.d)), np.zeros((self.d, self.d))],[np.zeros((self.d, self.d)), sigma]])

        for t in range(1, m):
            X_new = X_old + self.drift(X_old) * dt + np.sqrt(dt) * sigma_diff @ np.random.randn(2*self.d, 1)
            X[:, t] = X_new[:, 0]
            X_old = X_new
        return X
    
    def Simulate_CG(self, alpha_b, nn_data, alpha_sigma, Omega, transform_mat, x0, m:int, dt):
        """
            Generate trajectory of stochastic dynamics in Lemon Slice potential, using Euler scheme.

        alpha_b, nd-array(1d):
            coefficients to evaluate the drift
        alpha_sigma, function:
            coefficients to evaluate the diffusion
        Omega, nd-array(2d):
            Random Fourier features
        x0, nd-array(2d):
            initial values for the simulation.
        m, int:
            number time steps to be returned (including initial value)
        dt, float:
            integration time step
        Returns:
        --------
        X, nd-array (2d, m):
            simulation trajectory
        """
        b_loc_nn = nn_data['b_loc_nn']
        b_loc_y_scaler = nn_data['b_loc_y_scaler']
        b_loc_x2_scaler = nn_data['b_loc_x2_scaler']
        a_loc_nn = nn_data['a_loc_nn']
        a_loc_y_scaler = nn_data['a_loc_y_scaler']
        a_loc_x2_scaler = nn_data['a_loc_x2_scaler']
        # Initialize:
        d = int(len(x0)/2)
        X = np.zeros((len(x0), m))
        drift = np.zeros((1, m))
        diffusion = np.zeros((1, m))
        X[:, 0] = x0
        # Run simulation:
        t = 1
        while t < m:
            if  np.abs(X[1,t-1]) > 2/np.sqrt(self.beta): 
                x1_sin = np.sin(X[0, t-1])
                x1_cos = np.cos(X[0, t-1])
                x2_scaled = np.squeeze(a_loc_x2_scaler.transform(X[1, t-1].reshape(-1,1)))
                x_processed = np.array([[x1_sin, x1_cos, x2_scaled]])

                x_processed_tor = torch.tensor(x_processed, dtype = torch.float32)
                sigma = np.sqrt(np.squeeze(a_loc_y_scaler.inverse_transform(a_loc_nn(x_processed_tor).detach())[:,0]))

                x2_scaled = np.squeeze(b_loc_x2_scaler.transform(X[1, t-1].reshape(-1,1)))
                x_processed = np.array([[x1_sin, x1_cos, x2_scaled]])

                grad_V = np.squeeze(b_loc_y_scaler.inverse_transform(b_loc_nn(torch.tensor(x_processed, dtype = torch.float32)).detach().numpy()))
                b_ed = np.array([X[1,t-1], grad_V])
            else:
                arg = np.dot(X[:, t-1][None,:], Omega)
                M0 =  np.hstack((np.cos(arg),np.sin(arg)))
                M0_red = transform_mat.T @ M0.T
                sigma = np.sqrt(np.squeeze(abs(alpha_sigma[None,:] @ M0_red)))
                b_ed = np.array([X[1,t-1], np.squeeze(alpha_b[None,:] @ M0_red)]) 
            
            sigma_ed = np.array([[0,0],[0,sigma]])


            X_new = X[:, t-1] + b_ed * dt + np.sqrt(2*dt/self.beta) * np.squeeze(sigma_ed @ np.random.randn(2*d, 1))

            X_new[0] = (X_new[0] + np.pi) % (2 * np.pi) - np.pi

            X[:, t] = X_new
            drift[:, t] = b_ed[1]
            diffusion[:, t] = sigma_ed[1,1]**2
            
            t += 1
           
        return X, drift, diffusion

    def coarse_graining(self, x):
        """
            Computing the coarse-grained coordinates.

        x, nd-array(2d,m):
            full space simulated data.
        Returns:
        --------
        [z,v], nd-array (2, m):
            coarse-grained trajectory
        """
        r ,z = self._polar_rep(x[0], x[1])
        v = np.einsum('ij,ij->j', np.array([-x[1]/r**2,x[0]/r**2]), np.array([x[2]/self.mass[0],x[3]/self.mass[1]]))

        return [z,v]

