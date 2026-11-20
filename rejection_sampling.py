import numpy as np

def rejection_sampling(lemon_instance,size):

    batch_size = 10000
    proposal_bounds = [[-2.0, 2.0], [-2.0, 2.0]]
    xmin, xmax = proposal_bounds[0]
    ymin, ymax = proposal_bounds[1]
    samples = []
    xs = []
    while len(samples) < size:
        x_prop = np.random.uniform(low=xmin, high=xmax, size=batch_size)
        y_prop = np.random.uniform(low=ymin, high=ymax, size=batch_size)
        proposals = np.vstack((x_prop, y_prop)) # Shape (2, batch_size)

        u_prime = np.random.uniform(low=0, high=1, size=batch_size) # np.exp(lemon_instance.beta*lemon_instance.multiplier)

        accepted_mask = u_prime < lemon_instance.distribution(proposals)
        accepted_samples_batch = proposals[:, accepted_mask].T # Shape (n_accepted, 2)
            
        samples.extend(accepted_samples_batch.tolist())
        xs.extend(proposals.T.tolist())

    xs_acc = np.array(samples[:size])
    
    return xs_acc.T
