import numpy as np


def ESS_sampling(log_L, Sigma, f, **hyperparams):
    nu = np.random.multivariate_normal(mean=np.zeros_like(f), cov=Sigma)
    u = np.random.rand()
    log_y = log_L(f, **hyperparams) + np.log(u)
    # print(log_y)
    # print('1', f)
    theta = np.random.rand() * 2 * np.pi
    theta_min = theta - 2 * np.pi
    theta_max = theta
    f_prime = f * np.cos(theta) + nu * np.sin(theta)
    # print('2', f)
    while log_L(f_prime, **hyperparams) <= log_y:
        if theta < 0:
            theta_min = theta
        else:
            theta_max = theta
        theta = np.random.uniform(low=theta_min, high=theta_max)
        # print('3', f)
        f_prime = f * np.cos(theta) + nu * np.sin(theta)
        # print(log_L(f_prime, **hyperparams))
        # print('4', f)
    # import pdb; pdb.set_trace()
    return f_prime

def ESS_sampling_s(log_L, nu, f, **hyperparams):
    u = np.random.rand()
    log_y = log_L(f, **hyperparams) + np.log(u)
    # print(log_y)
    # print('1', f)
    theta = np.random.rand() * 2 * np.pi
    theta_min = theta - 2 * np.pi
    theta_max = theta
    f_prime = f * np.cos(theta) + nu * np.sin(theta)
    # print('2', f)
    while log_L(f_prime, **hyperparams) <= log_y:
        if theta < 0:
            theta_min = theta
        else:
            theta_max = theta
        theta = np.random.uniform(low=theta_min, high=theta_max)
        # print('3', f)
        f_prime = f * np.cos(theta) + nu * np.sin(theta)
        # print(log_L(f_prime, **hyperparams))
        # print('4', f)
    # import pdb; pdb.set_trace()
    return f_prime


def MH(log_pos, x, scale=0.1, **hyperparams):
    n_dim = x.shape[0]
    x_cand = x + np.random.randn(n_dim)*scale
    temp = log_pos(x_cand, **hyperparams) - log_pos(x, **hyperparams)
    # import pdb; pdb.set_trace()
    alpha = np.min([1., np.exp(temp)])
    u = np.random.rand()
    if u < alpha:
        return x_cand, 1
    else:
        return x, 0