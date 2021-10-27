import scipy, h5py
import numpy as np
from scipy.signal import resample
from utils.cov_util import calc_cross_cov_mats_from_data
from utils.plotting.fig1 import lorenz_fig_axes, plot_3d, plot_lorenz_3d, plot_traces, plot_dca_demo, plot_r2, plot_cov
import matplotlib.pyplot as plt
from dca import DynamicalComponentsAnalysis as DCA
from utils.plotting import style
import pickle


RESULTS_FILENAME = "../data/lorenz/lorenz_results.hdf5"


def gen_lorenz_system(T, integration_dt=0.005):
    """
    Period ~ 1 unit of time (total time is T)
    So make sure integration_dt << 1
    Known-to-be-good chaotic parameters
    See sussillo LFADS paper
    """
    rho = 28.0
    sigma = 10.0
    beta = 8 / 3.

    def dx_dt(state, t):
        x, y, z = state
        x_dot = sigma * (y - x)
        y_dot = x * (rho - z) - y
        z_dot = x * y - beta * z
        return (x_dot, y_dot, z_dot)

    x_0 = np.ones(3)
    t = np.arange(0, T, integration_dt)
    X = scipy.integrate.odeint(dx_dt, x_0, t)
    return X


def gen_lorenz_data(num_samples, normalize=True):
    integration_dt = 0.005
    data_dt = 0.025
    skipped_samples = 1000
    T = (num_samples + skipped_samples) * data_dt
    X = gen_lorenz_system(T, integration_dt)
    if normalize:
        X -= X.mean(axis=0)
        X /= X.std(axis=0)
    X_dwn = resample(X, num_samples + skipped_samples, axis=0)
    X_dwn = X_dwn[skipped_samples:, :]
    return X_dwn


def random_basis(N, D, rng):
    return scipy.stats.ortho_group.rvs(N, random_state=rng)[:, :D]


def median_subspace(N, D, rng, num_samples=5000, V_0=None):
    subspaces = np.zeros((num_samples, N, D))
    angles = np.zeros((num_samples, min(D, V_0.shape[1])))
    if V_0 is None:
        V_0 = np.eye(N)[:, :D]
    for i in range(num_samples):
        subspaces[i] = random_basis(N, D, rng)
        angles[i] = np.rad2deg(scipy.linalg.subspace_angles(V_0, subspaces[i]))
    median_angles = np.median(angles, axis=0)
    median_subspace_idx = np.argmin(np.sum((angles - median_angles)**2, axis=1))
    median_subspace = subspaces[median_subspace_idx]
    return median_subspace


def gen_noise_cov(N, D, var, rng, V_noise=None):
    noise_spectrum = var * np.exp(-2 * np.arange(N) / D)
    if V_noise is None:
        V_noise = scipy.stats.ortho_group.rvs(N, random_state=rng)
    noise_cov = np.dot(V_noise, np.dot(np.diag(noise_spectrum), V_noise.T))
    return noise_cov


def embedded_lorenz_cross_cov_mats(N, T, snr=1., noise_dim=7, return_samples=False,
                                   num_lorenz_samples=10000, num_subspace_samples=5000,
                                   V_dynamics=None, V_noise=None, X_dynamics=None, seed=20200326):
    """Embed the Lorenz system into high dimensions with additive spatially
    structued white noise. Signal and noise subspaces are oriented with the
    median subspace angle.
    Parameters
    ----------
    N : int
        Embedding dimension.
    T : int
        Number of timesteps (2 * T_pi)
    snr : float
        Signal-to-noise ratio. Specifically it is the ratio of the largest
        eigenvalue of the signal covariance to the largest eigenvalue of the
        noise covariance.
    noise_dim : int
        Dimension at which noise eigenvalues fall to 1/e. If noise_dim is
        np.inf then a flat spectrum is used.
    return_samples : bool
        Whether to return cross_cov_mats or data samples.
    num_lorenz_samples : int
        Number of data samples to use.
    num_subspace_samples : int
        Number of random subspaces used to calculate the median relative angle.
    seed : int
        Seed for Numpy random state.
    """

    rng = np.random.RandomState(seed)
    # Generate Lorenz dynamics
    if X_dynamics is None:
        X_dynamics = gen_lorenz_data(num_lorenz_samples)
    dynamics_var = np.max(scipy.linalg.eigvalsh(np.cov(X_dynamics.T)))
    noise_var = dynamics_var / snr
    # Generate dynamics embedding matrix (will remain fixed)
    if V_dynamics is None:
        if N == 3:
            V_dynamics = np.eye(3)
        else:
            V_dynamics = random_basis(N, 3, rng)
    if noise_dim == np.inf:
        noise_cov = np.eye(N) * noise_var
    else:
        # Generate a subspace with median principal angles w.r.t. dynamics subspace
        if V_noise is None:
            V_noise = median_subspace(N, noise_dim, rng, num_samples=num_subspace_samples,
                                      V_0=V_dynamics)
        # Extend V_noise to a basis for R^N
        if V_noise.shape[1] < N:
            V_noise_comp = scipy.linalg.orth(np.eye(N) - np.dot(V_noise, V_noise.T))
            V_noise = np.concatenate((V_noise, V_noise_comp), axis=1)
        # Add noise covariance
        noise_cov = gen_noise_cov(N, noise_dim, noise_var, rng, V_noise=V_noise)
    # Generate actual samples of high-D data
    cross_cov_mats = calc_cross_cov_mats_from_data(X_dynamics, T)
    cross_cov_mats = np.array([V_dynamics.dot(C).dot(V_dynamics.T) for C in cross_cov_mats])
    cross_cov_mats[0] += noise_cov
    if return_samples:
        X_samples = (np.dot(X_dynamics, V_dynamics.T) +
                     rng.multivariate_normal(mean=np.zeros(N),
                                             cov=noise_cov, size=len(X_dynamics)))
        return cross_cov_mats, X_samples
    else:
        return cross_cov_mats


def generate_syn():
    # Save params
    with h5py.File(RESULTS_FILENAME, "w") as f:
        f.attrs["T"] = T
        f.attrs["N"] = N
        f.attrs["noise_dim"] = noise_dim
        f.attrs["snr_vals"] = snr_vals

        # Generate Lorenz dynamics
        num_samples = 10000
        X_dynamics = gen_lorenz_data(num_samples)
        dynamics_var = np.max(scipy.linalg.eigvalsh(np.cov(X_dynamics.T)))

        # Save dynamics
        f.create_dataset("X_dynamics", data=X_dynamics)
        f.attrs["dynamics_var"] = dynamics_var

        # Generate dynamics embedding matrix (will remain fixed)
        np.random.seed(42)
        V_dynamics = random_basis(N, 3, np.random)
        X = np.dot(X_dynamics, V_dynamics.T)


        # Generate a subspace with median principal angles w.r.t. dynamics subspace
        V_noise = median_subspace(N, noise_dim, num_samples=5000, V_0=V_dynamics, rng=np.random)
        # ... and extend V_noise to a basis for R^N
        V_noise_comp = scipy.linalg.orth(np.eye(N) - np.dot(V_noise, V_noise.T))
        V_noise = np.concatenate((V_noise, V_noise_comp), axis=1)

        # Save embeded dynamics and embedding matrices
        f.create_dataset("X", data=X)
        f.attrs["V_dynamics"] = V_dynamics
        f.attrs["V_noise"] = V_noise

        # Run DCA
        opt = DCA(T=T, d=3)
        opt.fit(X)
        V_dca = opt.coef_

        # Run PCA
        V_pca = scipy.linalg.eigh(np.cov(X.T))[1][:, ::-1][:, :3]

        # Project data onto DCA and PCA bases
        X_dca = np.dot(X, V_dca)
        X_pca = np.dot(X, V_pca)

        # Linearly trasnform projected data to be close to original Lorenz attractor
        beta_pca = np.linalg.lstsq(X_pca, X_dynamics, rcond=None)[0]
        beta_dca = np.linalg.lstsq(X_dca, X_dynamics, rcond=None)[0]
        X_pca_trans = np.dot(X_pca, beta_pca)
        X_dca_trans = np.dot(X_dca, beta_dca)

        # Save transformed projections
        X_pca_trans_dset_true = X_pca_trans
        X_dca_trans_dset_true = X_dca_trans

        f.create_dataset("X_pca_trans_true", data=X_pca_trans_dset_true)
        f.create_dataset("X_dca_trans_true", data=X_dca_trans_dset_true)

        # To-save: noisy data, reconstructed PCA, reconstructed DCA
        X_noisy_dset = f.create_dataset("X_noisy", (len(snr_vals), num_samples, N))
        X_pca_trans_dset = f.create_dataset("X_pca_trans", (len(snr_vals), num_samples, 3))
        X_dca_trans_dset = f.create_dataset("X_dca_trans", (len(snr_vals), num_samples, 3))

        # Loop over SNR vals
        for snr_idx in range(len(snr_vals)):
            snr = snr_vals[snr_idx]
            print("snr =", snr)

            _, X_noisy = embedded_lorenz_cross_cov_mats(N, T, snr, noise_dim, return_samples=True,
                                                        V_dynamics=V_dynamics, V_noise=V_noise,
                                                        X_dynamics=X_dynamics)
            X_noisy = X_noisy - X_noisy.mean(axis=0)
            # Save noisy data
            X_noisy_dset[snr_idx] = X_noisy

            # Run DCA
            opt = DCA(T=T, d=3)
            opt.fit(X_noisy)
            V_dca = opt.coef_

            # Run PCA
            V_pca = scipy.linalg.eigh(np.cov(X_noisy.T))[1][:, ::-1][:, :3]

            # Project data onto DCA and PCA bases
            X_dca = np.dot(X_noisy, V_dca)
            X_pca = np.dot(X_noisy, V_pca)

            # Linearly trasnform projected data to be close to original Lorenz attractor
            beta_pca = np.linalg.lstsq(X_pca, X_dynamics, rcond=None)[0]
            beta_dca = np.linalg.lstsq(X_dca, X_dynamics, rcond=None)[0]
            X_pca_trans = np.dot(X_pca, beta_pca)
            X_dca_trans = np.dot(X_dca, beta_dca)

            # Save transformed projections
            X_pca_trans_dset[snr_idx] = X_pca_trans
            X_dca_trans_dset[snr_idx] = X_dca_trans


if __name__ == "__main__":
    np.random.seed(22)

    #Set parameters
    T = 4
    N = 30
    noise_dim = 7
    snr_vals = np.logspace(-2, 2, 5)

    # generate data
    # generate_syn()

    # load data and plot
    with h5py.File(RESULTS_FILENAME, "r") as f:
        snr_vals = f.attrs["snr_vals"][:]
        X = f["X"][:]
        X_noisy_dset = f["X_noisy"][:]
        X_pca_trans_dset = f["X_pca_trans"][:]
        X_dca_trans_dset = f["X_dca_trans"][:]
        X_dynamics = f["X_dynamics"][:]

        r2_vals = np.zeros((len(snr_vals), 2))
        for snr_idx in range(len(snr_vals)):
            X_pca_trans = X_pca_trans_dset[snr_idx]
            X_dca_trans = X_dca_trans_dset[snr_idx]
            r2_pca = 1 - np.sum((X_pca_trans - X_dynamics) ** 2) / np.sum(
                (X_dynamics - np.mean(X_dynamics, axis=0)) ** 2)
            r2_dca = 1 - np.sum((X_dca_trans - X_dynamics) ** 2) / np.sum(
                (X_dynamics - np.mean(X_dynamics, axis=0)) ** 2)
            r2_vals[snr_idx] = [r2_pca, r2_dca]

    # ax: Lorenz 3D Plot
    T_to_show_3d = 500
    linewidth_3d = 0.5
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plot_lorenz_3d(ax, X_dynamics[:T_to_show_3d], linewidth_3d)
    plt.show()
    ax = plt.axes(projection='3d')
    plot_lorenz_3d(ax, X_pca_trans[:T_to_show_3d], linewidth_3d)
    plt.show()
    ax = plt.axes(projection='3d')
    plot_lorenz_3d(ax, X_dca_trans[:T_to_show_3d], linewidth_3d)
    plt.show()

    # generate latent features
    # z = X_dynamics[:T_to_show_3d]
    T = X_dynamics.shape[0]
    z = X_dynamics
    f = np.random.randn(3)
    latent_vars = dict()
    latent_vars['z'] = z
    latent_vars['f'] = f
    with open("../data/lorenz/latent_vars.pkl", "wb") as res:
        pickle.dump(latent_vars, res)

    # generate observations
    f_mat = np.tile(f.reshape(1, -1), (T, 1))
    latent_vs = np.concatenate([z, f_mat], axis=1)
    np.random.seed(22)
    V_dynamics = random_basis(N, 6, np.random)
    X = np.dot(latent_vs, V_dynamics.T)
    with open("../data/lorenz/sim_obs.pkl", "wb") as res:
        pickle.dump(X, res)

    # generate nonlinear observations
    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    f_mat = np.tile(f.reshape(1, -1), (T, 1))
    latent_vs = np.concatenate([z, f_mat], axis=1)
    np.random.seed(22)
    V0_dynamics = random_basis(N, 6, np.random)
    V1_dynamics = random_basis(N, N, np.random)

    X0 = np.tanh(np.dot(latent_vs, V0_dynamics.T))
    X1 = np.dot(X0, V1_dynamics.T)

    with open("../data/lorenz/sim_obs_NN2.pkl", "wb") as res:
        pickle.dump(X1, res)

    breakpoint()