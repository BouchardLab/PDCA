import h5py
import numpy as np
err = 1e-8
import torch


def load_data(RESULTS_FILENAME="data/lorenz/lorenz_results.hdf5"):
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
    return snr_vals, X, X_noisy_dset, X_pca_trans_dset, X_dca_trans_dset, X_dynamics


def psd_inv(K, device='cpu'):
    while True:
        try:
            u = torch.cholesky(K)
            break
        except:
            K += err * torch.eye(K.shape[0]).to(device)
    K_inv = torch.cholesky_inverse(u)
    return K_inv