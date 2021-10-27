import numpy as np
import pickle
import h5py
from utils.plotting.fig1 import lorenz_fig_axes, plot_3d, plot_lorenz_3d
import matplotlib.pyplot as plt
import torch
from torch import nn
from module import generate_rbfcov
from utils import cov_utils
import math
from tqdm import tqdm
import pickle
from utils.utils import psd_inv
err = 1e-3


class ObjectiveWrapper(object):
    """Helper object to cache gradient computation for minimization.

    Parameters
    ----------
    f_params : callable
        Function to calculate the loss as a function of the parameters.
    """
    def __init__(self, f_params):
        self.common_computations = None
        self.params = None
        self.f_params = f_params
        self.n_f = 0
        self.n_g = 0
        self.n_c = 0

    def core_computations(self, *args, **kwargs):
        """Calculate the part of the computation that is common to computing
        the loss and the gradient.

        Parameters
        ----------
        args
            Any other arguments that self.f_params needs.
        """
        params = args[0]
        if not np.array_equal(params, self.params):
            self.n_c += 1
            self.common_computations = self.f_params(*args, **kwargs)
            self.params = params.copy()
        return self.common_computations

    def func(self, *args):
        """Calculate and return the loss.

        Parameters
        ----------
        args
            Any other arguments that self.f_params needs.
        """
        self.n_f += 1
        loss, _ = self.core_computations(*args)
        return loss.detach().cpu().numpy().astype(float)

    def grad(self, *args):
        """Calculate and return the gradient of the loss.

        Parameters
        ----------
        args
            Any other arguments that self.f_params needs.
        """
        self.n_g += 1
        loss, params_torch = self.core_computations(*args)
        loss.backward(retain_graph=True)
        grad = params_torch.grad
        return grad.detach().cpu().numpy().astype(float)


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


class BaseComponentsAnalysis(object):
    def __init__(self, T=None, init="random_ortho", n_init=1, device="cpu", dtype=torch.float64):
        self.T = T
        self.init = init
        self.n_init = n_init
        self.device = device
        self.dtype = dtype

    def fit(self, X, beta=1, d=None, T=None, n_init=None, *args, **kwargs):
        if T is None:
            T = self.T
        if n_init is None:
            n_init = self.n_init
        self.load_data(X)
        self.fit_projection(d=d, T=T, beta=beta, n_init=n_init, *args, **kwargs)
        return self

    def fit_projection(self, d=None, T=None, beta=None, n_init=None, do_map=False, epochs=1000, lr=0.001):
        if n_init is None:
            n_init = self.n_init
        scores = []
        Ws = []
        est_Zs = []
        for i in range(n_init):
            if do_map:
                W, est_Z, score = self._fit_projection_map(d=d, T=T, beta=beta, epochs=epochs, lr=lr)
            else:
                W, score = self._fit_projection(d=d, T=T, beta=beta, epochs=epochs, lr=lr)
                est_Z = self.transfer(X=X)
            scores.append(score)
            Ws.append(W)
            est_Zs.append(est_Z)
        idx = np.argmax(scores)
        self.W = Ws[idx]
        self.Z = est_Zs[idx]
        # import pdb; pdb.set_trace()


class ProbabilitsticDCA(BaseComponentsAnalysis):
    """Probabilitstic Dynamcis Components Analysis"""
    def __init__(self, T=None, init="random_ortho", n_init=1, device="cpu", dtype=torch.float64):
        super(ProbabilitsticDCA, self).__init__(T=T, init=init, n_init=n_init, device=device, dtype=dtype)

    def load_data(self, X):
        """
        Parameters
        ----------
        X : ndarray or list of ndarrays
            Data to estimate the cross covariance matrix.
        """
        self.X = X

    def _fit_projection(self, beta=1, d=None, T=None, epochs=1000, lr=0.001):
        if d is None:
            raise ValueError
        if T is None:
            raise ValueError
        if (2*T) > self.X.shape[0]:
            raise ValueError('T must less than or equal to the number of observations')

        X = self.X
        _, m = X.shape
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, device=self.device, dtype=self.dtype)

        self.model = pdca_model(d=d, m=m, device=self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        loss_list = list()
        for epoch in tqdm(range(epochs)):
            optimizer.zero_grad()
            A, K, _ = self.model(X)
            loss = self.model.get_loss(A, K, X, beta=beta, T_window=T)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        fig = plt.figure()
        plt.plot(np.asarray(loss_list))
        plt.show()
        plt.close(fig)

        return self.model.W.detach().cpu().numpy(), -loss.detach().cpu().numpy()

    def _fit_projection_map(self, beta=1, d=None, T=None, epochs=1000, lr=0.001):
        if d is None:
            raise ValueError
        if T is None:
            raise ValueError
        if (2*T) > self.X.shape[0]:
            raise ValueError('T must less than or equal to the number of observations')

        X = self.X
        _, m = X.shape
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, device=self.device, dtype=self.dtype)

        self.model = pdca_model(d=d, m=m, X=X, device=self.device)
        parameters = self.model.parameters()
        # parameters = [{'params': self.model.B_half}, {'params': self.model.W}, {'params': self.model.sigma2_log}, {'params': self.model.Z}]
        # parameters = [{'params': self.model.Z}]
        optimizer = torch.optim.Adam(parameters, lr=lr)

        loss_list = list()
        for epoch in tqdm(range(epochs)):
            optimizer.zero_grad()
            _, K, B = self.model(X)
            loss = self.model.get_maploss(B, K, X, beta=beta, T_window=T)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        fig = plt.figure()
        plt.plot(np.asarray(loss_list))
        plt.show()
        plt.close(fig)

        return self.model.W.detach().cpu().numpy(), self.model.Z.detach().cpu().numpy(), -loss.detach().cpu().numpy()

    def transfer(self, X):
        T, _ = X.shape
        W = self.model.W.detach().cpu().numpy()
        est_Z = X.dot(W).dot(np.linalg.inv(W.T.dot(W)))
        return est_Z

    def marginal_llik(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, device=self.device, dtype=self.dtype)
        mllk = self.model.marginal_llik(X)
        return mllk.to('cpu').detach().numpy()


class pdca_model(nn.Module):
    def __init__(self, d=None, m=None, X=None, device='cpu', dtype=torch.float64):
        """
        d is the latent dimension size
        m is the output dimension size
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.d = d
        self.m = m
        if X is not None:
            self.T, _ = X.shape
            self.Z = nn.Parameter(torch.randn((self.T, self.d), dtype=self.dtype, device=self.device))
        # initialzation
        self.prior_covmat = generate_rbfcov(raw_lengthscale=1.).to(self.device)
        self.B_half = nn.Parameter(torch.eye(self.d, dtype=self.dtype, device=self.device), requires_grad=False)
        self.W = nn.Parameter(torch.randn((self.m, self.d), dtype=self.dtype, device=self.device))
        self.sigma2_log = nn.Parameter(torch.tensor(0, dtype=self.dtype, device=self.device))

    def forward(self, X):
        T, m = X.shape
        sigma2 = torch.exp(self.sigma2_log)
        timestamps = torch.arange(T)[:, None].to(self.device)
        K = self.prior_covmat(timestamps, timestamps)
        # K += err * torch.eye(T).to(self.device)
        B = self.B_half.mm(self.B_half.T)
        A = self.W.mm(B).mm(self.W.T) + sigma2 * torch.diag(torch.ones(m, device=self.device))
        # import pdb;
        # pdb.set_trace()
        return A, K, B

    def get_loss(self, A, K, X, beta, T_window):
        X_vec = X.T.ravel()
        n = X_vec.shape[0]
        t2T = torch.arange(2 * T_window)[:, None].to(self.device)
        cov = self.prior_covmat(t2T, t2T)
        pi_loss = cov_utils.calc_pi_from_cov(cov)
        T, m = X.shape
        inv_cov = torch.kron(torch.linalg.inv(A), torch.linalg.inv(K))
        logdet_cov = torch.logdet(A) * T + torch.logdet(K) * m
        llk = -0.5*X_vec.matmul(inv_cov.matmul(X_vec)) - 0.5*(torch.log(2*torch.tensor(math.pi).to(self.device))*n + logdet_cov)
        loss = - llk - beta * pi_loss
        # import pdb;
        # pdb.set_trace()
        return loss

    def get_K_T(self, T):
        timestamps = torch.arange(T)[:, None].to(self.device)
        K = self.prior_covmat(timestamps, timestamps)
        return K

    def get_maploss(self, B, K, X, beta, T_window):
        Z_vec = self.Z.T.ravel()
        n_z = Z_vec.shape[0]

        # # PI for prior
        # t2T = torch.arange(2 * T_window)[:, None].to(self.device)
        # cov = self.prior_covmat(t2T, t2T)
        # pi = cov_utils.calc_pi_from_cov(cov)

        # PI for posterior
        z_pi = torch.unsqueeze(self.Z, dim=0)
        if T_window > 0:
            cov = cov_utils.calc_cov_from_data(z_pi, 2 * T_window, toeplitzify=False)
            pi = cov_utils.calc_pi_from_cov(cov)
        else:
            pi = 0

        sigma2 = torch.exp(self.sigma2_log)
        K_inv = psd_inv(K, self.device)
        # while True:
        #     try:
        #         u = torch.cholesky(K)
        #         break
        #     except:
        #         K += err*torch.eye(K.shape[0]).to(self.device)
        # K_inv = torch.cholesky_inverse(u)
        inv_cov_K = torch.kron(torch.linalg.inv(B), K_inv)
        logdet_cov_K = torch.logdet(B) * self.T + torch.logdet(K) * self.d
        jllk = -0.5*Z_vec.matmul(inv_cov_K.matmul(Z_vec)) - 0.5*(torch.log(2*torch.tensor(math.pi).to(self.device))*n_z + logdet_cov_K)
        diff = self.Z.mm(self.W.T) - X
        n = self.T * self.m
        # import pdb; pdb.set_trace()
        jllk += -0.5*torch.sum(diff**2) / sigma2 - 0.5*(torch.log(2*torch.tensor(math.pi).to(self.device))*n + self.sigma2_log*n)
        loss = - jllk - beta*pi

        print(loss.item(), jllk.item(), pi.item(), self.prior_covmat.raw_lengthscale.item())

        return loss

    def marginal_llik(self, X):
        T, m = X.shape
        sigma2 = torch.exp(self.sigma2_log)
        timestamps = torch.arange(T)[:, None].to(self.device)
        K = self.prior_covmat(timestamps, timestamps)
        B = self.B_half.mm(self.B_half.T)
        A = self.W.mm(B).mm(self.W.T) + sigma2 * torch.diag(torch.ones(m, device=self.device))
        X_vec = X.T.ravel()
        n = X_vec.shape[0]
        K_inv = psd_inv(K, self.device)
        # while True:
        #     try:
        #         u = torch.cholesky(K)
        #         break
        #     except:
        #         K += err*torch.eye(K.shape[0]).to(self.device)
        # K_inv = torch.cholesky_inverse(u)
        inv_cov = torch.kron(torch.linalg.inv(A), K_inv)
        logdet_cov = torch.logdet(A) * T + torch.logdet(K) * m
        mllk = -0.5 * X_vec.matmul(inv_cov.matmul(X_vec)) - 0.5 * (
                    torch.log(2 * torch.tensor(math.pi).to(self.device)) * n + logdet_cov)
        return mllk


if __name__ == "__main__":
    do_training = True
    snr_vals, X, X_noisy_dset, X_pca_trans_dset, X_dca_trans_dset, X_dynamics = load_data()
    beta = 100
    # beta = 1
    num_cases = 1
    do_saving = False

    if do_training:
        X_pdca_trans_dset = list()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(device)
        for i in range(num_cases):
            PDCA = ProbabilitsticDCA(T=5, device=device)
            # PDCA.fit(X_noisy_dset[i][:500], beta=beta, d=3)
            PDCA.fit(X_noisy_dset[i][:500], beta=beta, d=3, do_map=False, epochs=1000, lr=0.01)
            X_pdca = PDCA.Z.copy()
            fig = plt.figure()
            plt.plot(X_pdca[:500])
            plt.show()
            plt.close(fig)
            print(PDCA.model.prior_covmat.raw_lengthscale)
            print(PDCA.marginal_llik(X_noisy_dset[i][:500]))
            print(np.linalg.lstsq(X_pdca[:500], X_dynamics[:500], rcond=None))
            import pdb; pdb.set_trace()
            # Linearly trasnform projected data to be close to original Lorenz attractor
            beta_pdca = np.linalg.lstsq(X_pdca[:500], X_dynamics[:500], rcond=None)[0]
            X_pdca_trans = np.dot(X_pdca, beta_pdca)
            X_pdca_trans_dset.append(X_pdca_trans)
            # import pdb; pdb.set_trace()
        if do_saving:
            with open("res/lorenz/pdca0.pkl", "wb") as f:
                pickle.dump(X_pdca_trans_dset, f)
    else:
        with open("res/lorenz/pdca0.pkl", "rb") as f:
            X_pdca_trans_dset = pickle.load(f)

    # ax: Lorenz 3D Plot
    T_to_show_3d = 500
    linewidth_3d = 0.5
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plot_lorenz_3d(ax, X_dynamics[:T_to_show_3d], linewidth_3d)
    plt.show()
    for i in range(num_cases):
        ax = plt.axes(projection='3d')
        plot_lorenz_3d(ax, X_pca_trans_dset[i][:T_to_show_3d], linewidth_3d)
        plt.title("PCA, snr:{}".format(snr_vals[i]))
        plt.savefig("res/lorenz/pca_snr{}.png".format(snr_vals[i]))
        plt.show()
        ax = plt.axes(projection='3d')
        plot_lorenz_3d(ax, X_dca_trans_dset[i][:T_to_show_3d], linewidth_3d)
        plt.title("DCA, snr:{}".format(snr_vals[i]))
        plt.savefig("res/lorenz/dca_snr{}.png".format(snr_vals[i]))
        plt.show()
        ax = plt.axes(projection='3d')
        plot_lorenz_3d(ax, X_pdca_trans_dset[i][:T_to_show_3d], linewidth_3d)
        plt.title("PDCA, snr:{}".format(snr_vals[i]))
        plt.savefig("res/lorenz/pdca_snr{}_beta{}.png".format(snr_vals[i], beta))
        plt.show()
    import pdb; pdb.set_trace()
