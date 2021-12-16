import torch
import time
import numpy as np

def matrix_vector_kron(mat_list, b):
    """

    Parameters
    ----------
    mat_list: a list of D matrices
    b: length-N vector

    Returns
    -------
    alpha: (kron_1^D(A_d))b
    """
    D = len(mat_list)
    x_vec = b
    N = b.shape[0]

    for index in range(D):
        d = D - 1 - index
        A_d = mat_list[d]
        G_d = A_d.shape[0]
        X = torch.reshape(x_vec, shape=(int(N/G_d), G_d)).T
        Z = torch.matmul(A_d, X)
        x_vec = Z.reshape(-1)
    alpha = x_vec
    return alpha

def eigen_decomposition(mat_list):
    """
    eigen-decomposition for a list of covariance matrices
    Parameters
    ----------
    mat_list

    Returns
    -------
    eigen_mat_list
    eigenvalues
    """
    eigen_mat_list = list()
    eigen_val_list = list()
    for A in mat_list:
        eigvals, eigvecs = torch.linalg.eigh(A)
        eigen_mat_list.append(eigvecs)
        eigen_val_list.append(eigvals)
    curr_eigen_vals = None
    for eigvals in eigen_val_list:
        if curr_eigen_vals is None:
            curr_eigen_vals = eigvals
        else:
            curr_eigen_vals = torch.cat([v*eigvals for v in curr_eigen_vals])
    eigen_vals = curr_eigen_vals
    return eigen_mat_list, eigen_vals

def log_pdf_mn(A_list, sigma2, x):
    """
    log of probability density function of multivariate normal
    Parameters
    ----------
    A_list
    sigma2
    x

    Returns
    -------
    log_pdf
    """
    N = x.shape[0]
    Q_list, lambdas = eigen_decomposition(A_list)
    Q_t_list= [Q.T for Q in Q_list]
    transform_x = matrix_vector_kron(Q_t_list, x)
    lambdas_noise = lambdas + sigma2*torch.ones(N)
    d_inv = 1./(lambdas_noise)
    term1 = -0.5*torch.dot(transform_x, d_inv*transform_x)
    term2 = -0.5*torch.sum(torch.log(lambdas_noise))
    log_pdf = term1 + term2 -0.5*N*torch.log(2*torch.tensor(np.pi))
    return log_pdf

def log_pdf_mn_standard(A_list, sigma2, x):
    N = x.shape[0]
    A = torch.kron(A_list[0], A_list[1])
    A_noise = A + torch.diag(torch.ones(N)*sigma2)
    from torch.distributions.multivariate_normal import MultivariateNormal
    log_pdf = MultivariateNormal(torch.zeros(N), A_noise).log_prob(x)
    return log_pdf

if __name__ == "__main__":
    torch.manual_seed(22)
    n0 = 30
    n1 = 500
    A0 = torch.randn(n0, n0)
    A1 = torch.randn(n1, n1)
    A_list = [A0.matmul(A0.T), A1.matmul(A1.T)]
    b = torch.randn(n0*n1)
    sigma2 = 1e-2


    # ts0 = time.time()
    # A = torch.kron(A_list[0], A_list[1])
    # ans0 = torch.matmul(A, b)
    # print(ans0)
    # print(time.time() - ts0)
    # ts1 = time.time()
    # ans1 = matrix_vector_kron(A_list, b)
    # print(ans1)
    # print(time.time() - ts1)
    # assert torch.any(ans0 == ans1)

    # eigen_mat_list, eigenvs = eigen_decomposition(A_list)

    ts0 = time.time()
    log_pdf0 = log_pdf_mn(A_list, sigma2, b)
    print(log_pdf0)
    print(time.time() - ts0)
    ts1 = time.time()
    log_pdf1 = log_pdf_mn_standard(A_list, sigma2, b)
    print(log_pdf1)
    print(time.time() - ts1)

    import pdb; pdb.set_trace()