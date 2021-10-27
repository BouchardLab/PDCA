# covariance functions/kernels to be used by Gaussian process functions.
# There are three different kinds of covariance functions: simple, composite and FITC:
# 
# simple covariance functions:
#
# covSEiso      - isotropic squared exponential covariance function
# covSEard      - squared exponential covariance function with ard (Automatic Relevance Determination)
# covSEisoU     - isotropic squared exponential covariance function with unit magnitude
# covLIN        - linear covariance function
# covLINard     - linear covariance function with ard
# covPoly       - polynomial covariance function
# covPPiso      - piecewise polynomial covariance function with compact support
# covConst      - constant covariance function
# covMatern     - Matern covariance function 
# covPeriodic   - periodic covariance function
# covRQiso      - rational quadratic covariance function 
# covRQard      - rational quadratic covariance function with ard
# covNoise      - independent covariance function (ie white noise)
# 
# composite covariance functions (see explanation at the bottom):
# 
# covSum        - sum of (parameterized) covariance functions
# covProd       - product of (parametrized) covariance functions
# covScale      - composition of a covariance function as a scaled version of another covariance function
# covMask       - composition of a covariance function from another covariance function using a subset of input dimensions
#
# FITC:
#
# covFITC       - covariance function to be used together with the FITC (pseudo input) approximation
#
# Naming convention: all covariance functions start with "cov". A trailing
# "iso" means isotropic, "ard" means Automatic Relevance Determination.
# 
# The covariance functions are written according to a special convention where
# the exact behaviour depends on the number of input and output arguments
# passed to the function. If you want to add new covariance functions, you 
# should follow this convention if you want them to work with the function gp.
# There are four different ways of calling
# the covariance functions:
# 
# 1) With no input arguments:
#
#   num_prarams = src.Tools.general.feval(covfunc)   
# 
#   The covariance function returns a list with the number of hyperparameters it
#   expects. For ard covariacne function the list includes a string where "D" 
#   is the dimension of the input space.
#   For example, calling "covSEiso" returns [2]; "covSEard" returns ['D + 1'].
# 
# 2) With two input arguments:
# 
#   K =  src.Tools.general.feval(covfunc, hyp, x) 
# 
#   The function computes and returns the covariance matrix where logtheta are
#   the log of the hyperparameters and x is an n by D matrix of input points, where
#   D is the dimension of the input space. The returned covariance matrix is of
#   size n by n.
# 
# 3) With three input arguments:
# 
#   Ks  = src.Tools.general.feval(covfunc, hyp, x, z) 
# 
#   The function computes test set covariances; Ks is a (n by nn) matrix of cross
#   covariances between training cases x and test cases z,
#   where z is a nn by D matrix.
#
# 4) With three input arguments where one is 'diag':
#
#   kss = src.Tools.general.feval(covfunc, hyp, z, 'diag') 
# 
#   The function computes self-variances; kss is a (nn by 1) vector.
#
# 5) With four input arguments:
# 
#   Deriv = src.Tools.general.feval(covfunc, hyp, x, z, der) or
#   Deriv = src.Tools.general.feval(covfunc, hyp, x, None, der)
#
#   The function computes and returns the n by n resp. n by nn matrix of partial 
#   derivatives of either the training set covariance matrix or the train-test covariance
#   with respect to logtheta(der), i.e. with respect to the log of hyperparameter number der,
#   where der in [0,1, ..., num_prarams-1].
# 
# About the specification of simple and composite covariance functions to be
# used by the Gaussian process function gp:
# 
# covfunc = [['kernels.covSEard']]
# 
# Composite covariance functions can be specified as list. For example:
# 
# covfunc = [['kernels.covSum'], [['kernels.covSEard'],['kernels.covNoise']]]
# covfunc = [['kernels.covProd'],[['kernels.covPeriodic'],['kernels.covSEiso']]]
# 
# 
# @author: Marion Neumann (last update 30/09/13)
# Substantial updates by Daniel Marthaler Fall 2012.
#                        Shan Huang (Sep. 2013)
#
# This is a python implementation of gpml functionality (Copyright (c) by
# Carl Edward Rasmussen and Hannes Nickisch, 2011-02-18).
#
# Copyright (c) by Marion Neumann and Daniel Marthaler, 20/05/2013

import numpy as np
import scipy.spatial.distance as spdist
# import general


def covSpectralSingle(hyp=None, x=None, z=None, der=None):
    '''A single instance of the inverse Fourier transform of a scale location gaussian (from GPatt)
    covariance function. hyp = [ log_w, log_sigma, log_mu ]
    :param log_w: the weight of the instance
    :param log_sigma: inverse signal deviation.
    :param log_mu: the period of the instance
    '''
    if hyp is None:  # report number of parameters
        return [4]

    w2 = np.exp(2. * hyp[0])  # inhomogeneous offset
    sig = np.exp(hyp[1])  # signal variance
    mu = np.exp(hyp[2])  # order of polynomial
    index = hyp[3]  # Should be an integer, it is the index to which instance of the total product

    n, D = x.shape
    xi = np.reshape(x[:, index], (x.shape[0], 1))

    if z == 'diag':
        A = np.zeros((n, 1))
    elif z is None:
        A = spdist.cdist(xi * sig, xi * sig, 'sqeuclidean')
    else:  # compute covariance between data sets x and z
        zi = np.reshape(z[:, index], (z.shape[0], 1))
        A = spdist.cdist(xi * sig, zi * sig, 'sqeuclidean')  # self covariances

    f = w2 * np.exp(-2. * (np.pi ** 2) * A) * np.cos(2. * np.pi * np.sqrt(A) * mu)

    if der is None:  # compute covariance matix for dataset x
        A = f
    else:
        if der == 0:  # compute derivative matrix wrt w
            A = 2. * f
        elif der == 1:  # compute derivative matrix wrt sf2
            A = -4 * (np.pi ** 2) * A * f
        elif der == 2:  # no derivative wrt mu
            A = -2 * np.pi * np.sqrt(A) * mu * w2 * np.exp(-2. * (np.pi ** 2) * A) * np.sin(
                2. * np.pi * np.sqrt(A) * mu)
        elif der == 3:
            A = np.zeros_like(f)
        else:
            raise Exception("Wrong derivative entry in covSpectralSignal")
    return A


def covSEiso(hyp=None, x=None, z=None, der=None):
    ''' Squared Exponential covariance function with isotropic distance measure.
     The covariance function is parameterized as:

      k(x^p,x^q) = sf2 * exp(-(x^p - x^q)' * inv(P) * (x^p - x^q)/2)

      where the P matrix is ell^2 times the unit matrix and
      sf2 is the signal variance  

     The hyperparameters of the function are:

        hyp = [ log(ell)
                log(sqrt(sf2)) ]
     a column vector. 
     Each row of x resp. z is a data point.
    '''

    if hyp is None:  # report number of parameters
        return [2]

    ell = np.exp(hyp[0])  # characteristic length scale
    sf2 = np.exp(2. * hyp[1])  # signal variance
    n, D = x.shape

    if z == 'diag':
        A = np.zeros((n, 1))
    elif z is None:
        A = spdist.cdist(x / ell, x / ell, 'sqeuclidean')
    else:  # compute covariance between data sets x and z
        A = spdist.cdist(x / ell, z / ell, 'sqeuclidean')  # self covariances

    if der is None:  # compute covariance matix for dataset x
        A = sf2 * np.exp(-0.5 * A)
    else:
        if der == 0:  # compute derivative matrix wrt 1st parameter
            A = sf2 * np.exp(-0.5 * A) * A

        elif der == 1:  # compute derivative matrix wrt 2nd parameter
            A = 2. * sf2 * np.exp(-0.5 * A)
        else:
            raise Exception("Calling for a derivative in covSEiso that does not exist")

    return A


def covSEard(hyp=None, x=None, z=None, der=None):
    ''' Squared Exponential covariance function with Automatic Relevance Detemination
     (ARD) distance measure. The covariance function is parameterized as:

     k(x^p,x^q) = sf2 * exp(-(x^p - x^q)' * inv(P) * (x^p - x^q)/2)

     where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
     D is the dimension of the input space and sf2 is the signal variance.

     The hyperparameters are:

     hyp = [ log(ell_1)
             log(ell_2)
             ...
             log(ell_D)
             log(sqrt(sf2)) ]
    '''

    if hyp is None:  # report number of parameters
        return ['D + 1']  # USAGE: integer OR D_+_int (spaces are SIGNIFICANT)

    [n, D] = x.shape
    ell = 1 / np.exp(hyp[0:D])  # characteristic length scale

    sf2 = np.exp(2. * hyp[D])  # signal variance

    if z == 'diag':
        A = np.zeros((n, 1))
    elif z is None:
        tmp = np.dot(np.diag(ell), x.T).T
        A = spdist.cdist(tmp, tmp, 'sqeuclidean')
    else:  # compute covariance between data sets x and z
        A = spdist.cdist(np.dot(np.diag(ell), x.T).T, np.dot(np.diag(ell), z.T).T, 'sqeuclidean')  # cross covariances

    A = sf2 * np.exp(-0.5 * A)
    if not der is None:
        if der < D:  # compute derivative matrix wrt length scale parameters
            if z == 'diag':
                A = A * 0.
            elif z is None:
                tmp = np.atleast_2d(x[:, der]).T / ell[der]
                A = A * spdist.cdist(tmp, tmp, 'sqeuclidean')
            else:
                A = A * spdist.cdist(np.atleast_2d(x[:, der]).T / ell[der], np.atleast_2d(z[:, der]).T / ell[der],
                                     'sqeuclidean')
        elif der == D:  # compute derivative matrix wrt magnitude parameter
            A = 2. * A
        else:
            raise Exception("Wrong derivative index in covSEard")

    return A


def covSEisoU(hyp=None, x=None, z=None, der=None):
    ''' Squared Exponential covariance function with isotropic distance measure with
     unit magnitude. The covariance function is parameterized as:

     k(x^p,x^q) = exp( -(x^p - x^q)' * inv(P) * (x^p - x^q) / 2 )

     where the P matrix is ell^2 times the unit matrix. 

     The hyperparameters of the function are:

     hyp = [ log(ell) ]
    '''

    if hyp is None:  # report number of parameters
        return [1]

    ell = np.exp(hyp[0])  # characteristic length scale
    n, D = x.shape

    if z == 'diag':
        A = np.zeros((n, 1))
    elif z is None:
        A = spdist.cdist(x / ell, x / ell, 'sqeuclidean')
    else:  # compute covariance between data sets x and z
        A = spdist.cdist(x / ell, z / ell, 'sqeuclidean')  # self covariances

    if der is None:  # compute covariance matix for dataset x
        A = np.exp(-0.5 * A)
    else:
        if der == 0:  # compute derivative matrix wrt 1st parameter
            A = np.exp(-0.5 * A) * A
        else:
            raise Exception("Wrong derivative index in covSEisoU")

    return A


def covLIN(hyp=None, x=None, z=None, der=None):
    ''' Linear Covariance function.
     The covariance function is parameterized as:
     k(x^p,x^q) = x^p' * x^q

     There are no hyperparameters:

     hyp = []

     Note that there is no bias or scale term; use covConst and covScale to add these.
    '''

    if hyp is None:  # report number of parameters
        return [0]
    n, m = x.shape

    if z == 'diag':
        A = np.reshape(np.sum(x * x, 1), (n, 1))
    elif z is None:
        A = np.dot(x, x.T) + np.eye(n) * 1e-16  # required for numerical accuracy
    else:  # compute covariance between data sets x and z
        A = np.dot(x, z.T)  # cross covariances

    if der:
        raise Exception("No derivative available in covLIN")

    return A


def covLINard(hyp=None, x=None, z=None, der=None):
    ''' Linear covariance function with Automatic Relevance Detemination
     (ARD) distance measure. The covariance function is parameterized as:
     k(x^p,x^q) = x^p' * inv(P) * x^q

     where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
     D is the dimension of the input space and sf2 is the signal variance. The
     hyperparameters are:

     hyp = [ log(ell_1), log(ell_2), ... , log(ell_D) ]

     Note that there is no bias term; use covConst to add a bias.
    '''

    if hyp is None:  # report number of parameters
        return ['D + 0']  # USAGE: integer OR D_+_int (spaces are SIGNIFICANT)

    n, D = x.shape
    ell = np.exp(hyp)  # characteristic length scales
    x = np.dot(x, np.diag(1. / ell))

    if z == 'diag':
        A = np.reshape(np.sum(x * x, 1), (n, 1))
    elif z is None:
        A = np.dot(x, x.T)
    else:  # compute covariance between data sets x and z
        z = np.dot(z, np.diag(1. / ell))
        A = np.dot(x, z.T)  # cross covariances

    if not der is None and der < D:
        if z == 'diag':
            A = -2. * x[:, der] * x[:, der]
        elif z is None:
            A = -2. * np.dot(x[:, der], x[:, der].T)
        else:
            A = -2. * np.dot(x[:, der], z[:, der].T)  # cross covariances
    elif der:
        raise Exception("Wrong derivative index in covLINard")

    return A


def covPoly(hyp=None, x=None, z=None, der=None):
    ''' Polynomial covariance function 
     The covariance function is parameterized as:
     k(x^p,x^q) = sf2 * ( c +  (x^p)'*(x^q) ) ** d

     The hyperparameters of the function are:

     hyp = [ log(c)
             log(sqrt(sf2)) 
             d ]

     NOTE: d is not treated as a hyperparameter. 
    '''

    if hyp is None:  # report number of parameters
        return [3]

    c = np.exp(hyp[0])  # inhomogeneous offset
    sf2 = np.exp(2. * hyp[1])  # signal variance
    d = hyp[2]  # degree of polynomical

    if np.abs(d - np.round(d)) < 1e-8:  # remove numerical error from format of parameter
        d = int(round(d))
    assert (d >= 1.)  # only nonzero integers for d
    d = int(d)
    n, D = x.shape

    if z == 'diag':
        A = np.reshape(np.sum(x * x, 1), (n, 1))
    elif z is None:
        A = np.dot(x, x.T)
    else:  # compute covariance between data sets x and z
        A = np.dot(x, z.T)  # cross covariances

    if der is None:  # compute covariance matix for dataset x
        A = sf2 * (c + A) ** d
    else:
        if der == 0:  # compute derivative matrix wrt 1st parameter
            A = c * d * sf2 * (c + A) ** (d - 1)
        elif der == 1:  # compute derivative matrix wrt 2nd parameter
            A = 2. * sf2 * (c + A) ** d
        elif der == 2:  # NOTE: d is not treated as a hyperparameter -> we do not want to optimize w.r.t. d
            A = np.zeros_like(A)
        else:
            raise Exception("Wrong derivative entry in covPoly")

    return A


def covPPiso(hyp=None, x=None, z=None, der=None):
    ''' Piecewise polynomial covariance function with compact support
     The covariance function is:

     k(x^p,x^q) = s2f * (1-r)_+.^j * f(r,j)

     where r is the distance sqrt((x^p-x^q)' * inv(P) * (x^p-x^q)), P is ell^2 times
     the unit matrix and sf2 is the signal variance. 
     The hyperparameters are:

     hyp = [ log(ell)
             log(sqrt(sf2)) 
             log(v) ]
    '''

    def ppmax(A, B):
        return np.maximum(A, B * np.ones_like(A))

    def func(v, r, j):
        if v == 0:
            return 1
        elif v == 1:
            return (1. + (j + 1) * r)
        elif v == 2:
            return (1. + (j + 2) * r + (j * j + 4. * j + 3) / 3. * r * r)
        elif v == 3:
            return (1. + (j + 3) * r + (6. * j * j + 36. * j + 45.) / 15. * r * r + (
                        j * j * j + 9. * j * j + 23. * j + 15.) / 15. * r * r * r)
        else:
            raise Exception(["Wrong degree in covPPiso.  Should be 0,1,2 or 3, is " + str(v)])

    def dfunc(v, r, j):
        if v == 0:
            return 0
        elif v == 1:
            return (j + 1)
        elif v == 2:
            return ((j + 2) + 2. * (j * j + 4. * j + 3.) / 3. * r)
        elif v == 3:
            return ((j + 3) + 2. * (6. * j * j + 36. * j + 45.) / 15. * r + (
                        j * j * j + 9. * j * j + 23. * j + 15.) / 5. * r * r)
        else:
            raise Exception(["Wrong degree in covPPiso.  Should be 0,1,2 or 3, is " + str(v)])

    def pp(r, j, v, func):
        return func(v, r, j) * (ppmax(1 - r, 0) ** (j + v))

    def dpp(r, j, v, func, dfunc):
        return ppmax(1 - r, 0) ** (j + v - 1) * r * ((j + v) * func(v, r, j) - ppmax(1 - r, 0) * dfunc(v, r, j))

    if hyp is None:  # report number of parameters
        return [3]

    ell = np.exp(hyp[0])  # characteristic length scale
    sf2 = np.exp(2. * hyp[1])  # signal variance
    v = np.exp(hyp[2])  # degree (v = 0,1,2 or 3 only)

    if np.abs(v - np.round(v)) < 1e-8:  # remove numerical error from format of parameter
        v = int(round(v))

    assert (int(v) in range(4))  # Only allowed degrees: 0,1,2 or 3
    v = int(v)

    n, D = x.shape

    j = np.floor(0.5 * D) + v + 1

    if z == 'diag':
        A = np.zeros((n, 1))
    elif z is None:
        A = np.sqrt(spdist.cdist(x / ell, x / ell, 'sqeuclidean'))
    else:  # compute covariance between data sets x and z
        A = np.sqrt(spdist.cdist(x / ell, z / ell, 'sqeuclidean'))  # cross covariances

    if der is None:  # compute covariance matix for dataset x
        A = sf2 * pp(A, j, v, func)
    else:
        if der == 0:  # compute derivative matrix wrt 1st parameter
            A = sf2 * dpp(A, j, v, func, dfunc)

        elif der == 1:  # compute derivative matrix wrt 2nd parameter
            A = 2. * sf2 * pp(A, j, v, func)

        elif der == 2:  # wants to compute derivative wrt order
            A = np.zeros_like(A)
        else:
            raise Exception("Wrong derivative entry in covPPiso")

    return A


def covConst(hyp=None, x=None, z=None, der=None):
    ''' Covariance function for a constant function.
    The covariance function is parameterized as:
    k(x^p,x^q) = sf2 

    The scalar hyperparameter is:

    hyp = [ log(sqrt(sf2)) ]
    '''

    if hyp is None:  # report number of parameters
        return [1]
    sf2 = np.exp(2. * hyp[0])  # s2

    n, m = x.shape
    if z == 'diag':
        A = sf2 * np.ones((n, 1))
    elif z is None:
        A = sf2 * np.ones((n, n))
    else:
        A = sf2 * np.ones((n, z.shape[0]))

    if der == 0:  # compute derivative matrix wrt sf2
        A = 2. * A
    elif der:
        raise Exception("Wrong derivative entry in covConst")
    return A


def covMatern(hyp=None, x=None, z=None, der=None):
    ''' Matern covariance function with nu = d/2 and isotropic distance measure. For d=1 
     the function is also known as the exponential covariance function or the 
     Ornstein-Uhlenbeck covariance in 1d. The covariance function is:

        k(x^p,x^q) = s2f * f( sqrt(d) * r ) * exp(-sqrt(d) * r)

     with f(t)=1 for d=1, f(t)=1+t for d=3 and f(t)=1+t+(t * t)/3 for d=5. 
     Here, r is the distance sqrt( (x^p-x^q)' * inv(P) * (x^p-x^q)), 
     where P is ell times the unit matrix and sf2 is the signal variance.

     The hyperparameters of the function are:

     hyp = [ log(ell) 
             log(sqrt(sf2)) 
             d ]
    '''

    def func(d, t):
        if d == 1:
            return 1
        elif d == 3:
            return 1 + t
        elif d == 5:
            return 1 + t * (1 + t / 3.)
        else:
            raise Exception("Wrong value for d in covMatern")

    def dfunc(d, t):
        if d == 1:
            return 1
        elif d == 3:
            return t
        elif d == 5:
            return t * (1 + t / 3.)
        else:
            raise Exception("Wrong value for d in covMatern")

    def mfunc(d, t):
        return func(d, t) * np.exp(-1. * t)

    def dmfunc(d, t):
        return dfunc(d, t) * t * np.exp(-1. * t)

    if hyp is None:  # report number of parameters
        return [3]

    ell = np.exp(hyp[0])  # characteristic length scale
    sf2 = np.exp(2. * hyp[1])  # signal variance
    d = np.exp(hyp[2])  # 2 times nu

    if np.abs(d - np.round(d)) < 1e-8:  # remove numerical error from format of parameter
        d = int(round(d))

    try:
        assert (int(d) in [1, 3, 5])  # Check for valid values of d
    except AssertionError:
        d = 3

    d = int(d)

    if z == 'diag':
        A = np.zeros((x.shape[0], 1))
    elif z is None:
        x = np.sqrt(d) * x / ell
        A = np.sqrt(spdist.cdist(x, x, 'sqeuclidean'))
    else:
        x = np.sqrt(d) * x / ell
        z = np.sqrt(d) * z / ell
        A = np.sqrt(spdist.cdist(x, z, 'sqeuclidean'))

    if der is None:  # compute covariance matix for dataset x
        A = sf2 * mfunc(d, A)
    else:
        if der == 0:  # compute derivative matrix wrt 1st parameter
            A = sf2 * dmfunc(d, A)

        elif der == 1:  # compute derivative matrix wrt 2nd parameter
            A = 2 * sf2 * mfunc(d, A)
        elif der == 2:  # Wants to compute derivative wrt nu
            A = np.zeros_like(A)  # Do nothing
        else:
            raise Exception("Wrong derivative value in covMatern")

    return A


def covPeriodic(hyp=None, x=None, z=None, der=None):
    ''' Stationary covariance function for a smooth periodic function,
     with period p:

     k(x^p,x^q) = sf2 * exp( -2*sin^2( pi*||x^p - x^q)||/p )/ell**2 )

     The hyperparameters of the function are:
        hyp = [ log(ell)
                log(p)
                log(sqrt(sf2)) ]
    '''

    if hyp is None:  # report number of parameters
        return [3]

    ell = np.exp(hyp[0])  # characteristic length scale
    p = np.exp(hyp[1])  # period
    sf2 = np.exp(2. * hyp[2])  # signal variance

    n, D = x.shape

    if z == 'diag':
        A = np.zeros((n, 1))
    elif z is None:
        A = np.sqrt(spdist.cdist(x, x, 'sqeuclidean'))
    else:
        A = np.sqrt(spdist.cdist(x, z, 'sqeuclidean'))

    A = np.pi * A / p

    if der is None:  # compute covariance matix for dataset x
        A = np.sin(A) / ell
        A = A * A
        A = sf2 * np.exp(-2. * A)
    else:
        if der == 0:  # compute derivative matrix wrt 1st parameter
            A = np.sin(A) / ell
            A = A * A
            A = 4. * sf2 * np.exp(-2. * A) * A

        elif der == 1:  # compute derivative matrix wrt 2nd parameter
            R = np.sin(A) / ell
            A = 4 * sf2 / ell * np.exp(-2. * R * R) * R * np.cos(A) * A

        elif der == 2:  # compute derivative matrix wrt 3rd parameter
            A = np.sin(A) / ell
            A = A * A
            A = 2. * sf2 * np.exp(-2. * A)
        else:
            raise Exception("Wrong derivative index in covPeriodic")

    return A


def covRQiso(hyp=None, x=None, z=None, der=None):
    ''' Rational Quadratic covariance function with isotropic distance measure.
     The covariance function is parameterized as:

     k(x^p,x^q) = sf2 * [1 + (x^p - x^q)' * inv(P) * (x^p - x^q)/(2 * alpha)]^(-alpha)

     where the P matrix is ell^2 times the unit matrix,
     sf2 is the signal variance, and alpha is the shape parameter for the RQ
     covariance.  

     The hyperparameters of the function are:
       hyp = [ log(ell)
               log(sqrt(sf2)) 
               log(alpha) ]

     each row of x/z is a data point
    '''

    if hyp is None:  # report number of parameters
        return [3]

    ell = np.exp(hyp[0])  # characteristic length scale
    sf2 = np.exp(2. * hyp[1])  # signal variance
    alpha = np.exp(hyp[2])

    n, D = x.shape

    if z == 'diag':
        D2 = np.zeros((n, 1))
    elif z is None:
        D2 = spdist.cdist(x / ell, x / ell, 'sqeuclidean')
    else:
        D2 = spdist.cdist(x / ell, z / ell, 'sqeuclidean')

    if der is None:  # compute covariance matix for dataset x
        A = sf2 * ((1.0 + 0.5 * D2 / alpha) ** (-alpha))
    else:
        if der == 0:  # compute derivative matrix wrt 1st parameter
            A = sf2 * (1.0 + 0.5 * D2 / alpha) ** (-alpha - 1) * D2

        elif der == 1:  # compute derivative matrix wrt 2nd parameter
            A = 2. * sf2 * ((1.0 + 0.5 * D2 / alpha) ** (-alpha))

        elif der == 2:  # compute derivative matrix wrt 3rd parameter
            K = (1.0 + 0.5 * D2 / alpha)
            A = sf2 * K ** (-alpha) * (0.5 * D2 / K - alpha * np.log(K))
        else:
            raise Exception("Wrong derivative index in covRQiso")

    return A


def covRQard(hyp=None, x=None, z=None, der=None):
    ''' Rational Quadratic covariance function with Automatic Relevance Detemination
     (ARD) distance measure. The covariance function is parameterized as:

     k(x^p,x^q) = sf2 * [1 + (x^p - x^q)' * inv(P) * (x^p - x^q)/(2 * alpha)]^(-alpha)

     where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
     D is the dimension of the input space, sf2 is the signal variance and alpha is 
     the shape parameter for the RQ covariance. The hyperparameters are:

       hyp = [ log(ell_1)
               log(ell_2)
               ...
               log(ell_D)
               log(sqrt(sf2)) 
               log(alpha)]
    '''

    if hyp is None:  # report number of parameters
        return ['D + 2']  # USAGE: integer OR D_+_int (spaces are SIGNIFICANT)

    [n, D] = x.shape
    ell = 1 / np.exp(hyp[0:D])  # characteristic length scale
    sf2 = np.exp(2. * hyp[D])  # signal variance
    alpha = np.exp(hyp[D + 1])

    if z == 'diag':
        D2 = np.zeros((n, 1))
    elif z is None:
        tmp = np.dot(np.diag(ell), x.T).T
        D2 = spdist.cdist(tmp, tmp, 'sqeuclidean')
    else:
        D2 = spdist.cdist(np.dot(np.diag(ell), x.T).T, np.dot(np.diag(ell), z.T).T, 'sqeuclidean')

    if der is None:  # compute covariance matix for dataset x
        A = sf2 * ((1.0 + 0.5 * D2 / alpha) ** (-alpha))
    else:
        if der < D:  # compute derivative matrix wrt length scale parameters
            if z == 'diag':
                A = D2 * 0
            elif z is None:
                tmp = np.atleast_2d(x[:, der]) / ell[der]
                A = sf2 * (1.0 + 0.5 * D2 / alpha) ** (-alpha - 1) * spdist.cdist(tmp, tmp, 'sqeuclidean')
            else:
                A = sf2 * (1.0 + 0.5 * D2 / alpha) ** (-alpha - 1) * spdist.cdist(np.atleast_2d(x[:, der]).T / ell[der],
                                                                                  np.atleast_2d(z[:, der]).T / ell[der],
                                                                                  'sqeuclidean')
        elif der == D:  # compute derivative matrix wrt magnitude parameter
            A = 2. * sf2 * ((1.0 + 0.5 * D2 / alpha) ** (-alpha))

        elif der == (D + 1):  # compute derivative matrix wrt magnitude parameter
            K = (1.0 + 0.5 * D2 / alpha)
            A = sf2 * K ** (-alpha) * (0.5 * D2 / K - alpha * np.log(K))
        else:
            raise Exception("Wrong derivative index in covRQard")

    return A


def covNoise(hyp=None, x=None, z=None, der=None):
    ''' Independent covariance function, ie "white noise", with specified variance.
     The covariance function is specified as:

     k(x^p,x^q) = s2 * \delta(p,q)

     where s2 is the noise variance and \delta(p,q) is a Kronecker delta function
     which is 1 iff p=q and zero otherwise. The hyperparameter is

     hyp = [ log(sqrt(s2)) ]
    '''

    tol = 1.e-9  # Tolerance for declaring two vectors "equal"
    if hyp is None:  # report number of parameters
        return [1]

    s2 = np.exp(2. * hyp[0])  # noise variance
    n, D = x.shape

    if z == 'diag':
        A = np.ones((n, 1))
    elif z is None:
        A = np.eye(n)
    else:  # compute covariance between data sets x and z
        M = spdist.cdist(x, z, 'sqeuclidean')
        A = np.zeros_like(M, dtype=np.float)
        A[M < tol] = 1.

    if der is None:
        A = s2 * A
    else:  # compute derivative matrix
        if der == 0:
            A = 2. * s2 * A
        else:
            raise Exception("Wrong derivative index in covNoise")

    return A


if __name__ == "__main__":
    N = 10
    M = 8
    D = 5
    x = np.random.rand(N, D)
    y = np.random.rand(M, D)
    hyp = np.random.rand(D+1)
    K = covSEard(hyp, x, y)
    print(K.shape)