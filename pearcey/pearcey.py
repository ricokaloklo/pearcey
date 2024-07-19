import numpy as np
from scipy.special import factorial, gamma, hyp1f1
from scipy.integrate import simps

def hyperu(a, b, z):
    """
    Tricomi's confluent hypergeometric function U(a, b; z)

    Parameters
    ----------
    a : float
        First parameter of the hypergeometric function
    b : float
        Second parameter of the hypergeometric function
    z : float/complex
        Argument of the hypergeometric function. Can be complex-valued
    
    Returns
    -------
    float
        The value of the Tricomi's confluent hypergeometric function U(a, b; z)
    """
    # Code adopted from https://stackoverflow.com/questions/49932930/tricomi-hypergeometric-u-in-python
    # See https://functions.wolfram.com/HypergeometricFunctions/HypergeometricU/02/0001/

    # First term of the equation
    term1 = gamma(1 - b) / gamma(a + 1 - b) * hyp1f1(a, b, z)

    # Second term of the equation
    term2 = gamma(b - 1) / gamma(a) * z**(1 - b) * hyp1f1(a + 1 - b, 2 - b, z)

    # Combine the terms
    result = term1 + term2
    return result

# Convergent series expansion of the Pearcey function using confluent hypergeometric functions
# See also https://ui.adsabs.harvard.edu/abs/2006JCoAM.190..437P/abstract
def _Pn(x, n):
    # See Eq. (5) in 1601.03615
    if np.real(x) >= 0:
        return 2**(-(n + 3./2)) * gamma(n + 1./2) * hyperu(n/2. + 1./4, 1./2, (x**2)/4.)
    else:
        return (1./4)*gamma(n/2. + 1./4) * hyp1f1(n/2. + 1./4, 1./2, (x**2)/4.) - (x/4.)*gamma(n/2. + 3./4) * hyp1f1(n/2. + 3./4, 3./2, (x**2)/4.)

def _P_using_confluent_hypergeometric_funcs(x, y, nmax=5):
    # See Eq. (4) in 1601.03615
    sum = 0
    for n in range(0, nmax+1):
        sum += ((-y**2)**n / factorial(2*n)) * _Pn(x, n)
    
    return sum

# Convergent power series expansion of the Pearcey function
# See also https://dlmf.nist.gov/36.8
def _a_n(x, y, n, cache={}):
    assert n >= 0, "n must be a non-negative integer"
    # See Eq. (3) in 1601.03615
    if n in cache.keys():
        return cache[n]
    
    if n == 0:
        a_n = 1
    elif n == 1:
        a_n = y
    else:
        a_n = (1./n)*( y * _a_n(x, y, n-1, cache=cache) + 2 * x * _a_n(x, y, n-2, cache=cache) )

    cache[n] = a_n
    return a_n

def _P_using_power_series(x, y, nmax=5):
    # See Eq. (3) in 1601.03615
    sum = 0
    _cached_a_n = {}
    for n in range(0, nmax+1):
        sum += (-1)**n * gamma((2*n + 1)/4.) * _a_n(x, y, 2*n, cache=_cached_a_n)

    return (1./4) * sum

def pearcey_numerical(alpha, beta, nstep=50):
    # Code adopted from https://gist.github.com/dpiponi/9176c7f6bf32803e9b2bf6e8c0b93ab5
    x = np.linspace(-4.0, 4.0, nstep)
    # f(z) = z⁴+αz²+β
    # g(z) = exp(if(z))
    # Instead of integrating along x-axis we're
    # going to integrate along a contour displaced
    # vertically from the x-axis.
    # A good choice of displacement is the gradient
    # d/(Im f(x+iy))/dy.
    # That way, we're displacing in a direction that makes
    # |exp(if(x+iy))| smaller.
    rate = 0.01
    y = rate*(4*x**3+2*alpha*x+beta)
    z = x+1j*y

    f = z**4+alpha*z**2+beta*z
    g = np.exp(1j*f)

    # ∫f(z)dz = ∫f(z)dz/dx dz
    dz = 1.0+1j*rate*(12*x**2+2*alpha)
    I = simps(g*dz, x)

    return I

def pearcey(x, y, algo="power-series", nmax=50):
    """
    Compute the Pearcey function/integral P(x, y) using the specified algorithm

    Parameters
    ----------
    x : float/complex
        First argument of the Pearcey function
    y : float
        Second argument of the Pearcey function
    algo : str, optional
        Algorithm to use to compute the Pearcey function.
        Can be either 'power-series', 'confluent-hypergeometric' or 'numerical'
    nmax : int, optional
        Number of terms to use in the expansion. Default is 50.
        If algo is 'numerical', nmax is the number of steps to use in the numerical integration

    Returns
    -------
    float/complex
        The value of the Pearcey function
    """
    if algo == "power-series":
        # See the inline expression just above Eq. (2) in 1601.03615
        return 2*np.exp(1j*np.pi/8) * _P_using_power_series(x * np.exp(-1j*np.pi/4), y * np.exp(1j*np.pi/8), nmax=nmax)
    elif algo == "confluent-hypergeometric":
        # See the inline expression just above Eq. (2) in 1601.03615
        return 2*np.exp(1j*np.pi/8) * _P_using_confluent_hypergeometric_funcs(x * np.exp(-1j*np.pi/4), y * np.exp(1j*np.pi/8), nmax=nmax)
    elif algo == "numerical":
        # Direct numerical integration
        return pearcey_numerical(x, y, nstep=nmax)
    else:
        raise ValueError("Invalid algorithm. Choose either 'power-series', 'confluent-hypergeometric' or 'numerical'.")

