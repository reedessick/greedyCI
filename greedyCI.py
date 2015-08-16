description="""a module to compute "greedy" confidence intervals for several distributions"""

#===================================================================================================

import numpy as np

#===================================================================================================
# BASIC UTILITIES
#===================================================================================================

def factorial(n, nmax=100):
    """
    if n<nmax:
        compute factorial directly
    else:
        use Sterling's approximation
    """
    if n == 0:
        return 1
    elif n < nmax:
        return n*factorial(n-1, nmax=nmax)
    else:
        print "WARNING: you don't quite have Sterling's approximation correct..."
        return n**n - n ### THIS CAN BE IMPROVED

def log_factorial(n, nmax=100):
    if n == 0:
        return 0
    elif n < nmax:
        return np.log(n) + log_factorial(n-1, nmax=nmax)
    else:
        return n*np.log(n) - n

#===================================================================================================
# POISSON DISTRIBUTION
#===================================================================================================

def poisson_pdf(n, k):
    """
    p(n|k) = ( k**n / n! ) * e**{-k}
    """
    return k**n * np.exp(-k) / factorial(n)

def poisson_dpdk(n, k):
    """
    if n > 0:
        dp/dk = (1 - kH/n) * p(n-1|k)
    else:
        dp/dk = - e**{-k}
    """
    if n > 0:
        return (1-k/n) * poisson_pdf(n-1, k)
    else:
        return -np.exp(-k)

def poisson_map(n, alpha=1, beta=0):
    """
    finds the poisson MAP point given a Gamma distribution prior with parameters alpha and beta
    """
    kMAP = (n + alpha - 1) / (1 + beta)
    if kMAP < 0:
        raise ValueError("kMAP < 0")
    return kMAP

def poisson_greedyLOW2greedyHIGH(kLOW, n, rtol=1e-6, atol=1e-6):
    """
    returns the corresponding kHIGH for which p(n|kLOW) = p(n|kHIGH) by solving a transcendental equation using Newton's method
    """
    kMAP = poisson_map(n)
    if kLOW == kMAP:
        return kMAP

    pLOW = poisson_pdf(n, kLOW)

    kHIGH = kMAP + (kMAP-kLOW) ### reflect kLOW around kMAP as a starting point
    pHIGH = poisson_pdf(n, kHIGH)

    dp = abs(pLOW-pHIGH)
    while (dp > atol) or (2*dp > rtol*(pLOW+pHIGH)):
        ### predict new kHIGH, compute new pHIGH
        dpdkHIGH = poisson_dpdk(n, kHIGH)

        kHIGH = kHIGH - (pHIGH-pLOW)/dpdkHIGH
        pHIGH = poisson_pdf(n, kHIGH)
        dp = abs(pLOW-pHIGH)

    return kHIGH

def poisson_weight(n, klow, khigh, npts=1001):
    """
    computes the weight betweeen kLOW and kHIGH
    """
    ### either used closed form "incomplete gamma distribution" or compute numerically....
    k = np.linspace(klow, khigh, npts)

    ### trapazoidal approximation
    weight = 0.5*poisson_pdf(n, k[0]) ### normalization is wrong?
    for ind, K in enumerate(k[1:-1]):
        weight += poisson_pdf(n, K)
    weight += 0.5*poisson_pdf(n, k[-1])

    weight *= (k[1]-k[0]) ### multiply by spacing

    return weight

def poisson_bs(conf, n, rtol=1e-6, atol=1e-6):
    """
    finds the greedy CI for the confidence level "conf" given "n" observations
    performs a "bisection search" (thus the bs in the name), with termination conditions defined by rtol and atol
    """
    kLOW_l = 0
    kLOW_r = poisson_map(n, alpha=1, beta=0)
    if kLOW_r == kLOW_l: # kMAP==0
        raise ValueError("special case when kLOW_r == kLOW_l == 0")
    else:
        kLOW_m = 0.5*kLOW_r
        kHIGH = poisson_greedyLOW2greedyHIGH(kLOW_m, n, rtol=rtol, atol=atol)

    pl = 1
    pr = 0
    pm = poisson_weight( n, kLOW_m, kHIGH )

    dp = pm - conf
    absdp = abs(dp)
    while (absdp > atol) or (2*absdp > rtol*(conf+pm)):

        if dp > 0: ### too much confidence -> move to the right
            kLOW_l = kLOW_m
        else: ### too little confidence -> move to the left
            kLOW_r = kLOW_m

        kLOW_m = 0.5*(kLOW_l + kLOW_r)
        kHIGH = poisson_greedyLOW2greedyHIGH(kLOW_m, n, rtol=rtol, atol=atol)
        pm = poisson_weight( n, kLOW_m, kHIGH )
        dp = pm - conf
        absdp = abs(dp)

    return [kLOW_m, kHIGH]






