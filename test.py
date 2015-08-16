import time

from greedyCI import *

### poisson test
n = 4
conf = 0.68

klow, khigh = poisson_bs(conf, n)
assert((abs(klow-2.2976226806640625)<1e-4) and (abs(khigh-6.3874831975541984)<1e-4))

### binomial test
n = 4
N = 20
conf = 0.68

klow, khigh = binomial_bs(conf, n, N)

