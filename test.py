import time

from greedyCI import *

n = 20
conf = 0.68

y = []
for x in xrange(100):
    to = time.time()
    klow, khigh = poisson_bs(conf, n)
    y.append( time.time()-to )

print np.mean(y), np.std(y)
