# greedyCI


this module will provide functionality to compute "greedy" confidence intervals, where we do not demand symmetry around the median or MAP points but instead add weight where it is most probable.

This is accomplished with a combination of numerical techniques:

	(1) solve for MAP analytically
	(2) perform a binary search to the left to determine the "quantiles" associted with the requested CI (to within a given precision)
		(a) we pick a lower bound
		(b) numerically solve for the corresponding upper bound (defined by the requirement that the probability densities be equal at both bounds)
		(c) we integrate the probability density between the two bounds to get the associated confidence, which then informs the binary search

Currently, we support the following distributions:

	- Poisson (NOT IMPLEMENTED, BUT PLANNED)

In general, we assume conjugate priors where possible although the default will be "uniform" priors in some sense. This may differe from distribution to distribution.

