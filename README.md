# Mesh Adaptive Direct Search (MADS)

This is a pure julia implementation of (Robust)LtMADS and (Robust)OrthoMADS for
blackbox optimization. See [NOMAD.jl](https://github.com/ppascal97/NOMAD.jl) for
a julia wrapper of [NOMAD](https://www.gerad.ca/nomad/).

## Installation

```julia
] add https://github.com/jbrea/MeshAdaptiveDirectSearch.jl
```

## Usage
```julia
using MeshAdaptiveDirectSearch

f(x) = (1 - exp(-sum(abs2, x))) * max(sum(abs2, x .- [30, 40]), sum(abs2, x .+ [30, 40]))
noisyf(x) = f(x) + .1 * randn()

?minimize

minimize(LtMADS(2), f, [-2.1, 1.7], lowerbound = [-10, -10], upperbound = [10, 10])
minimize(LtMADS(2), f, [-2.1, 1.7], lowerbound = [-10, -10], upperbound = [10, 10], constraints = [x -> sum(x) < .5])
minimize(OrthoMADS(2), f, [-2.1, 1.7], lowerbound = [-10, -10], upperbound = [10, 10])
minimize(RobustLtMADS(2), noisyf, [-2.1, 1.7], lowerbound = [-10, -10], upperbound = [10, 10])
minimize(RobustOrthoMADS(2), noisyf, [-2.1, 1.7], lowerbound = [-10, -10], upperbound = [10, 10])
```

## References
Audet, Charles and Dennis, J. E., "Mesh Adaptive Direct Search Algorithms for
Constrained Optimization", 2006, [doi](http://dx.doi.org/10.1137/040603371)

Audet, Charles and Ihaddadene, Amina and Le Digabel, Sébastien and Tribes,
Christophe, "Robust optimization of noisy blackbox problems using the Mesh
Adaptive Direct Search algorithm", 2018, [doi](http://dx.doi.org/10.1007/s11590-017-1226-6)

Abramson, Mark A. and Audet, Charles and Dennis, J. E. and Le Digabel,
Sébastien, "OrthoMADS: A Deterministic MADS Instance with Orthogonal
Directions", 2009, [doi](http://dx.doi.org/10.1137/080716980).
