# scqpth
SCQPTH is a differentiable first-order splitting method for convex quadratic programs. The QP solver is implemented as a custom PyTorch module. The forward solver invokes a basic implementation of the ADMM algorithm. Backward differentiation is performed by implicit differentiation of a fixed-point mapping customized to the ADMM algorithm.

For more information please see our publication:

[arXiv (preprint)](https://github.com/ipo-lab/scqpth)

For experimental results please see [scqpth_bench](https://github.com/ipo-lab/scqpth_bench)

## Core Dependencies:
To use the ADMM solver you will need to install [numpy](https://numpy.org) and [Pytorch](https://pytorch.org). 
