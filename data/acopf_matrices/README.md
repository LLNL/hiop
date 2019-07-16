# ACOPF (basecase) matrices/linear systems arising in interior-point methods

## Standard CSR Format ##

1. number of rows (nrows)

2. number of nonzeros (nnz)

3. column indexes (nrows+1 entries)

4. nonzero entries  (nnz entries)

5. right-hand side (nrows entries)

## Instances

1. net01 - small

2. net12 - larger but medium sized


Both instances are relatively well behaved numerically. They do not require inertia correction (PARDISO linear solver was used).

To be of use with interior-point solvers such as Ipopt and PIPS, the code that solves these linear systems should be also capable of returning the inertia of the matrix.

## References

OPF formulation from the [GO Competition](https://gocompetition.energy.gov/sites/default/files/SCOPF_Formulation_GO_Comp_20181130.pdf)
