# ACOPF (basecase) matrices/linear systems arising in interior-point methods

## Standard CSR Format ##

The matrix is assumed to symmetric and the indexes are Fortran style (1-based).

The .iajaaa files contain

1. number of rows (nrows) [1 double]

2. number of nonzeros (nnz) [1 double]

3. array of pointers/offsets in 4. and 5. of the first nonzero of each row; first entry is 1 and the last entry is nnz+1 [nrows+1 ints]

4. array of the column indexes of nonzeros [nnz ints]

5. array of nonzero entries  [nnz doubles]

6. right-hand side [nrows doubles]

## Instances

1. net01 - small

2. net12 - larger but medium sized


Both instances are relatively well behaved numerically. They do not require inertia correction (PARDISO linear solver was used).

To be of use with interior-point solvers such as Ipopt and PIPS, the code that solves these linear systems should be also capable of returning the inertia of the matrix.

## References

OPF formulation from the [GO Competition](https://gocompetition.energy.gov/sites/default/files/SCOPF_Formulation_GO_Comp_20181130.pdf)
