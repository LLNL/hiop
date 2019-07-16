# ACOPF (basecase) matrices/linear systems from the interior-point solver Ipopt

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

## References

OPF formulation from the [GO Competition](https://gocompetition.energy.gov/sites/default/files/SCOPF_Formulation_GO_Comp_20181130.pdf)
