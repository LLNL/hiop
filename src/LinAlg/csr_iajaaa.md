# CSR Format used by HiOp to save linear systems

Each of the linear systems saved by HiOp in a .iajaaa file (see [this](readme.md) for more information) consists of the systems's matrix, the right-hand side(s) (rhs), and the solution(s). The matrix is assumed to be symmetric and the indexes are Fortran style (1-based). An example Matlab script that loads and solves such linear systems is provided [here](load_kkt_mat.m). 

The .iajaaa files contain

1. number of rows (nrows) [1 double]

2. number of nonzeros (nnz) [1 double]

3. array of pointers/offsets in 4. and 5. of the first nonzero of each row; first entry is 1 and the last entry is nnz+1 [nrows+1 ints]

4. array of the column indexes of nonzeros [nnz ints]

5. array of nonzero entries  [nnz doubles]

6. rhs [nrows doubles]

7. solution [nrows doubles]

8. more rhs-solution pairs (repetitions of 6-7 above)

Please remark that there is a slight variation  of the .iajaaa format used by Ipopt (more exactly by Pardiso from within Ipopt), namely,
+ HiOp's also saves the solution, see 7. below;
+ multiple rhs-solution pairs can be present (*i.e.*,6-7 can repeat) at the end of the output files
