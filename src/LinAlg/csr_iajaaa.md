# CSR Format used by HiOp to save linear systems

Each of the linear systems saved by HiOp in a .iajaaa file (see [this](readme.md) for more information) consists of the systems's matrix, the right-hand side(s) (rhs), and the solution(s). The matrix is assumed to be square and the indexes are Fortran style (1-based). For KKT symmetric matrices only the elements from the upper or lower triangle are used. The lower triangle elements are saved for all symmetric KKT systems, while for the rest of the symmetric KKT systems (MDS and dense) the upper triangle elements are saved.

An example Matlab script that loads and solves such linear systems is provided [here](load_kkt_mat.m). 

The .iajaaa files contain

1. number of rows (nrows) [1 int]

2. number of primal variables, number of equality constraints, and number of inequality constraints of the underlying NLP problem [3 ints]

3. number of nonzeros (nnz) [1 int]

4. array of pointers/offsets in 4. and 5. of the first nonzero of each row; first entry is 1 and the last entry is nnz+1 [nrows+1 ints]

5. array of the column indexes of nonzeros [nnz ints]

6. array of nonzero entries  [nnz doubles]

7. rhs [nrows doubles]

8. solution [nrows doubles]

9. more rhs-solution pairs (repetitions of 7-8 above)

Please remark that there is a slight variation  of the .iajaaa format used by Ipopt (more exactly by Pardiso from within Ipopt), namely,
+ HiOp's also saves the solution, see 8. above;
+ multiple rhs-solution pairs can be present (*i.e.*,7-8 can repeat) at the end of the output files
