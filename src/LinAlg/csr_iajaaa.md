# (Modified) CSR Format used by HiOp to save linear systems

The saved linear system consists of the matrix, right-hand side (rhs), and solution. The matrix is assumed to be symmetric and the indexes are Fortran style (1-based). A Matlab script that loads and solves such linear systems is provided here. This is a slight variation  of the .iajaaa format used by Ipopt (more exactly by Pardiso from within Ipopt). The only differences are that i. HiOp's also saves the solution, see 7. below and multiple rhs and solution pair can be present at the end of the output files.

The .iajaaa files contain

1. number of rows (nrows) [1 double]

2. number of nonzeros (nnz) [1 double]

3. array of pointers/offsets in 4. and 5. of the first nonzero of each row; first entry is 1 and the last entry is nnz+1 [nrows+1 ints]

4. array of the column indexes of nonzeros [nnz ints]

5. array of nonzero entries  [nnz doubles]

6. rhs [nrows doubles]

7. solution [nrow doubles]