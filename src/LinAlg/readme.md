# Some  conventions on matrices

## Dense matrices - hiopMatrixDense
Dense matrices have contiguous memory storage by rows that can be accessed by 
```cpp
double*  hiopMatrixDense::local_buffer() const;
```
Internally `hiopMatrixDense` also builds an array of pointers to each row start, which can be accessed by 
```cpp
double** hiopMatrixDense::get_M();
double** hiopMatrixDense::get_data() const;
```
This is done to allow double indexing specific to matrices, for example
```cpp
hiopMatrixDense W(10,10);
double** WM = W.get_M();
//set entry (7,10) to -17
WM[6][9] = -17.;
```
### *Symmetric* dense matrices 
`hiopMatrixDense` is also used for symmetric dense matrices by enforcing `n()==m()`. The general rule for users (such as in providing symmetric matrices to HiOp via the interface(s)) is to provide both the lower and upper triangle elements (so that `M(i,j)==M(j,i)`). 

However, internally, the `hiopMatrixDense` class may only store the **upper triangle** when it is used as a holder for KKT linearization systems. This is done for efficiency purposes. 

```cpp
/* block of W += alpha*this
 * For efficiency, only upper triangular matrix is updated since this will be eventually sent to LAPACK or MAGMA
 * Preconditions: 
 *  1. 'this' has to fit in the upper triangle of W 
 *  2. W.n() == W.m()
 */
 virtual void addToSymDenseMatrixUpperTriangle(int row_dest_start, int col_dest_start, double alpha, hiopMatrixDense& W) const;
 /* block of W += alpha*transpose(this)
 * For efficiency, only upper triangular matrix is updated since this will be eventually sent to LAPACK or MAGMA
 * Preconditions: 
 *  1. transpose of 'this' has to fit in the upper triangle of W 
 *  2. W.n() == W.m()
 */
 virtual void transAddToSymDenseMatrixUpperTriangle(int row_dest_start, int col_dest_start, double alpha, hiopMatrixDense& W) const;
```
If `this` contributes to `W`'s elements both in the upper and lower triangle, it would be unclear whether a symmetric update of `W` should be enforced or not (that is, when `this` contributes to `(i,j), i>j` one should or should not update `(j,i)` in `W`). To keep things simple and to avoid confusion, the two methods require (precondition) that *`this` matrix fits inside the upper triangle of `W`*. This precondition is always satisfied the off-diagonal blocks (Jacobians or their transposes) of the (KKT) symmetric linear system are updated.

For the diagonal blocks of symmetric linear system, the update is done from a (source) symmetric matrix and, hence, increased efficiency can be achieved by accessing only the upper triangle of the source (and, as before, updating only the upper triangle of the destination). The following method in `hiopMatrix` interface should be used/implemented
```cpp
/* diagonal block of W += alpha*this with 'diag_start' indicating the diagonal entry of W where
 * 'this' should start to contribute.
 * 
 * For efficiency, only upper triangle of W is updated since this will be eventually sent to LAPACK
 * and only the upper triangle of 'this' is accessed
 * 
 * Preconditions: 
 *  1. this->n()==this-m()
 *  2. W.n() == W.m()
 */
virtual void addUpperTriangleToSymDenseMatrixUpperTriangle(int diag_start, double alpha, hiopMatrixDense& W) const;
```

## Sparse matrices 
Triplet format is momentarily used for sparse matrices. The index arrays `i` and `j` used to store sparse matrices in triplet format need to be kept sorted by the following comparison rules: 
 - row indexes `i` in increasing order
 - column indexes `j` for the same row index `i` in strictly increasing order.
 
 Please remark that this does not allow duplicated entries. In a couple of places, HiOp may internally relax this requirement; however, this is  documented at the method level; otherwise, the precondition of sorted (i,j) entries holds for all other methods related to sparse matrices.
 
HiOp offers support for converting sparse triplet format to compressed sparse row (CSR) format, which, for example, is used with PARDISO and STRUMPACK linear solvers. It is worth mentioning that HiOp will also sort the arrays used by the CSR format based on the same rule, which seems to increase robustness with the third party linear solvers that require the CSR format.

### *Symmetric* sparse matrices 
Only upper triangular nonzero entries should be specified, accessed, and maintained.
The Hessian and the symmetric KKT systems are implemented as symmetric matrices. Users only need to provide the upper triangle nonzero entries to Hessian.
For the symmetric KKT linearizations, some linear algebra package, e.g., MA57 from HSL, can read entries from both the upper and lower triangular part, however, only one from the entries (i,j) and (j,i) is required to be passed to HSL solvers. 
HiOp developers should remark that the internal sorting rules for triplet format enable efficient copying of the constraint Jacobian and Lagrangian Hessian in the KKT linear system matrix; also, the symmetric KKT linear systems store only the lower triangle entries for the sake of efficiency.


## Obtaining matrices from HiOp

One can instruct HiOp to save the KKT linear systems solved internally during the optimization by setting 'write_kkt' string option to 'yes'. Which linear system is saved depends on the configuration of HiOp's internal linear algebra via the option 'KKTLinsys' (possible values 'xycyd', 'xdycyd' and 'full').

The output format is based on compressed sparse row (CSR) described [here](csr_iajaaa.md). A Matlab script that loads and solves such linear systems is provided [here](load_kkt_mat.m).
