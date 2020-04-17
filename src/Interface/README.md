HiOp supports two input formats: `hiopInterfaceDenseConstraints` and `hiopInterfaceMDS`. Both formats are in the form of C++ interfaces (e.g., abstract classes), see hiopInterface.hpp file, that the user must instantiate/implement and provide to HiOp.

Both interfaces are for specialized NLPs. Documentation is provided in hiopInterface.hpp. Below we discuss conventions made by  `hiopInterfaceMDS`

## Key points/conventions for `hiopInterfaceMDS`

MDS stands for mixed dense-sparse, meaning that the derivatives (Jacobian and Hessian) have both dense and sparse blocks. The `hiopInterfaceMDS` allows the user to pass these blocks to HiOp, which then exploits this structure using a specialized linear algebra implementation. This feature is especially relevant for GPU (hybrid) computing mode.


### Optimization variables

* the (vector of) optimization variables are supposed to be split in *dense* and *sparse* variables
* HiOp expects the optimization variables in a certain order: sparse variables first, followed by dense variables
  * the implementer/user (inconviniently) has to keep an map between his internal variables indexes and the indexes HiOp expects in order to avoid expensive memory moves inside HiOp
  
### Jacobian

* the columns in the Jacobian corresponding to sparse variables form the sparse Jacobian block (and can/should be provided as a sparse matrix, see `eval_Jac` function(s))
* the columns in the Jacobian corresponding to dense variables form the dense Jacobian block (and can/should be provided via the `double**` buffer provided by `eval_Jac` functions(); this buffer is primed to support double indexing, i.e., use `buffer[i][j]` to access the (i,j) element of the dense block)
* as indicated at the previous bullet, the user needs to map the column/variable index in the true Jacobian to the column index in the sparse or dense Jacobian block. The indexes inside the sparse and dense Jacobian blocks is zero-based

### Hessian
* the Hessian's structure is slightly different that that of the Jacobian. The Hessian has three relevant blocks
  * Hessian with respect to (sparse,sparse) variables
  * Hessian with respect to (dense,dense) variables	
  * Hessian with respect to (sparse,dense) variables, which is the transpose of (dense,sparse) Hessian. This blocks are ignored currentlty by HiOp (subject to change)
* all the above indexing rules for the Jacobian blocks apply to the Hessian blocks
* for conventions on symmetric matrices and sparse matrices see [this](../LinAlg/readme.md)
  
