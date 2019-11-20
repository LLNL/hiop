

### Some (sometimes contradicting) conventions

1. Symmetric dense matrices -> is a hiopMatrixDense with n()==m() with (i,j)==(j,i); however, in some circumstances,for example when the class is used to store a (symmetric) dense linear system matrix only upper triangular part is accessed and (cheaper) methods that update the upper triangular part are offered

2. Symmetric sparse matrices (triplet format) -> only upper triangular nz entries should be specified, accessed and maintained