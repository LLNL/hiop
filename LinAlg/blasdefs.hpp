#ifndef HIOP_BLASDEFS
#define HIOP_BLASDEFS

extern "C" double dnrm2_(int* n, double* x, int* incx);
extern "C" double ddot_ (int* n, double* dx, int* incx, double* dy, int* incy);
extern "C" void   dscal_(int* n, double* da, double* dx, int* incx);
extern "C" void   daxpy_(int* n, double* da, double* dx, int* incx, double* dy, int* incy );
extern "C" void   dcopy_(int* n,  double* da, int* incx, double* dy, int* incy);
extern "C" void   dgemv_(char* trans, int* m, int* n, double* alpha, double* a, int* lda,
			 const double* x, int* incx, double* beta, double* y, int* incy );
/* C := alpha*op( A )*op( B ) + beta*C
 * op( A ) an m by k matrix, op( B ) a  k by n matrix and C an m by n matrix
 */
extern "C" void   dgemm_(char* transA, char* transB, int* m, int* n, int* k,
			 double* alpha, double* a, int* lda,
			 double* b, int* ldb,
			 double* beta, double* C, int*ldc);


/* op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,
 * where alpha is a scalar, X and B are m by n matrices, A is a unit, or
 * non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
 *    op( A ) = A   or   op( A ) = A**T.
 *
 * The matrix X is overwritten on B.
 */
//!opt DTPTRS packed format triangular solve
extern "C" void   dtrsm_(char* side, char* uplo, char* transA, char* diag, 
			 int* M, int* N, 
			 double* alpha, 
			 const double* a, int* lda,
			 double* b, int* ldb);

/* Cholesky factorization of a real symmetric positive definite matrix A.
 * The factorization has the form
 *   A = U**T * U,  if UPLO = 'U', or  A = L  * L**T,  if UPLO = 'L',
 * where U is an upper triangular matrix and L is lower triangular.
 */
extern "C" void   dpotrf_(char* uplo, int* N, double* A, int* lda, int* info);

/* solves a system of linear equations A*X = B with a symmetric
 * positive definite matrix A using the Cholesky factorization
 * A = U**T*U or A = L*L**T computed by DPOTRF
 * A contains  the triangular factor U or L 
*/
extern "C" void   dpotrs_(char* uplo, int* N, int* NRHS, 
			  double*A, int* lda, 
			  double* B, int* ldb,
			  int* info);
extern "C" double   dlange_(char* norm, int* M, int* N, double*A, int* lda, double* work);
#endif
