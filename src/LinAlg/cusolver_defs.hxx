

#ifndef CUSOLVERDEFS_H
#define CUSOLVERDEFS_H
#if 1
#include "cusparse.h"
#include "cusolverSp.h"
#include <assert.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "cusolverSp_LOWLEVEL_PREVIEW.h"

#include "cholmod.h"
#include "klu.h"
#endif
#define REAL double
#define INT int

template <typename T_ELEM>
int loadMMSparseMatrix(
    char *filename,
    char elem_type,
    bool csrFormat,
    int *m,
    int *n,
    int *nnz,
    T_ELEM **aVal,
    int **aRowInd,
    int **aColInd,
    int extendSymMatrix);

template <typename T>
void check(T result, char const *const func, const char *const file,
    int const line) {
  if (result) {
    printf("CUDA error at %s:%d, error# %d\n", file, line, result);
    exit(EXIT_FAILURE);
  }
}
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

static REAL second (void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (REAL)tv.tv_sec + (REAL)tv.tv_usec / 1000000.0;
}


static REAL vec_norminf(int n, const REAL *x)
{
  REAL norminf = 0;
  for(int j = 0 ; j < n ; j++){
    REAL x_abs = fabs( (REAL) x[j]);
    norminf = (norminf > x_abs)? norminf : x_abs;
  }
  return norminf;
}

/**************************************
 * |A| = max { |A|*ones(m,1) }
 **************************************/
static REAL csr_mat_nrminf(
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const REAL *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA)
{
  const int baseA = (CUSPARSE_INDEX_BASE_ONE == cusparseGetMatIndexBase(descrA))? 1:0;

  REAL norminf = 0;
  for(int i = 0 ; i < m ; i++){
    REAL sum = 0.0;
    const int start = csrRowPtrA[i  ] - baseA;
    const int end   = csrRowPtrA[i+1] - baseA;
    for(int colidx = start ; colidx < end ; colidx++){
      REAL A_abs = fabs( (REAL) csrValA[colidx] );
      sum += A_abs;
    }
    norminf = (norminf > sum)? norminf : sum;
  }
  return norminf;
}

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

  /*
   * prototype not in public header file 
   */
  cusolverStatus_t CUSOLVERAPI cusolverSpXcsrluConfigHost(
      csrluInfoHost_t info,
      int reorder  /* 0 or 1 */
      );

  cusolverStatus_t CUSOLVERAPI cusolverSpDcsrlucondHost(
      csrluInfoHost_t info,
      REAL *maxDiagU,
      REAL *minDiagU,
      REAL *maxL
      );

  cusolverStatus_t CUSOLVERAPI cusolverSpXcsrluNnzMHost(
      cusolverSpHandle_t handle,
      int *nnzMRef,
      csrluInfoHost_t info);

  cusolverStatus_t CUSOLVERAPI cusolverSpDcsrluExtractMHost(
      cusolverSpHandle_t handle,
      int *P, 
      int *Q, 
      const cusparseMatDescr_t descrM,
      REAL *csrValM, 
      int *csrRowPtrM, 
      int *csrColIndM, 
      csrluInfoHost_t info,
      void *pBuffer   
      );

  struct csrgluInfo;
  typedef struct csrgluInfo *csrgluInfo_t;

  cusolverStatus_t CUSOLVERAPI cusolverSpCreateGluInfo(
      csrgluInfo_t *info);

  cusolverStatus_t CUSOLVERAPI cusolverSpDestroyGluInfo(
      csrgluInfo_t info);

  cusolverStatus_t CUSOLVERAPI cusolverSpDgluSetup(
      cusolverSpHandle_t handle,
      int m,
      /* A can be base-0 or base-1 */
      int nnzA,
      const cusparseMatDescr_t descrA,
      const int *h_csrRowPtrA,
      const int *h_csrColIndA,
      const int *h_P, /* base-0 */
      const int *h_Q, /* base-0 */
      /* M can be base-0 or base-1 */
      int nnzM,
      const cusparseMatDescr_t descrM,
      const int *h_csrRowPtrM,
      const int *h_csrColIndM,
      csrgluInfo_t info);

  cusolverStatus_t CUSOLVERAPI cusolverSpDgluBufferSize(
      cusolverSpHandle_t handle,
      csrgluInfo_t info,
      size_t *pBufferSize);

  cusolverStatus_t CUSOLVERAPI cusolverSpDgluAnalysis(
      cusolverSpHandle_t handle,
      csrgluInfo_t info,
      void *workspace);

  cusolverStatus_t CUSOLVERAPI cusolverSpDgluReset(
      cusolverSpHandle_t handle,
      int m,
      /* A is original matrix */
      int nnzA,
      const cusparseMatDescr_t descr_A,
      const REAL *d_csrValA,
      const int *d_csrRowPtrA,
      const int *d_csrColIndA,
      csrgluInfo_t info);

  cusolverStatus_t CUSOLVERAPI cusolverSpDgluFactor(
      cusolverSpHandle_t handle,
      csrgluInfo_t info,
      void *workspace);

  cusolverStatus_t CUSOLVERAPI cusolverSpDgluSolve(
      cusolverSpHandle_t handle,
      int m,
      /* A is original matrix */
      int nnzA,
      const cusparseMatDescr_t descr_A,
      const REAL *d_csrValA,
      const int *d_csrRowPtrA,
      const int *d_csrColIndA,

      const REAL *d_b0, /* right hand side */
      REAL *d_x, /* left hand side */
      int *ite_refine_succ,
      REAL *r_nrminf_ptr,
      csrgluInfo_t info,
      void *workspace);

  cusolverStatus_t CUSOLVERAPI cusolverSpDnrminf(
      cusolverSpHandle_t handle,
      int n,
      const REAL *x,
      REAL *result, /* |x|_inf, host */
      void *d_work  /* at least 8192 bytes */
      );

#if defined(__cplusplus)
}
#endif /* __cplusplus */


static void check_residaul (
    cusolverSpHandle_t handle,
    cusparseHandle_t sp_handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    REAL *d_csrValA,
    int *d_csrRowPtrA,
    int *d_csrColIndA,
    REAL *d_x,
    const REAL *d_b,
    REAL *r_nrminf_ptr,
    REAL *b_nrminf_ptr,
    REAL *x_nrminf_ptr)
{
  REAL *d_r = NULL; /* r = b - A*x */
  cudaError_t cudaStat1 = cudaSuccess;
  cusparseStatus_t sp_status = CUSPARSE_STATUS_SUCCESS;
  cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
  const size_t size_nrminf = 8192;
  char *d_work = NULL;

  const REAL h_one = 1.0;
  const REAL h_minus_one = -1;

  REAL r_nrminf = 0.;
  REAL b_nrminf = 0.;
  REAL x_nrminf = 0.;

  cudaStat1 = cudaMalloc ((void**)&d_r, sizeof(REAL) * n);
  assert( cudaSuccess == cudaStat1 );
  cudaStat1 = cudaMalloc ((void**)&d_work, size_nrminf);
  assert( cudaSuccess == cudaStat1 );

  cudaDeviceSynchronize();
  /* r = b - A*x */
  cudaStat1 = cudaMemcpy(d_r, d_b, sizeof(REAL) * n, cudaMemcpyDeviceToDevice);
  assert( cudaSuccess == cudaStat1 );
  cudaDeviceSynchronize();

#ifdef CUDA10
  sp_status = cusparseDcsrmv(
      sp_handle,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      n,
      n,
      nnzA,
      &h_minus_one,  /* alpha */
      descrA,
      d_csrValA,
      d_csrRowPtrA,
      d_csrColIndA,
      d_x,
      &h_one, /* beta */
      d_r);
  assert(CUSPARSE_STATUS_SUCCESS == sp_status);
#else  // for CUDA11                                                                                                    
  cusparseSpMatDescr_t matA = NULL;
  sp_status = cusparseCreateCsr(&matA, n, n, nnzA,
      d_csrRowPtrA, d_csrColIndA, d_csrValA,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
  assert(CUSPARSE_STATUS_SUCCESS == sp_status);

  // initialize dense vector descriptor vecx for the soluton                                                          
  cusparseDnVecDescr_t vecx = NULL;
  cusparseCreateDnVec(&vecx, n, d_x, CUDA_R_64F);
  cusparseDnVecDescr_t vecAx = NULL;
  cusparseCreateDnVec(&vecAx, n, d_r, CUDA_R_64F);

  size_t bufferSize = 0;
  sp_status = cusparseSpMV_bufferSize(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &h_minus_one, matA, vecx,
      &h_one, vecAx,
      CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, &bufferSize);
  assert(CUSPARSE_STATUS_SUCCESS == sp_status);
  void *buffer = NULL;
  cudaMalloc(&buffer, bufferSize);
  sp_status = cusparseSpMV(sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &h_minus_one, matA, vecx, &h_one, vecAx, CUDA_R_64F,
      CUSPARSE_MV_ALG_DEFAULT, &buffer);
  assert(CUSPARSE_STATUS_SUCCESS == sp_status);

#endif
  status = cusolverSpDnrminf(
      handle,
      n,
      d_r,
      &r_nrminf,
      d_work  /* at least 8192 bytes */
      );
  assert(CUSOLVER_STATUS_SUCCESS == status);
  /* |b|_inf */
  status = cusolverSpDnrminf(
      handle,
      n,
      d_b,
      &b_nrminf,
      d_work  /* at least 8192 bytes */
      );
  assert(CUSOLVER_STATUS_SUCCESS == status);

  /* |x|_inf */
  status = cusolverSpDnrminf(
      handle,
      n,
      d_x,
      &x_nrminf,
      d_work  /* at least 8192 bytes */
      );
  assert(CUSOLVER_STATUS_SUCCESS == status);
  cudaDeviceSynchronize();

  *r_nrminf_ptr = r_nrminf;
  *b_nrminf_ptr = b_nrminf;
  *x_nrminf_ptr = x_nrminf;

  if (d_r   )   cudaFree(d_r);
  if (d_work)   cudaFree(d_work);
  //if (buffer)   cudaFree(buffer);                                                                                   
}

static void show_deviceInfo(int deviceId)
{
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, deviceId);
  printf("device %d, %s, cc %d.%d \n", deviceId, prop.name, prop.major, prop.minor);
  printf("   multiGPUBard=%d\n", prop.isMultiGpuBoard);
  printf("   # of SMs = %d\n", prop.multiProcessorCount);
  printf("   threads per SM = %d\n", prop.maxThreadsPerMultiProcessor);
  printf("   regs per SM = %d\n", prop.regsPerMultiprocessor);
  printf("   smem per SM = %lld bytes\n", (long long) prop.sharedMemPerMultiprocessor);
}
#if 0
void print_vector(
    int n,
    const int *h_P, /* base-0 */
    const char *name
    )
{
  printf("show %s, size %d with base-1\n", name, n );
  for(int j = 0 ; j < n ; j++ ){
    printf("%s(%d) = %d\n", name, j+1, h_P[j]+1);
  }
}

void print_csr(
    int n,
    int nnz,
    const int *csrRowPtr, /* base-0 */
    const int *csrColInd,
    const double *csrVal,
    const char *name)
{
  printf("matrix %s, %d-by-%d, with nnz =%d with base-1 \n", name, n, n, nnz);
  for (int row = 0; row < n; row++){
    const int start = csrRowPtr[row  ];
    const int end   = csrRowPtr[row+1];
    for (int colidx = start; colidx < end ; colidx++){
      const int col = csrColInd[ colidx ];
      if (NULL != csrVal){
        double Areg = csrVal[colidx];
        printf("%s(%d, %d) = %20.16E\n", name, row+1, col+1, Areg);
      }else{
        printf("%s(%d, %d)\n", name, row+1, col+1);
      }
    }
  }
}


void print_csc(
    int n,
    int nnz,
    const int *cscColPtr, /* base-0 */
    const int *cscRowInd,
    const double *cscVal,
    const char *name)
{
  printf("matrix %s, %d-by-%d, with nnz =%d with base-1 \n", name, n, n, nnz);
  for (int col = 0; col < n; col++){
    const int start = cscColPtr[col  ];
    const int end   = cscColPtr[col+1];
    for (int colidx = start; colidx < end ; colidx++){
      const int row = cscRowInd[ colidx ];
      if (NULL != cscVal){
        double Areg = cscVal[colidx];
        printf("%s(%d, %d) = %20.16E\n", name, row+1, col+1, Areg);
      }else{
        printf("%s(%d, %d)\n", name, row+1, col+1);
      }
    }
  }
}


#endif
#if 0
int parse_klu(
    klu_symbolic *Symbolic,
    klu_numeric *Numeric,
    klu_common *Common,
    int n,
    /* A is CSC, base-0 */
    int nnzA,
    int *Ap,  /* <int> n+1 */
    int *Ai,  /* <int> nnzA */
    double *Ax, /* <double> nnzA */
    /* M is CSC, base-0 */
    int nnzM,
    int *csrRowPtrM,  /* <int> n+1 */
    int *csrColIndM,  /* <int> nnzM */
    int *P,  /* <int> n */
    int *Q   /* <int> n */
    )
{
  /* workspace */
  int* Lp = NULL;
  int* Li = NULL;
  double *Lx = NULL;
  int* Up = NULL;
  int* Ui = NULL;
  double *Ux = NULL;
  int* Fp = NULL;
  int* Fi = NULL;
  double* Fx = NULL;
  int* R = NULL;
  double* Rs = NULL;

  int *csrRowPtrTemp = NULL;
  int *csrRowPtrA = NULL;
  int *csrColIndA = NULL;
  double *csrValA = NULL;



  csrRowPtrTemp = (int*)malloc(sizeof(int)*n);
  csrRowPtrA    = (int*)malloc(sizeof(int)*(n+1));
  csrColIndA    = (int*)malloc(sizeof(int)*nnzA);
  csrValA       = (double*)malloc(sizeof(double)*nnzA);
  if ( (NULL == csrRowPtrTemp) ||
      (NULL == csrRowPtrA) ||
      (NULL == csrColIndA) ||
      (NULL == csrValA)
     ){
    return 1;
  }

  /*
   * step 1: convert csc(A) to CSR(A)
   */

  /*
   * step 1.1: compute csrRowPtrA[i+1] = |A(i,:)| 
   */
  //    printf("step 1.1: compute csrRowPtrA[i+1] = |A(i,:)| \n");
  memset(csrRowPtrA, 0, sizeof(int)*(n+1));
  for(int col = 0 ; col < n ; col++){
    const int start = Ap[col];
    const int end   = Ap[col+1];
    for(int colidx = start ; colidx < end; colidx++){
      const int row = Ai[colidx];
      csrRowPtrA[row+1]++;
    }
  }
  /*
   * step 1.2: compute csrRowPtrA
   */
  //    printf("step 1.2: compute csrRowPtrA \n");
  for(int i = 0 ; i < n ; i++){
    csrRowPtrA[i+1] += csrRowPtrA[i];
  }
  /*
   * step 1.3: compute csrColIndA and csrValA
   */
  //   printf("step 1.3: compute csrColIndA and csrValA \n");
  memcpy(csrRowPtrTemp, csrRowPtrA, sizeof(int)*n);
  for(int col = 0 ; col < n ; col++){
    const int start = Ap[col];
    const int end   = Ap[col+1];
    for(int colidx = start ; colidx < end; colidx++){
      const int row = Ai[colidx];
      const double val = Ax[colidx];
      /* A(row, col) = val */
      const int startA = csrRowPtrTemp[row];
      csrColIndA[startA] = col;
      csrValA[startA] = val; 
      csrRowPtrTemp[row] = startA + 1;
    }
  }


  /*
   * step 1.4: copy back to Ap, Ai and Ax
   */
  //    printf("step 1.4: copy back to Ap, Ai and Ax \n");
  memcpy(Ap, csrRowPtrA, sizeof(int)*(n+1));
  memcpy(Ai, csrColIndA, sizeof(int)*nnzA);
  memcpy(Ax, csrValA   , sizeof(double)*nnzA);

  if (NULL != csrRowPtrA) free(csrRowPtrA);
  if (NULL != csrColIndA) free(csrColIndA);
  if (NULL != csrValA   ) free(csrValA);

  /*
   * step 2: copy P and Q
   */
  //    printf("step 2: copy P and Q \n");
  memcpy(P, Numeric->Pnum, sizeof(int)*n);
  memcpy(Q, Symbolic->Q, sizeof(int)*n);

  // Get the nnz in L, U and off-diagonal blocks
  const int Lnz = Numeric->lnz;
  const int Unz = Numeric->unz;
  //    const int nzoff = Numeric->nzoff;
  //    printf("lnz = %d, unz = %d, nzoff = %d\n", Lnz,Unz,nzoff);

  Lp = (int*)malloc((n+1)*sizeof(int));
  Li = (int*)malloc(Lnz*sizeof(int));
  Lx = (double*)malloc(Lnz*sizeof(double));
  Up = (int*)malloc((n+1)*sizeof(int));
  Ui = (int*)malloc(Unz * sizeof(int));
  Ux = (double*)malloc(Unz * sizeof(double));
  if ( (NULL == Lp) || 
      (NULL == Li) ||
      (NULL == Up) || 
      (NULL == Ui) ||
      (NULL == Lx) ||
      (NULL == Ux)
     ){
    printf("host allocation: failed\n");
    return 1;
  }

  /*
   * step 3: extract L and U which are CSC 
   */
  //    printf("step 3: extract L and U which are CSC \n");
  /* sorting is not necessary because csc2csr will sort row indices automatically */
#if 0
  int ok = 0;
  ok = klu_sort (Symbolic, Numeric, Common);
  if(!ok){
    printf("klu_sort: failed\n");
    return 1;
  }
#endif

  /* 
   * Lx and Ux must be valid, otherwise the csc(L) and csc(U) are not assigned.
   */
  int ok = klu_extract(Numeric, Symbolic, 
Lp, Li, Lx, 
Up, Ui, Ux, 
Fp, Fi, Fx, 
NULL, NULL, 
Rs, R, Common);
  if(!ok){
    printf("klu_extract: failed\n");
    return 1;
  }

#ifdef SHOW_DEBUG_INFO
  print_csc(n, Lnz, Lp, Li, Lx, "L");
  print_csc(n, Unz, Up, Ui, Ux, "U");
#endif


  /*
   * step 4: M = L+U 
   *
   * L and U are CSC, row indices are not sorted.
   */
  /*
   * step 4.1: compute csrRowPtrM[i+1] = nnz(M(i, :)) 
   */ 
  //printf("step 4.1: compute csrRowPtrM[i+1] = nnz(M(i, :)) \n");
  memset(csrRowPtrM, 0, sizeof(int)*(n+1));
  for(int col = 0 ; col < n ; col++){
    const int start = Lp[col];
    const int end   = Lp[col+1];
    for(int colidx = start ; colidx < end; colidx++){
      const int row = Li[colidx];
      /* remove diag(L) */
      if ( col != row ){
        csrRowPtrM[row+1]++;
      }
    }
  }

  for(int col = 0 ; col < n ; col++){
    const int start = Up[col];
    const int end   = Up[col+1];
    for(int colidx = start ; colidx < end; colidx++){
      const int row = Ui[colidx];
      csrRowPtrM[row+1]++;
    }
  }

  /*
   * step 4.2: compute csrRowPtrM
   */
  //printf("step 4.2: compute csrRowPtrM \n");
  for(int i = 0 ; i < n ; i++){
    csrRowPtrM[i+1] += csrRowPtrM[i];
  }

  /*
   * step 4.3: compute csrColIndM
   */
  //printf("step 4.3: compute csrColIndM \n");
  memcpy(csrRowPtrTemp, csrRowPtrM, sizeof(int)*n);
  for(int col = 0 ; col < n ; col++){
    const int start = Lp[col];
    const int end   = Lp[col+1];
    for(int colidx = start ; colidx < end; colidx++){
      const int row = Li[colidx];
      /* remove diag(L) */
      if ( col != row ){
        const int startM = csrRowPtrTemp[row];
        csrColIndM[startM] = col;
        csrRowPtrTemp[row] = startM+1; 
      }
    }
  }

  for(int col = 0 ; col < n ; col++){
    const int start = Up[col];
    const int end   = Up[col+1];
    for(int colidx = start ; colidx < end; colidx++){
      const int row = Ui[colidx];
      const int startM = csrRowPtrTemp[row];
      csrColIndM[startM] = col;
      csrRowPtrTemp[row] = startM+1; 
    }
  }

  if (NULL != Lp) free(Lp);
  if (NULL != Li) free(Li);
  if (NULL != Lx) free(Lx);
  if (NULL != Up) free(Up);
  if (NULL != Ui) free(Ui);
  if (NULL != Ux) free(Ux);
  if (NULL != csrRowPtrTemp) free(csrRowPtrTemp);

  return 0;
}
#endif
#endif
