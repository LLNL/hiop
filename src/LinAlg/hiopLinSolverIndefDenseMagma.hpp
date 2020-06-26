#ifndef HIOP_MAGMASOLVER
#define HIOP_MAGMASOLVER

#include "hiopLinSolver.hpp"

#ifdef HIOP_USE_MAGMA
#include "magma_v2.h"

/** Notes:
 * *** Bunch-Kaufmann ***
 * magma_dsytrf(magma_uplo_t uplo, magma_int_t n, double *A, magma_int_t lda, magma_int_t *ipiv, magma_int_t *info)
 *  - no _gpu version
 *
 * *** same for Aasen ***
 * 
 * *** no pivoting ***
 * magma_int_t magma_dsytrf_nopiv (magma_uplo_t uplo, magma_int_t n, double *A, magma_int_t lda, magma_int_t *info)
 *magma_int_t magma_dsytrf_nopiv_gpu (magma_uplo_t uplo, magma_int_t n, magmaDouble_ptr dA, magma_int_t ldda, magma_int_t *info)
 * Guidelines on when to use _gpu ?
 *
 *
 *  Forward and backsolves
 *  magma_int_t magma_dsytrs_nopiv_gpu(magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, magmaDouble_ptr dA, magma_int_t ldda, magmaDouble_ptr dB, magma_int_t lddb, magma_int_t * info)
 *
 * How about when use (cpu) magma_dsytrf? What dsytrs function to use? 
 * In the example, the (triu) solves are done with blas blasf77_dsymv
 * 
 *
 */

namespace hiop {
class hiopLinSolverIndefDenseMagmaDev : public hiopLinSolverIndefDense
{
public:
  hiopLinSolverIndefDenseMagmaDev(int n, hiopNlpFormulation* nlp_)
    : hiopLinSolverIndefDense(n, nlp_)
  {

    ipiv = new int[n];
    dwork = new hiopVectorPar(0);


    magma_int_t ndevices;
    magma_device_t devices[ MagmaMaxGPUs ];
    magma_getdevices( devices, MagmaMaxGPUs, &ndevices );
    assert(ndevices>=1);

    int device = 0;
    magma_setdevice(device);

    magma_queue_create( devices[device], &magma_device_queue);

    int magmaRet;
    magmaRet = magma_dmalloc(&device_M, n*n);
    magmaRet = magma_dmalloc(&device_rhs, n );

  }
  virtual ~hiopLinSolverIndefDenseMagmaDev()
  {
    magma_free(device_M);
    magma_free(device_rhs);
    magma_queue_destroy(magma_device_queue);
    magma_device_queue = NULL;

    delete [] ipiv;
    delete dwork;
  }

  /** Triggers a refactorization of the matrix, if necessary. */
  int matrixChanged()
  {
    assert(M.n() == M.m());
    int N=M.n(), lda = N, info;
    if(N==0) return 0;

    nlp_->runStats.linsolv.tmFactTime.start();

    double gflops = FLOPS_DPOTRF( N )  / 1e9;

    magma_uplo_t uplo=MagmaLower; // M is upper in C++ so it's lower in fortran

    //
    //query sizes
    //
    magma_dsytrf(uplo, N, M.local_buffer(), lda, ipiv, &info );

    nlp_->runStats.tmFactTime.stop();
    nlp_->runStats.linsolv.flopsFact = gflops / nlp_->runStats.tmFactTime.getElapsedTime() / 1000.);

    if(info<0) {
      nlp->log->printf(hovError,
		       "hiopLinSolverMagma error: %d argument to dsytrf has an illegal value.\n",
		       -info);
      return -1;
    } else {
      if(info>0) {
	nlp->log->printf(hovWarning,
			 "hiopLinSolverMagma error: %d entry in the factorization's diagonal\n"
			 "is exactly zero. Division by zero will occur if it a solve is attempted.\n",
			 info);
	//matrix is singular
	return -1;
      }
    }
    assert(info==0);

    nlp_->runStats.linsolv.tmInertiaComp.start();
    //
    // Compute the inertia. Only negative eigenvalues are returned.
    // Code originally written by M. Schanenfor PIPS based on
    // LINPACK's dsidi Fortran routine (http://www.netlib.org/linpack/dsidi.f)
    // 04/08/2020 - petra: fixed the test for non-positive pivots (was only for negative pivots)
    int negEigVal=0;
    int posEigVal=0;
    int nullEigVal=0;
    double t=0;
    double** MM = M.get_M();
    for(int k=0; k<N; k++) {
      //c       2 by 2 block
      //c       use det (d  s)  =  (d/t * c - t) * t  ,  t = dabs(s)
      //c               (s  c)
      //c       to avoid underflow/overflow troubles.
      //c       take two passes through scaling.  use  t  for flag.
      double d = MM[k][k];
      if(ipiv[k] <= 0) {
	if(t==0) {
	  assert(k+1<N);
	  if(k+1<N) {
	    t=fabs(MM[k][k+1]);
	    d=(d/t) * MM[k+1][k+1]-t;
	  }
	} else {
	  d=t;
	  t=0.;
	}
      }
      //printf("d = %22.14e \n", d);
      //if(d<0) negEigVal++;
      if(d < -1e-14) {
	negEigVal++;
      } else if(d < 1e-14) {
	nullEigVal++;
	//break;
      } else {
	posEigVal++;
      }
    }
    //printf("(pos,null,neg)=(%d,%d,%d)\n", posEigVal, nullEigVal, negEigVal);
    nlp_->runStats.linsolv.tmInertiaComp.stop();
    
    if(nullEigVal>0) return -1;
    return negEigVal;
  }

  /** solves a linear system.
   * param 'x' is on entry the right hand side(s) of the system to be solved. On
   * exit is contains the solution(s).  */
  void solve ( hiopVector& x_ )
  {
    assert(M.n() == M.m());
    assert(x_.get_size() == M.n());
    int N = M.n();
    int LDA = N;
    int LDB = N;
    int NRHS = 1;
    if(N == 0) return;

    nlp_->runStats.linsolv.tmTriuSolves.start();
    
    hiopVectorPar* x = dynamic_cast<hiopVectorPar*>(&x_);
    assert(x != NULL);

    char uplo='L'; // M is upper in C++ so it's lower in fortran
    int info;
    DSYTRS(&uplo, &N, &NRHS, M.local_buffer(), &LDA, ipiv, x->local_data(), &LDB, &info);
    if(info<0) {
      nlp->log->printf(hovError, "hiopLinSolverMAGMA: (LAPACK) DSYTRS returned error %d\n", info);
      assert(false);
    } else if(info>0) {
      nlp->log->printf(hovError, "hiopLinSolverMAGMA: (LAPACK) DSYTRS returned error %d\n", info);
    }

    nlp_->runStats.linsolv.tmTriuSolves.stop();
    /*
    printf("Solve starts on a matrix %d x %d\n", N, N);

    magma_int_t info; 

    hiopVectorPar* x = dynamic_cast<hiopVectorPar*>(&x_);
    assert(x != NULL);
    
    magma_uplo_t uplo=MagmaLower; // M is upper in C++ so it's lower in fortran
    magma_int_t NRHS=1;

    const int align=32;
    magma_int_t LDDA=N;//magma_roundup( N, align );  // multiple of 32 by default
    magma_int_t LDDB=LDA;

    double gflops = ( FLOPS_DPOTRF( N ) + FLOPS_DPOTRS( N, NRHS ) ) / 1e9;

    hiopTimer t_glob; t_glob.start();
    hiopTimer t; t.start();
    magma_dsetmatrix( N, N,    M.local_buffer(), LDA, device_M,   LDDA, magma_device_queue );
    magma_dsetmatrix( N, NRHS, x->local_data(),  LDB, device_rhs, LDDB, magma_device_queue );
    t.stop();
    printf("cpu->gpu data transfer in %g sec\n", t.getElapsedTime());
    fflush(stdout);

    //DSYTRS(&uplo, &N, &NRHS, M.local_buffer(), &LDA, ipiv, x->local_data(), &LDB, &info);
    t.reset(); t.start();
    magma_dsysv_nopiv_gpu(uplo, N, NRHS, device_M, LDDA, device_rhs, LDDB, &info);
    t.stop();
    printf("gpu solve in %g sec  TFlops: %g\n", t.getElapsedTime(), gflops / t.getElapsedTime() / 1000.);

    if(0 != info) {
      printf("dsysv_nopiv returned %d [%s]\n", info, magma_strerror( info ));
    }
    assert(info==0);

    if(info<0) {
      nlp->log->printf(hovError, "hiopLinSolverIndefDenseMagma: DSYTRS returned error %d\n", info);
      assert(false);
    }
    t.reset(); t.start();
    magma_dgetmatrix( N, NRHS, device_rhs, LDDB, x->local_data(), LDDB, magma_device_queue );
    t.stop(); t_glob.stop();
    printf("gpu->cpu solution transfer in %g sec\n", t.getElapsedTime());
    printf("including tranfer time -> TFlops: %g\n", gflops / t_glob.getElapsedTime() / 1000.);
    */
  }
  void solve ( hiopMatrix& x ) { assert(false && "not needed; see the other solve method for implementation"); }

  hiopMatrixDense& sysMatrix() { return M; }
protected:
  int* ipiv;
  hiopVectorPar* dwork;

  magma_queue_t magma_device_queue;
  magmaDouble_ptr device_M, device_rhs;
private:
  hiopLinSolverIndefDenseMagmaDev() { assert(false); }
};


}

#endif //of HIOP_USE_MAGMA
#endif
