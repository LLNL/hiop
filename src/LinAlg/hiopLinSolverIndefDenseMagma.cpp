#include "hiopLinSolverIndefDenseMagma.hpp"

namespace hiop
{
  hiopLinSolverIndefDenseMagmaBuKa::hiopLinSolverIndefDenseMagmaBuKa(int n, hiopNlpFormulation* nlp_)
    : hiopLinSolverIndefDense(n, nlp_)
  {
    ipiv = new int[n];

    //
    // commented code - this class is using CPU MAGMA interface for now
    //
    // magma_int_t ndevices;
    // magma_device_t devices[ MagmaMaxGPUs ];
    // magma_getdevices( devices, MagmaMaxGPUs, &ndevices );
    // assert(ndevices>=1);

    // int device = 0;
    // magma_setdevice(device);

    // magma_queue_create( devices[device], &magma_device_queue);

    // int magmaRet;
    // magmaRet = magma_dmalloc(&device_M, n*n);
    // magmaRet = magma_dmalloc(&device_rhs, n );
  }

  hiopLinSolverIndefDenseMagmaBuKa::~hiopLinSolverIndefDenseMagmaBuKa()
  {
    //
    // commented code - this class is using CPU MAGMA interface
    //
    // magma_free(device_M);
    // magma_free(device_rhs);
    // magma_queue_destroy(magma_device_queue);
    // magma_device_queue = NULL;

    delete [] ipiv;
  }

  /** Triggers a refactorization of the matrix, if necessary. */
  int hiopLinSolverIndefDenseMagmaBuKa::matrixChanged()
  {
    assert(M_->n() == M_->m());
    int N=M_->n(), lda = N, info;
    if(N==0) return 0;

    nlp_->runStats.linsolv.tmFactTime.start();

    double gflops = FLOPS_DPOTRF( N )  / 1e9;

    magma_uplo_t uplo=MagmaLower; // M is upper in C++ so it's lower in fortran

    //
    //query sizes
    //
    magma_dsytrf(uplo, N, M_->local_buffer(), lda, ipiv, &info );

    nlp_->runStats.linsolv.tmFactTime.stop();
    nlp_->runStats.linsolv.flopsFact = gflops / nlp_->runStats.linsolv.tmFactTime.getElapsedTime() / 1000.;

    if(info<0) {
      nlp_->log->printf(hovError,
		       "hiopLinSolverMagmaBuka error: %d argument to dsytrf has an illegal value.\n",
		       -info);
      return -1;
    } else {
      if(info>0) {
	nlp_->log->printf(hovWarning,
			 "hiopLinSolverMagmaBuka error: %d entry in the factorization's diagonal\n"
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
    double* MM = M_->local_buffer();
    for(int k=0; k<N; k++) {
      //c       2 by 2 block
      //c       use det (d  s)  =  (d/t * c - t) * t  ,  t = dabs(s)
      //c               (s  c)
      //c       to avoid underflow/overflow troubles.
      //c       take two passes through scaling.  use  t  for flag.
      double d = MM[N*k + k];
      if(ipiv[k] <= 0) {
        if(t==0) {
          assert(k+1<N);
          if(k+1<N) {
            t=fabs(MM[N*k + (k+1)]);
            d=(d/t) * MM[N*(k+1) + (k+1)]-t;
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
    // std::cout << "Using Magma factorization ...\n";
    // std::cout << "Matrix inertia =(" << posEigVal << ", " << nullEigVal << ", " << negEigVal << ")\n";
    //printf("(pos,null,neg)=(%d,%d,%d)\n", posEigVal, nullEigVal, negEigVal);
    nlp_->runStats.linsolv.tmInertiaComp.stop();
    
    if(nullEigVal>0) return -1;
    return negEigVal;
  }

  bool hiopLinSolverIndefDenseMagmaBuKa::solve(hiopVector& x)
  {
    assert(M_->n() == M_->m());
    assert(x.get_size() == M_->n());
    int N = M_->n();
    int LDA = N;
    int LDB = N;
    int NRHS = 1;
    if(N == 0) return true;

    nlp_->runStats.linsolv.tmTriuSolves.start();
    
    char uplo='L'; // M is upper in C++ so it's lower in fortran
    int info;
    DSYTRS(&uplo, &N, &NRHS, M_->local_buffer(), &LDA, ipiv, x.local_data(), &LDB, &info);
    if(info<0) {
      nlp_->log->printf(hovError, "hiopLinSolverMagmaBuKa: (LAPACK) DSYTRS returned error %d\n", info);
      assert(false);
    } else if(info>0) {
      nlp_->log->printf(hovError, "hiopLinSolverMagmaBuKa: (LAPACK) DSYTRS returned warning %d\n", info);
    }
    nlp_->runStats.linsolv.tmTriuSolves.stop();

    return info==0;
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

  /*******************************************************************************************************
   * MAGMA indefinite solver without pivoting (fast)
   *
   *******************************************************************************************************/
  hiopLinSolverIndefDenseMagmaNopiv::hiopLinSolverIndefDenseMagmaNopiv(int n, hiopNlpFormulation* nlp)
  : hiopLinSolverIndefDense(n, nlp) 
  {
    magma_int_t ndevices;
    magma_device_t devices[MagmaMaxGPUs];
    magma_getdevices(devices, MagmaMaxGPUs, &ndevices);
    assert(ndevices>=1);

    int device = 0;
    magma_setdevice(device);

    magma_queue_create(devices[device], &magma_device_queue_);

    int magmaRet;

    const int align=32;
    const int nrhs=1;
    ldda_ = magma_roundup(n, align );  // multiple of 32 by default
    lddb_ = ldda_;


    std::string mem_space = nlp_->options->GetString("mem_space");
    if(mem_space == "default" || mem_space == "host")
    {
      magmaRet = magma_dmalloc(&device_M_, n*ldda_);
      assert(MAGMA_SUCCESS == magmaRet);
      magmaRet = magma_dmalloc(&device_rhs_, nrhs*lddb_ );
      assert(MAGMA_SUCCESS == magmaRet);
    }
    else
    {
      device_M_   = nullptr;
      device_rhs_ = nullptr;
    }

    nFakeNegEigs_ = 0;
  }

  hiopLinSolverIndefDenseMagmaNopiv::~hiopLinSolverIndefDenseMagmaNopiv()
  {
    std::string mem_space = nlp_->options->GetString("mem_space");
    if(mem_space == "default" || mem_space == "host")
    {
      magma_free(device_M_);
      magma_free(device_rhs_);
    }
    magma_queue_destroy(magma_device_queue_);
    magma_device_queue_ = NULL;
  }

  /** Triggers a refactorization of the matrix, if necessary. */
  int hiopLinSolverIndefDenseMagmaNopiv::matrixChanged()
  {
    nlp_->runStats.linsolv.tmFactTime.start();
    //TODO: split solve (factorization done in 'solve' now)
    int negEigVal = nFakeNegEigs_;
    nlp_->runStats.linsolv.tmFactTime.stop();
    return negEigVal;
  }

  bool hiopLinSolverIndefDenseMagmaNopiv::solve( hiopVector& x )
  {
    assert(M_->n() == M_->m());
    assert(x.get_size()==M_->n());
    int N=M_->n(), LDA = N, LDB=N;
    if(N==0) return true;

    magma_int_t info; 

    magma_uplo_t uplo=MagmaLower; // M is upper in C++ so it's lower in fortran
    magma_int_t NRHS=1;

    double gflops = ( FLOPS_DPOTRF( N ) + FLOPS_DPOTRS( N, NRHS ) ) / 1e9;

    std::string mem_space = nlp_->options->GetString("mem_space");
    if(mem_space == "default" || mem_space == "host")
    {
      nlp_->runStats.linsolv.tmDeviceTransfer.start();
      magma_dsetmatrix(N, N,    M_->local_buffer(), LDA, device_M_,   ldda_, magma_device_queue_);
      magma_dsetmatrix(N, NRHS, x.local_data(),   LDB, device_rhs_, lddb_, magma_device_queue_);
      nlp_->runStats.linsolv.tmDeviceTransfer.stop();
    }
    else
    {
      device_M_   = M_->local_buffer();
      device_rhs_ = x.local_data();
    }
    
    nlp_->runStats.linsolv.tmTriuSolves.start();
    //DSYTRS(&uplo, &N, &NRHS, M.local_buffer(), &LDA, ipiv, x->local_data(), &LDB, &info);

    assert(device_M_);
    assert(device_rhs_);
    //
    // the call
    //
    magma_dsysv_nopiv_gpu(uplo, N, NRHS, device_M_, ldda_, device_rhs_, lddb_, &info);

    //if(0 != info) {
    //  printf("dsysv_nopiv returned %d [%s]\n", info, magma_strerror(info));
    //}
    //assert(info==0);

    nlp_->runStats.linsolv.tmTriuSolves.stop();
    nlp_->runStats.linsolv.flopsTriuSolves = 
      gflops / nlp_->runStats.linsolv.tmTriuSolves.getElapsedTime()/1000.;
    
    if(info != 0) {
      nlp_->log->printf(hovError, 
			"hiopLinSolverMagmaNopiv: dsysv_nopiv returned error %d [%s]\n", 
			info, magma_strerror(info));
      return false;
    }

    if(mem_space == "default" || mem_space == "host")
    {
      nlp_->runStats.linsolv.tmDeviceTransfer.start();
      magma_dgetmatrix(N, NRHS, device_rhs_, lddb_, x.local_data(), LDB, magma_device_queue_);
      nlp_->runStats.linsolv.tmDeviceTransfer.stop();
    }

    return true;
  }

} // end namespace
