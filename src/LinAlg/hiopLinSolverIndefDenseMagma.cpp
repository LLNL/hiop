#include "hiopLinSolverIndefDenseMagma.hpp"

namespace hiop
{
  hiopLinSolverIndefDenseMagmaBuKa::hiopLinSolverIndefDenseMagmaBuKa(int n, hiopNlpFormulation* nlp_)
    : hiopLinSolverIndefDense(n, nlp_), work_(NULL)
  {
    ipiv_ = new int[n];
  }

  hiopLinSolverIndefDenseMagmaBuKa::~hiopLinSolverIndefDenseMagmaBuKa()
  {
    delete [] work_;
    delete [] ipiv_;
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
    magma_dsytrf(uplo, N, M_->local_data(), lda, ipiv_, &info );

    nlp_->runStats.linsolv.tmFactTime.stop();
    nlp_->runStats.linsolv.flopsFact = gflops / nlp_->runStats.linsolv.tmFactTime.getElapsedTime() / 1000.;

    if(info<0) {
      nlp_->log->printf(hovError,
		       "hiopLinSolverMagmaBuka error: %d argument to dsytrf has an illegal value.\n",
		       -info);
      return -1;
    } else {
      if(info>0) {
	nlp_->log->printf(hovScalars,
			 "hiopLinSolverMagmaBuka error: %d entry in the factorization's diagonal\n"
			 "is exactly zero. Division by zero will occur if it a solve is attempted.\n",
			 info);
	//matrix is singular
	return -1;
      }
    }
    assert(info==0);
    int negEigVal, posEigVal, nullEigVal;

    if(!compute_inertia(N, ipiv_, posEigVal, negEigVal, nullEigVal)) {
      return -1;
    }

    if(nullEigVal>0) return -1;
    return negEigVal;
  }
  
  bool hiopLinSolverIndefDenseMagmaBuKa_old::compute_inertia(int N, 
                                                             int *ipiv,
                                                             int& posEigVal, 
                                                             int& negEigVal, 
                                                             int& nullEigVal)
  {
    negEigVal=0;
    posEigVal=0;
    nullEigVal=0;

    nlp_->runStats.linsolv.tmInertiaComp.start();
    //
    // Compute the inertia. Only negative eigenvalues are returned.
    // Code originally written by M. Schanen for PIPS based on
    // LINPACK's dsidi Fortran routine (http://www.netlib.org/linpack/dsidi.f)
    // 04/08/2020 - petra: fixed the test for non-positive pivots (was only for negative pivots)
    double t=0;

    double* MM = M_->local_data();

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
    //printf("MagmaBuKa Eigs (pos,null,neg)=(%d,%d,%d)\n", posEigVal, nullEigVal, negEigVal);
    nlp_->runStats.linsolv.tmInertiaComp.stop();

    return true;
  }

  bool hiopLinSolverIndefDenseMagmaBuKa::compute_inertia(int N, 
                                                         int *ipiv,
                                                         int& posEigVal, 
                                                         int& negEigVal, 
                                                         int& nullEigVal)
  {
    double det[2];
    int inert[3];
    int job = 100;
    int info;

    nlp_->runStats.linsolv.tmInertiaComp.start();

    if(NULL==work_) {
      work_ = new double[N];
    }

    //determinant if only for logging/output/curiosity purposes
    magma_dsidi(M_->local_data(), N, N, ipiv, det, inert, work_, job, &info);
    if(0!=info) {
      nlp_->log->printf(hovError, 
                        "Magma dsidi inertia computation failed with [%d] (MagmaBuKa)\n",
                        info);
      posEigVal = negEigVal = nullEigVal = -1;
      return false;
    }

    posEigVal = inert[0];
    negEigVal = inert[1];
    nullEigVal = inert[2];
    
    //printf("SIDI ON : det = %g\n", det[0] * std::pow(10, det[1]));
    nlp_->log->printf(hovScalars, "BuKa dsidi eigs: pos/neg/null  %d %d %d \n", 
                      posEigVal,negEigVal,nullEigVal);
    //fflush(stdout);
    nlp_->runStats.linsolv.tmInertiaComp.stop();
    return info==0;
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
    DSYTRS(&uplo, &N, &NRHS, M_->local_data(), &LDA, ipiv_, x.local_data(), &LDB, &info);

    if(info<0) {
      nlp_->log->printf(hovError, "hiopLinSolverMagmaBuKa: (LAPACK) DSYTRS returned error %d\n", info);
      assert(false);
    } else if(info>0) {
      nlp_->log->printf(hovError, "hiopLinSolverMagmaBuKa: (LAPACK) DSYTRS returned warning %d\n", info);
    }
    nlp_->runStats.linsolv.tmTriuSolves.stop();

    return info==0;
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
      ldda_ = n;
      lddb_ = ldda_;
    }
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
    assert(M_->n() == M_->m());
    int N=M_->n(), LDA = N, LDB=N;
    if(N==0) return true;

    magma_int_t info; 

    magma_uplo_t uplo=MagmaLower; // M is upper in C++ so it's lower in fortran

    double gflops = FLOPS_DPOTRF( N ) / 1e9;

    std::string mem_space = nlp_->options->GetString("mem_space");
    if(mem_space == "default" || mem_space == "host")
    {
      nlp_->runStats.linsolv.tmDeviceTransfer.start();
      magma_dsetmatrix(N, N,    M_->local_data(), LDA, device_M_,  ldda_, magma_device_queue_);
      nlp_->runStats.linsolv.tmDeviceTransfer.stop();
    }
    else
    {
      device_M_   = M_->local_data();
    }
    nlp_->runStats.linsolv.tmFactTime.start();

    //
    // Factorization of the matrix
    //
    magma_dsytrf_nopiv_gpu(uplo, N, device_M_, ldda_, &info);

    nlp_->runStats.linsolv.tmFactTime.stop();
    nlp_->runStats.linsolv.flopsFact = gflops / nlp_->runStats.linsolv.tmFactTime.getElapsedTime() / 1000.;

    if(info<0) {
      nlp_->log->printf(hovError,
		       "hiopLinSolverIndefDenseNoPiv error: %d argument to dsytrf has an illegal value.\n",
		       -info);
      return -1;
    } else {
      if(info>0) {
	nlp_->log->printf(hovScalars,
			 "hiopLinSolverIndefDenseNoPiv error: %d entry in the factorization's diagonal\n"
			 "is exactly zero. Division by zero will occur if it a solve is attempted.\n",
			 info);
	//matrix is singular
	return -1;
      }
    }
    assert(info==0);
    int negEigVal, posEigVal, nullEigVal;
    
    if(!compute_inertia(N, posEigVal, negEigVal, nullEigVal)) {
      return -1;
    }

    if(nullEigVal>0) return -1;
    return negEigVal;
  }

  bool hiopLinSolverIndefDenseMagmaNopiv::compute_inertia(int n, 
                                                          int& posEigvals, 
                                                          int& negEigvals, 
                                                          int& zeroEigvals)
  {
    assert(device_M_);

    //
    // inertia
    //
    int info;
    int *dinert, inert[3];
    if(MAGMA_SUCCESS != magma_malloc((void**)&dinert, 3*sizeof(int))) {
      nlp_->log->printf(hovError, 
			"hiopLinSolverMagmaNopiv: error in allocating memory on the device "
                        "(MAGMA_ERR_INVALID_PTR).\n");
      return false;
      
    }

    info = magmablas_ddiinertia(n, device_M_, ldda_, dinert, magma_device_queue_);
    if(MAGMA_SUCCESS != info) {
      nlp_->log->printf(hovError, 
			"hiopLinSolverMagmaNopiv: magmablas_ddiinertia returned error %d [%s]\n", 
			info, magma_strerror(info));
      posEigvals = negEigvals = zeroEigvals = -1;
      return false;
    }
    magma_getvector( 3, sizeof(int), dinert, 1, inert, 1, magma_device_queue_);
    magma_free( dinert );

    posEigvals  = inert[0];
    negEigvals  = inert[1];
    zeroEigvals = inert[2];

    nlp_->log->printf(hovScalars, 
                      "inertia: positive / negative / zero = %d / %d / %d\n",
                      inert[0], inert[1], inert[2]);

    return true;
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

    double gflops = FLOPS_DPOTRS(N, NRHS) / 1e9;

    std::string mem_space = nlp_->options->GetString("mem_space");
    if(mem_space == "default" || mem_space == "host")
    {
      nlp_->runStats.linsolv.tmDeviceTransfer.start();
      magma_dsetmatrix(N, NRHS, x.local_data(),   LDB, device_rhs_, lddb_, magma_device_queue_);
      nlp_->runStats.linsolv.tmDeviceTransfer.stop();
    }
    else
    {
      device_rhs_ = x.local_data();
    }
    
    nlp_->runStats.linsolv.tmTriuSolves.start();

    assert(device_M_);
    assert(device_rhs_);
    //
    // the call
    //
    //magma_dsysv_nopiv_gpu(uplo, N, NRHS, device_M_, ldda_, device_rhs_, lddb_, &info);
    magma_dsytrs_nopiv_gpu(uplo, N, NRHS, device_M_, ldda_, device_rhs_, lddb_, &info);

    nlp_->runStats.linsolv.tmTriuSolves.stop();
    nlp_->runStats.linsolv.flopsTriuSolves = 
      gflops / nlp_->runStats.linsolv.tmTriuSolves.getElapsedTime()/1000.;
    
    if(info != 0) {
      nlp_->log->printf(hovError, 
			"hiopLinSolverMagmaNopiv: dsytrs_nopiv_gpu returned error %d [%s]\n", 
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
