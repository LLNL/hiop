// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// Written by Cosmin G. Petra, petra1@llnl.gov.
// LLNL-CODE-742473. All rights reserved.
//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop. HiOp 
// is released under the BSD 3-clause license (https://opensource.org/licenses/BSD-3-Clause). 
// Please also read “Additional BSD Notice” below.
//
// Redistribution and use in source and binary forms, with or without modification, 
// are permitted provided that the following conditions are met:
// i. Redistributions of source code must retain the above copyright notice, this list 
// of conditions and the disclaimer below.
// ii. Redistributions in binary form must reproduce the above copyright notice, 
// this list of conditions and the disclaimer (as noted below) in the documentation and/or 
// other materials provided with the distribution.
// iii. Neither the name of the LLNS/LLNL nor the names of its contributors may be used to 
// endorse or promote products derived from this software without specific prior written 
// permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
// SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR 
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS 
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED 
// AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Additional BSD Notice
// 1. This notice is required to be provided under our contract with the U.S. Department 
// of Energy (DOE). This work was produced at Lawrence Livermore National Laboratory under 
// Contract No. DE-AC52-07NA27344 with the DOE.
// 2. Neither the United States Government nor Lawrence Livermore National Security, LLC 
// nor any of their employees, makes any warranty, express or implied, or assumes any 
// liability or responsibility for the accuracy, completeness, or usefulness of any 
// information, apparatus, product, or process disclosed, or represents that its use would
// not infringe privately-owned rights.
// 3. Also, reference herein to any specific commercial products, process, or services by 
// trade name, trademark, manufacturer or otherwise does not necessarily constitute or 
// imply its endorsement, recommendation, or favoring by the United States Government or 
// Lawrence Livermore National Security, LLC. The views and opinions of authors expressed 
// herein do not necessarily state or reflect those of the United States Government or 
// Lawrence Livermore National Security, LLC, and shall not be used for advertising or 
// product endorsement purposes.

#ifndef HIOP_LINSOLVER
#define HIOP_LINSOLVER

#include "hiopMatrix.hpp"
#include "hiopVector.hpp"

#include "hiop_blasdefs.hpp"

namespace hiop
{

/**
 * Abstract class for Linear Solvers used by HiOp
 * Specifies interface for linear solver arising in Interior-Point methods, thus,
 * the underlying assumptions are that the system's matrix is symmetric (positive
 * definite or indefinite).
 *
 * Implementations of this abstract class have the purpose of serving as wrappers
 * of existing CPU and GPU libraries for linear systems. 
 */

class hiopLinSolver
{
public:
  hiopLinSolver() : nlp(NULL) {}
  virtual ~hiopLinSolver() {}

  /** Triggers a refactorization of the matrix, if necessary. 
   * Returns number of negative eigenvalues or -1 if null eigenvalues 
   * are encountered. */
  virtual int matrixChanged() = 0;

  /** solves a linear system.
   * param 'x' is on entry the right hand side(s) of the system to be solved. On
   * exit is contains the solution(s).  */
  virtual void solve ( hiopVector& x ) = 0;
  virtual void solve ( hiopMatrix& x ) { assert(false && "not yet supported"); }
public: 
  hiopNlpFormulation* nlp;
};

  /** Base class for Indefinite Dense Solvers */
class hiopLinSolverIndefDense : public hiopLinSolver
{
public:
  hiopLinSolverIndefDense(int n, hiopNlpFormulation* nlp_)
    : M(n,n)
  {
    nlp = nlp_;
  }
  virtual ~hiopLinSolverIndefDense()
  { 
  }

  hiopMatrixDense& sysMatrix() { return M; }
protected:
  hiopMatrixDense M;
protected:
  hiopLinSolverIndefDense() : M(0,0) { assert(false); }
};

/** Wrapper for LAPACK's DSYTRF */
class hiopLinSolverIndefDenseLapack : public hiopLinSolverIndefDense
{
public:
  hiopLinSolverIndefDenseLapack(int n, hiopNlpFormulation* nlp_)
    : hiopLinSolverIndefDense(n, nlp_)
  {
    ipiv = new int[n];
    dwork = new hiopVectorPar(0);
  }
  virtual ~hiopLinSolverIndefDenseLapack()
  {
    delete [] ipiv;
    delete dwork;
  }

  /** Triggers a refactorization of the matrix, if necessary. */
  int matrixChanged()
  {
    assert(M.n() == M.m());
    int N=M.n(), lda = N, info;
    if(N==0) return 0;

    double dwork_tmp;
    char uplo='L'; // M is upper in C++ so it's lower in fortran

    //
    //query sizes
    //
    int lwork=-1;
    DSYTRF(&uplo, &N, M.local_buffer(), &lda, ipiv, &dwork_tmp, &lwork, &info );
    assert(info==0);

    lwork=(int)dwork_tmp;
    if(lwork != dwork->get_size()) {
      delete dwork;
      dwork = NULL;
      dwork = new hiopVectorPar(lwork);
    }

    //
    // factorization
    //
    DSYTRF(&uplo, &N, M.local_buffer(), &lda, ipiv, dwork->local_data(), &lwork, &info );
    if(info<0)
      nlp->log->printf(hovError, "hiopLinSolverIndefDense error: %d argument to dsytrf has an"
		       " illegal value\n", -info);
    else if(info>0)
      nlp->log->printf(hovError, "hiopLinSolverIndefDense error: %d entry in the factorization's "
		       "diagonal is exactly zero. Division by zero will occur if it a solve is attempted.\n", info);
    assert(info==0);

    //
    // Compute the inertia. Only negative eigenvalues are returned.
    // Code originally written by M. Schanenfor PIPS based on
    // LINPACK's dsidi Fortran routine (http://www.netlib.org/linpack/dsidi.f)
    // 04/08/2020 - petra: fixed the test for non-positive pivots (was only for negative pivots)
    int negEigVal=0;
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
      if(d<0) negEigVal++;
      if(d==0) {
	negEigVal=-1;
	break;
      }
    }
    return negEigVal;
  }
    
  /** solves a linear system.
   * param 'x' is on entry the right hand side(s) of the system to be solved. On
   * exit is contains the solution(s).  */
  void solve ( hiopVector& x_ )
  {
    assert(M.n() == M.m());
    assert(x_.get_size()==M.n());
    int N=M.n(), LDA = N, info;
    if(N==0) return;

    hiopVectorPar* x = dynamic_cast<hiopVectorPar*>(&x_);
    assert(x != NULL);

    char uplo='L'; // M is upper in C++ so it's lower in fortran
    int NRHS=1, LDB=N;
    DSYTRS(&uplo, &N, &NRHS, M.local_buffer(), &LDA, ipiv, x->local_data(), &LDB, &info);
    if(info<0) {
      nlp->log->printf(hovError, "hiopLinSolverIndefDenseLapack: DSYTRS returned error %d\n", info);
      assert(false);
    }
    
  }
  void solve ( hiopMatrix& x ) { assert(false && "not needed; see the other solve method for implementation"); }

protected:
  int* ipiv;
  hiopVectorPar* dwork;
private:
  hiopLinSolverIndefDenseLapack() : ipiv(NULL), dwork(NULL) { assert(false); }
};

#ifdef HIOP_USE_MAGMA
#include "magma_v2.h"

#define FADDS_POTRF(n_) ((n_) * (((1. / 6.) * (n_)      ) * (n_) - (1. / 6.)))
#define FADDS_POTRI(n_) ( (n_) * ((1. / 6.) + (n_) * ((1. / 3.) * (n_) - 0.5)) )
#define FADDS_POTRS(n_, nrhs_) ((nrhs_) * (n_) * ((n_) - 1 ))
#define FMULS_POTRF(n_) ((n_) * (((1. / 6.) * (n_) + 0.5) * (n_) + (1. / 3.)))
#define FMULS_POTRI(n_) ( (n_) * ((2. / 3.) + (n_) * ((1. / 3.) * (n_) + 1. )) )
#define FMULS_POTRS(n_, nrhs_) ((nrhs_) * (n_) * ((n_) + 1 ))
#define FLOPS_DPOTRF(n_) (     FMULS_POTRF((double)(n_)) +       FADDS_POTRF((double)(n_)) )
#define FLOPS_DPOTRI(n_) (     FMULS_POTRI((double)(n_)) +       FADDS_POTRI((double)(n_)) )
#define FLOPS_DPOTRS(n_, nrhs_) (     FMULS_POTRS((double)(n_), (double)(nrhs_)) +       FADDS_POTRS((double)(n_), (double)(nrhs_)) )

class hiopLinSolverIndefDenseMagma : public hiopLinSolverIndefDense
{
public:
  hiopLinSolverIndefDenseMagma(int n, hiopNlpFormulation* nlp_)
    : hiopLinSolverIndefDense(n, nlp_)
  {

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
  virtual ~hiopLinSolverIndefDenseMagma()
  {
    magma_free(device_M);
    magma_free(device_rhs);
    magma_queue_destroy(magma_device_queue);
    magma_device_queue = NULL;
  }

  /** Triggers a refactorization of the matrix, if necessary. */
  int matrixChanged()
  {
    //TO DO: factorization done in 'solve' for now
    int negEigVal=0;
    return negEigVal;
  }
    



  /** solves a linear system.
   * param 'x' is on entry the right hand side(s) of the system to be solved. On
   * exit is contains the solution(s).  */
  void solve ( hiopVector& x_ )
  {
    assert(M.n() == M.m());
    assert(x_.get_size()==M.n());
    int N=M.n(), LDA = N, LDB=N;
    if(N==0) return;

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
  }
  void solve ( hiopMatrix& x ) { assert(false && "not needed; see the other solve method for implementation"); }

  hiopMatrixDense& sysMatrix() { return M; }
protected:
  magma_queue_t magma_device_queue;
  magmaDouble_ptr device_M, device_rhs;
private:
  hiopLinSolverIndefDenseMagma() { assert(false); }
};
#endif//def HIOP_USE_MAGMA
} //end namespace

#endif

