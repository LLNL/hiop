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

#include "blasdefs.hpp"

namespace hiop
{

/**
 * Abstract class for Linear Solvers used by HiOp
 * Specifies interface for linear solver arising in Interior-Point methods, thus,
 * the underlying assumptions are that the system's matrix is symmetric positive
 * definite or symmetric indefinite.
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


/** Wrapper for LAPACK's DSYTRF */
class hiopLinSolverIndefDense : public hiopLinSolver
{
public:
  hiopLinSolverIndefDense(int n, hiopNlpFormulation* nlp_)
    : M(n,n)
  {
    nlp = nlp_;
    ipiv = new int[n];
    dwork = new hiopVectorPar(0);
  }
  virtual ~hiopLinSolverIndefDense()
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
    // Code originally written by M. Schanen, ANL for PIPS based on
    // LINPACK's dsidi Fortran routine (http://www.netlib.org/linpack/dsidi.f)
    //
    int negEigVal=0;
    double t=0;
    double** MM = M.get_M();
    for(int k=0; k<N; k++) {
      double d = MM[k][k];
      if(ipiv[k] < 0) {
	if(t==0) {
	  t=fabs(MM[k+1][k]);
	  d=(d/t) * MM[k+1][k+1]-t;
	} else {
	  d=t;
	  t=0;
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
      nlp->log->printf(hovError, "hiopLinSolverIndefDense: DSYTRS returned error %d\n", info);
      assert(false);
    }
    
  }
  void solve ( hiopMatrix& x ) { assert(false && "not needed; see the other solve method for implementation"); }

  hiopMatrixDense& sysMatrix() { return M; }
protected:
  hiopMatrixDense M;
  int* ipiv;
  hiopVectorPar* dwork;
private:
  hiopLinSolverIndefDense() : M(0,0), ipiv(NULL), dwork(NULL) { assert(false); }
};

} //end namespace

#endif

