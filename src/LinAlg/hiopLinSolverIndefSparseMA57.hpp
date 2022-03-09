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

#ifndef HIOP_LINSOLVER_MA57
#define HIOP_LINSOLVER_MA57

#include "hiopLinSolver.hpp"
#include "hiopMatrixSparseTriplet.hpp"


/** implements the linear solver class using the HSL MA57 solver
 *
 * @ingroup LinearSolvers
 */

#ifndef FNAME
#ifndef __bg__
#define FNAME(f) f ## _
#else
#define FNAME(f) f
#endif
#endif

namespace hiop {

extern "C" {
  void FNAME(ma57id)( double cntl[],  int icntl[] );

  void FNAME(ma57ad)( int * n,        int * ne,       int irn[],
		int jcn[],      int * lkeep,    int keep[],
		int iwork[],    int icntl[],    int info[],
		double rinfo[] );

  void FNAME(ma57bd)( int * n,        int * ne,       double a[],
		double fact[],  int * lfact,    int ifact[],
		int * lifact,   int * lkeep,    int keep[],
		int ppos[],     int * icntl,    double cntl[],
		int info[],     double rinfo[] );
  void FNAME(ma57cd)( int * job,      int * n,        double fact[],
		int * lfact,    int ifact[],    int * lifact,
		int * nrhs,     double rhs[],   int * lrhs,
		double w[],     int * lw,       int iw1[],
		int icntl[],    int info[]);
  void FNAME(ma57dd)( int * job,      int * n,        int * ne,
		double a[],     int irn[],      int jcn[],
		double fact[],  int * lfact,    int ifact[],
		int * lifact,   double rhs[],   double x[],
		double resid[], double w[],     int iw[],
		int icntl[],    double cntl[],  int info[],
		double rinfo[] );
  void FNAME(ma57ed)( int * n,        int * ic,       int keep[],
		double fact[],  int * lfact,    double * newfac,
		int * lnew,     int  ifact[],   int * lifact,
		int newifc[],   int * linew,    int * info );
}


/** Wrapper for MA57 */
class hiopLinSolverIndefSparseMA57: public hiopLinSolverSymSparse
{
public:
  hiopLinSolverIndefSparseMA57(const int& n, const int& nnz, hiopNlpFormulation* nlp);
  virtual ~hiopLinSolverIndefSparseMA57();

  /** Triggers a refactorization of the matrix, if necessary.
   * Overload from base class. */
  int matrixChanged();

  /** solves a linear system.
   * param 'x' is on entry the right hand side(s) of the system to be solved. On
   * exit is contains the solution(s).  */
  bool solve ( hiopVector& x_ );

private:
  int     icntl_[20];
  int     info_[40];
  double  cntl_[5];
  double  rinfo_[20];

  int     n_;                         // dimension of the whole matrix
  int     nnz_;                       // number of nonzeros in the matrix

  /// row indexes used by the factorization
  int* irowM_;
  
  /// col indexes used by the factorization
  int* jcolM_;           
  // note: the values array is reused (from the sys matrix)

  int     lkeep_;                     // temporary storage
  int*    keep_;                      // temporary storage
  int     lifact_;                    // temporary storage
  int*    ifact_;                     // temporary storage
  int     lfact_;                     // temporary storage for the factorization process
  double* fact_;                      // storage for the factors
  double  ipessimism_;                // amounts by which to increase allocated factorization space
  double  rpessimism_;                // amounts by which to increase allocated factorization space

  int* iwork_;
  double* dwork_;

  /// Right-hand side working array 
  hiopVector* rhs_;

  /// Working array used for residual computation 
  hiopVector* resid_;
  
  /// parameters to control pivoting
  double pivot_tol_;
  double pivot_max_;
  bool pivot_changed_;

public:

  /** called the very first time a matrix is factored. Allocates space
   * for the factorization and performs ordering */
  virtual void firstCall();
//  virtual void diagonalChanged( int idiag, int extent );

  // increase pivot tolarence
  virtual bool increase_pivot_tol();

};

} // end namespace
#endif
