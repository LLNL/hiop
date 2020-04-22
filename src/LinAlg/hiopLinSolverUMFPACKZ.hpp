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

#ifndef HIOP_LINSOLVER_UMFPACKZ
#define HIOP_LINSOLVER_UMFPACKZ

#include "umfpack.h"

#include "hiopNlpFormulation.hpp"
#include "hiopLinSolver.hpp"
#include "hiopMatrixComplexSparseTriplet.hpp"
#include "hiopMatrixComplexDense.hpp"

namespace hiop
{
  class hiopLinSolverUMFPACKZ : public hiopLinSolver
  {
  public:
    hiopLinSolverUMFPACKZ(hiopMatrixComplexSparseTriplet& sysmat, hiopNlpFormulation* nlp_=NULL);
    virtual ~hiopLinSolverUMFPACKZ();
    
    /** Triggers a refactorization of the matrix, if necessary. 
     * Returns -1 if trouble in factorization is encountered. */
    virtual int matrixChanged();
    
    /** solves a linear system.
     * param 'x' is on entry the right hand side(s) of the system to be solved. On
     * exit is contains the solution(s).  */
    virtual void solve(hiopVector& x);
    virtual void solve(hiopMatrix& X);
    virtual void solve(const hiopMatrixComplexSparseTriplet& B, hiopMatrixComplexDense& X);

  private: 
    void* m_symbolic;
    void* m_numeric;
    double* m_null;

    hiopNlpFormulation* nlp;
    
    int *m_colptr, *m_rowidx;
    double *m_vals; //size 2*nnz !!!
    const hiopMatrixComplexSparseTriplet& sys_mat;
    int n, nnz;

    double m_control [UMFPACK_CONTROL], m_info [UMFPACK_INFO];

  private:
    //returns the "abs" norm of the residual A*x-b
    double resid_abs_norm(int n, int* Ap, int* Ai, double* Ax/*packed*/,
			  double* x/*packed*/,
			  double* b/*packed*/); 
  };
} //end namespace hiop

#endif

