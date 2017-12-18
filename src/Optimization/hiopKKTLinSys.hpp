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

#ifndef HIOP_KKTLINSYSY
#define HIOP_KKTLINSYSY

#include "hiopIterate.hpp"
#include "hiopResidual.hpp"
#include "hiopHessianLowRank.hpp"

namespace hiop
{

class hiopKKTLinSys 
{
public:
  /* updates the parts in KKT system that are dependent on the iterate. 
   * It may trigger a refactorization for direct linear systems, or it may not do 
   * anything, for example, LowRank linear system */
  virtual bool update(const hiopIterate* iter, 
		      const hiopVector* grad_f, 
		      const hiopMatrixDense* Jac_c, const hiopMatrixDense* Jac_d, 
		      hiopHessianLowRank* Hess)=0;
  virtual bool computeDirections(const hiopResidual* resid, hiopIterate* direction)=0;
  virtual ~hiopKKTLinSys() {}
};

class hiopKKTLinSysLowRank : public hiopKKTLinSys
{
public:
  hiopKKTLinSysLowRank(hiopNlpFormulation* nlp_);
  virtual ~hiopKKTLinSysLowRank();

  virtual bool update(const hiopIterate* iter, 
		      const hiopVector* grad_f, 
		      const hiopMatrixDense* Jac_c, const hiopMatrixDense* Jac_d, 
		      hiopHessianLowRank* Hess);
  virtual bool computeDirections(const hiopResidual* resid, hiopIterate* direction);

  /* Solves the system corresponding to directions for x, yc, and yd, namely
   * [ H_BFGS + Dx   Jc^T  Jd^T   ] [ dx]   [ rx_tilde ]
   * [    Jc          0     0     ] [dyc] = [   ryc    ]
   * [    Jd          0   -Dd^{-1}] [dyd]   [ ryd_tilde]
   */
  virtual void solveCompressed(hiopVectorPar& rx, hiopVectorPar& ryc, hiopVectorPar& ryd,
			       hiopVectorPar& dx, hiopVectorPar& dyc, hiopVectorPar& dyd);

  //int factorizeMat(hiopMatrixDense& M);
  //int solveWithFactors(hiopMatrixDense& M, hiopVectorPar& r);

  //LAPACK wrappers
  int solve(hiopMatrixDense& M, hiopVectorPar& rhs);
  int solveWithRefin(hiopMatrixDense& M, hiopVectorPar& rhs);
#ifdef DEEP_CHECKING
  static double solveError(const hiopMatrixDense& M,  const hiopVectorPar& x, hiopVectorPar& rhs);

  //computes the solve error for the KKT Linear system; used only for correctness checking
  double errorKKT(const hiopResidual* resid, const hiopIterate* sol);
  double errorCompressedLinsys(const hiopVectorPar& rx, const hiopVectorPar& ryc, const hiopVectorPar& ryd,
			       const hiopVectorPar& dx, const hiopVectorPar& dyc, const hiopVectorPar& dyd);
#endif
private:
  const hiopIterate* iter;
  const hiopVectorPar* grad_f;
  const hiopMatrixDense *Jac_c, *Jac_d;
  hiopHessianLowRank* Hess;

  hiopNlpDenseConstraints* nlp;

  hiopMatrixDense* N; //the kxk reduced matrix
#ifdef DEEP_CHECKING
  hiopMatrixDense* Nmat; //a copy of the above to compute the residual
#endif
  hiopVectorPar *Dx, *Dd_inv;
  //internal buffers
  hiopVectorPar *rx_tilde, *ryd_tilde;
  hiopMatrixDense* _kxn_mat; //!opt (work directly with the Jacobian)
  hiopVectorPar* _k_vec1;
};

};

#endif
