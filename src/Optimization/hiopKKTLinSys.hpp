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
  hiopKKTLinSys(hiopNlpFormulation* nlp_) 
    : nlp(nlp_), iter(NULL), grad_f(NULL), Jac_c(NULL), Jac_d(NULL), Hess(NULL)
  { }
  virtual ~hiopKKTLinSys() 
  { }
  /* updates the parts in KKT system that are dependent on the iterate. 
   * It may trigger a refactorization for direct linear systems, or it may not do 
   * anything, for example, LowRank linear system */
  virtual bool update(const hiopIterate* iter, 
		      const hiopVector* grad_f, 
		      const hiopMatrix* Jac_c, const hiopMatrix* Jac_d, hiopMatrix* Hess) = 0;

  /* forms the residual of the underlying linear system, uses the factorization
   * computed by 'update' to compute the "reduced-space" search directions by solving
   * with the factors, then computes the "full-space" directions */
  virtual bool computeDirections(const hiopResidual* resid, hiopIterate* direction) = 0;

#ifdef HIOP_DEEPCHECKS
  //computes the solve error for the KKT Linear system; used only for correctness checking
  virtual double errorKKT(const hiopResidual* resid, const hiopIterate* sol);

protected:
  //y=beta*y+alpha*H*x
  virtual void HessianTimesVec_noLogBarrierTerm(double beta, hiopVector& y, 
						double alpha, const hiopVector&x)
  {
    Hess->timesVec(beta, y, alpha, x);
  }
#endif
protected:
  hiopNlpFormulation* nlp;
  const hiopIterate* iter;
  const hiopVectorPar* grad_f;
  const hiopMatrix *Jac_c, *Jac_d;
  hiopMatrix* Hess;
};

class hiopKKTLinSysCompressed : public hiopKKTLinSys
{
public:
  hiopKKTLinSysCompressed(hiopNlpFormulation* nlp_)
    : hiopKKTLinSys(nlp_), Dx(NULL), rx_tilde(NULL)
  {
    Dx = dynamic_cast<hiopVectorPar*>(nlp->alloc_primal_vec());
    assert(Dx != NULL);
    rx_tilde  = Dx->alloc_clone(); 
  }
  virtual ~hiopKKTLinSysCompressed() 
  {
    delete Dx;
    delete rx_tilde;
  }
  virtual bool update(const hiopIterate* iter, 
		      const hiopVector* grad_f, 
		      const hiopMatrix* Jac_c, const hiopMatrix* Jac_d, hiopMatrix* Hess) = 0;

  virtual bool computeDirections(const hiopResidual* resid, hiopIterate* direction) = 0;

protected:
  hiopVectorPar *Dx;
  hiopVectorPar *rx_tilde;
};

/* Provides the functionality for reducing the KKT linear system to the 
 * compressed linear below in dx, dyc, and dyd variables and then to perform 
 * the basic ops needed to compute the remaining directions. 
 *
 * Relies on the pure virtual 'solveCompressed' to solve the compressed linear system
 * [  H  +  Dx     Jc^T  Jd^T   ] [ dx]   [ rx_tilde ]
 * [    Jc          0     0     ] [dyc] = [   ryc    ]
 * [    Jd          0   -Dd^{-1}] [dyd]   [ ryd_tilde]
 */
class hiopKKTLinSysCompressedXYcYd : public hiopKKTLinSysCompressed
{
public:
  hiopKKTLinSysCompressedXYcYd(hiopNlpFormulation* nlp_);
  virtual ~hiopKKTLinSysCompressedXYcYd();

  virtual bool update(const hiopIterate* iter, 
		      const hiopVector* grad_f, 
		      const hiopMatrix* Jac_c, const hiopMatrix* Jac_d, hiopMatrix* Hess) = 0;


  virtual bool computeDirections(const hiopResidual* resid, hiopIterate* direction);

  virtual void solveCompressed(hiopVectorPar& rx, hiopVectorPar& ryc, hiopVectorPar& ryd,
			       hiopVectorPar& dx, hiopVectorPar& dyc, hiopVectorPar& dyd) = 0;

#ifdef HIOP_DEEPCHECKS
  virtual double errorCompressedLinsys(const hiopVectorPar& rx, 
				       const hiopVectorPar& ryc, 
				       const hiopVectorPar& ryd,
				       const hiopVectorPar& dx, 
				       const hiopVectorPar& dyc, 
				       const hiopVectorPar& dyd);
#endif

protected:
  hiopVectorPar *Dd_inv;
  hiopVectorPar *ryd_tilde;
};

/* Provides the functionality for reducing the KKT linear system to the 
 * compressed linear below in dx, dd, dyc, and dyd variables and then to perform 
 * the basic ops needed to compute the remaining directions. 
 *
 * Relies on the pure virtual 'solveCompressed' to form and solve the compressed 
 * linear system
 * [  H  +  Dx    0    Jc^T  Jd^T   ] [ dx]   [ rx_tilde ]
 * [    0         Dd    0     -I    ] [ dd]   [ rd_tilde ]
 * [    Jc        0     0      0    ] [dyc] = [   ryc    ]
 * [    Jd       -I     0      0    ] [dyd]   [   ryd    ]   
 * and then to compute the rest of the search directions
 */
class hiopKKTLinSysCompressedXDYcYd : public hiopKKTLinSysCompressed
{
public:
  hiopKKTLinSysCompressedXDYcYd(hiopNlpFormulation* nlp_);
  virtual ~hiopKKTLinSysCompressedXDYcYd();

  virtual bool update(const hiopIterate* iter, 
		      const hiopVector* grad_f, 
		      const hiopMatrix* Jac_c, const hiopMatrix* Jac_d, hiopMatrix* Hess) = 0;

  virtual bool computeDirections(const hiopResidual* resid, hiopIterate* direction);

  virtual void solveCompressed(hiopVectorPar& rx, hiopVectorPar& rd, 
			       hiopVectorPar& ryc, hiopVectorPar& ryd,
			       hiopVectorPar& dx, hiopVectorPar& dd, 
			       hiopVectorPar& dyc, hiopVectorPar& dyd) = 0;

#ifdef HIOP_DEEPCHECKS
  virtual double errorCompressedLinsys(const hiopVectorPar& rx,  const hiopVectorPar& rd, 
				       const hiopVectorPar& ryc, const hiopVectorPar& ryd,
				       const hiopVectorPar& dx,  const hiopVectorPar& dd, 
				       const hiopVectorPar& dyc, const hiopVectorPar& dyd);
#endif

protected:
  hiopVectorPar *Dd;
  hiopVectorPar *rd_tilde;
protected: 
#ifdef HIOP_DEEPCHECKS
  //y=beta*y+alpha*H*x
  virtual void HessianTimesVec_noLogBarrierTerm(double beta, hiopVector& y, 
						double alpha, const hiopVector&x)
  {
    Hess->timesVec(beta, y, alpha, x);
  }
#endif
};

class hiopKKTLinSysLowRank : public hiopKKTLinSysCompressedXYcYd
{
public:
  hiopKKTLinSysLowRank(hiopNlpFormulation* nlp_);
  virtual ~hiopKKTLinSysLowRank();

  bool update(const hiopIterate* iter, 
	      const hiopVector* grad_f, 
	      const hiopMatrix* Jac_c_, const hiopMatrix* Jac_d_, 
	      hiopMatrix* Hess_)
  {
    const hiopMatrixDense* Jac_c = dynamic_cast<const hiopMatrixDense*>(Jac_c_);
    const hiopMatrixDense* Jac_d = dynamic_cast<const hiopMatrixDense*>(Jac_d_);
    hiopHessianLowRank* Hess = dynamic_cast<hiopHessianLowRank*>(Hess_);
    if(Jac_c==NULL || Jac_d==NULL || Hess==NULL) {
      assert(false);
      return false;
    }
    return update(iter, grad_f, Jac_c, Jac_d, Hess);
  }

  virtual bool update(const hiopIterate* iter, 
		      const hiopVector* grad_f, 
		      const hiopMatrixDense* Jac_c, const hiopMatrixDense* Jac_d, 
		      hiopHessianLowRank* Hess);

  /* Solves the system corresponding to directions for x, yc, and yd, namely
   * [ H_BFGS + Dx   Jc^T  Jd^T   ] [ dx]   [ rx_tilde ]
   * [    Jc          0     0     ] [dyc] = [   ryc    ]
   * [    Jd          0   -Dd^{-1}] [dyd]   [ ryd_tilde]
   *
   * This is done by forming and solving
   * [ Jc*(H+Dx)^{-1}*Jc^T   Jc*(H+Dx)^{-1}*Jd^T          ] [dyc] = [ Jc(H+Dx)^{-1} rx - ryc ]
   * [ Jd*(H+Dx)^{-1}*Jc^T   Jd*(H+Dx)^{-1}*Jd^T + Dd^{-1}] [dyd]   [ Jd(H+dx)^{-1} rx - ryd ]
   * and then solving for dx from
   *  dx = - (H+Dx)^{-1}*(Jc^T*dyc+Jd^T*dyd - rx)
   * 
   */
  virtual void solveCompressed(hiopVectorPar& rx, hiopVectorPar& ryc, hiopVectorPar& ryd,
			       hiopVectorPar& dx, hiopVectorPar& dyc, hiopVectorPar& dyd);

  //int factorizeMat(hiopMatrixDense& M);
  //int solveWithFactors(hiopMatrixDense& M, hiopVectorPar& r);

  //LAPACK wrappers
  int solve(hiopMatrixDense& M, hiopVectorPar& rhs);
  int solveWithRefin(hiopMatrixDense& M, hiopVectorPar& rhs);
#ifdef HIOP_DEEPCHECKS
  static double solveError(const hiopMatrixDense& M,  const hiopVectorPar& x, hiopVectorPar& rhs);
  double errorCompressedLinsys(const hiopVectorPar& rx, const hiopVectorPar& ryc, const hiopVectorPar& ryd,
			       const hiopVectorPar& dx, const hiopVectorPar& dyc, const hiopVectorPar& dyd);
protected:
  //y=beta*y+alpha*H*x
  void HessianTimesVec_noLogBarrierTerm(double beta, hiopVector& y, double alpha, const hiopVector& x)
  {
    hiopHessianLowRank* HessLowR = dynamic_cast<hiopHessianLowRank*>(Hess);
    assert(NULL != HessLowR);
    if(HessLowR) HessLowR->timesVec_noLogBarrierTerm(beta, y, alpha, x);
  }
#endif

private:
  hiopNlpDenseConstraints* nlpD;
  hiopHessianLowRank* HessLowRank;

  hiopMatrixDense* N; //the kxk reduced matrix
#ifdef HIOP_DEEPCHECKS
  hiopMatrixDense* Nmat; //a copy of the above to compute the residual
#endif
  //internal buffers
  hiopMatrixDense* _kxn_mat; //!opt (work directly with the Jacobian)
  hiopVectorPar* _k_vec1;
};

};

#endif
