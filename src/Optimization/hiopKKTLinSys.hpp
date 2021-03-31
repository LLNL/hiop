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
#include "hiopPDPerturbation.hpp"
#include "hiopLinSolver.hpp"
#include "hiopFactAcceptor.hpp"

#include "hiopCppStdUtils.hpp"

namespace hiop
{

class hiopKKTLinSys
{
public:
  hiopKKTLinSys(hiopNlpFormulation* nlp)
    : nlp_(nlp), iter_(NULL), grad_f_(NULL), Jac_c_(NULL), Jac_d_(NULL), Hess_(NULL),
      perturb_calc_(NULL), safe_mode_(true)
  {
    perf_report_ = "on"==hiop::tolower(nlp_->options->GetString("time_kkt"));
  }
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

  virtual void set_PD_perturb_calc(hiopPDPerturbation* p)
  {
    perturb_calc_ = p;
  }

  virtual void set_fact_acceptor(hiopFactAcceptor* p_fact_acceptor)
  {
    fact_acceptor_ = p_fact_acceptor;
  }  
  
  inline void set_safe_mode(bool val)
  {
    safe_mode_ = val;
  }
#ifdef HIOP_DEEPCHECKS
  //computes the solve error for the KKT Linear system; used only for correctness checking
  virtual double errorKKT(const hiopResidual* resid, const hiopIterate* sol);

protected:
  /** 
   * @brief y=beta*y+alpha*H*x
   * 
   * @pre Should not include log barrier diagonal terms
   * @pre Should not include IC perturbations
   *
   * A default implementation is below
   */
  virtual void HessianTimesVec_noLogBarrierTerm(double beta, hiopVector& y,
						double alpha, const hiopVector&x)
  {
    Hess_->timesVec(beta, y, alpha, x);
  }


#endif
protected:
  hiopNlpFormulation* nlp_;
  const hiopIterate* iter_;
  const hiopVector* grad_f_;
  const hiopMatrix *Jac_c_, *Jac_d_;
  hiopMatrix* Hess_;
  hiopPDPerturbation* perturb_calc_;
  hiopFactAcceptor* fact_acceptor_;  
  bool perf_report_;
  bool safe_mode_;
};

class hiopKKTLinSysCurvCheck : public hiopKKTLinSys
{
public:
  hiopKKTLinSysCurvCheck(hiopNlpFormulation* nlp)
    : hiopKKTLinSys(nlp), linSys_{nullptr}
  {}

  virtual ~hiopKKTLinSysCurvCheck()
  {if(linSys_) delete linSys_;}

  virtual bool update(const hiopIterate* iter,
                    const hiopVector* grad_f,
                    const hiopMatrix* Jac_c, const hiopMatrix* Jac_d, hiopMatrix* Hess) = 0;

  virtual bool computeDirections(const hiopResidual* resid, hiopIterate* direction) = 0;

  virtual bool factorize();
  
  /**
   * @brief factorize the matrix and check curvature
   */ 
  virtual int factorizeWithCurvCheck();

  /** 
   * @brief updates the iterate matrix, given regularizations 'delta_wx', 'delta_wd', 'delta_cc' and 'delta_cd'.
   */
  virtual bool updateMatrix(const double& delta_wx, const double& delta_wd,
                            const double& delta_cc, const double& delta_cd) = 0;

  hiopLinSolver* linSys_;

};


class hiopKKTLinSysCompressed : public hiopKKTLinSysCurvCheck
{
public:
  hiopKKTLinSysCompressed(hiopNlpFormulation* nlp)
    : hiopKKTLinSysCurvCheck(nlp), Dx_(NULL), rx_tilde_(NULL)
  {
    Dx_ = nlp->alloc_primal_vec();
    assert(Dx_ != NULL);
    rx_tilde_  = Dx_->alloc_clone();
  }
  virtual ~hiopKKTLinSysCompressed()
  {
    delete Dx_;
    delete rx_tilde_;
  }
  virtual bool update(const hiopIterate* iter,
		      const hiopVector* grad_f,
		      const hiopMatrix* Jac_c, const hiopMatrix* Jac_d, hiopMatrix* Hess) = 0;

  virtual bool computeDirections(const hiopResidual* resid, hiopIterate* direction) = 0;

  virtual bool updateMatrix(const double& delta_wx, const double& delta_wd,
                            const double& delta_cc, const double& delta_cd) = 0;
protected:
  hiopVector* Dx_;
  hiopVector* rx_tilde_;
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
  hiopKKTLinSysCompressedXYcYd(hiopNlpFormulation* nlp);
  virtual ~hiopKKTLinSysCompressedXYcYd();

  virtual bool update(const hiopIterate* iter, 
                      const hiopVector* grad_f, 
                      const hiopMatrix* Jac_c, const hiopMatrix* Jac_d, 
                      hiopMatrix* Hess);


  virtual bool computeDirections(const hiopResidual* resid, hiopIterate* direction);

  virtual bool updateMatrix(const double& delta_wx, const double& delta_wd,
                            const double& delta_cc, const double& delta_cd) = 0;

  virtual bool solveCompressed(hiopVector& rx, hiopVector& ryc, hiopVector& ryd,
                               hiopVector& dx, hiopVector& dyc, hiopVector& dyd) = 0;

#ifdef HIOP_DEEPCHECKS
  virtual double errorCompressedLinsys(const hiopVector& rx,
				       const hiopVector& ryc,
				       const hiopVector& ryd,
				       const hiopVector& dx,
				       const hiopVector& dyc,
				       const hiopVector& dyd);
#endif

protected:
  hiopVector *Dd_inv_;
  hiopVector *ryd_tilde_;
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
  hiopKKTLinSysCompressedXDYcYd(hiopNlpFormulation* nlp);
  virtual ~hiopKKTLinSysCompressedXDYcYd();

  virtual bool update(const hiopIterate* iter, 
                      const hiopVector* grad_f, 
                      const hiopMatrix* Jac_c, const hiopMatrix* Jac_d, hiopMatrix* Hess);

  virtual bool computeDirections(const hiopResidual* resid, hiopIterate* direction);

  virtual bool updateMatrix(const double& delta_wx, const double& delta_wd,
                            const double& delta_cc, const double& delta_cd) = 0;

  virtual bool solveCompressed(hiopVector& rx, hiopVector& rd, 
                               hiopVector& ryc, hiopVector& ryd,
                               hiopVector& dx, hiopVector& dd,
                               hiopVector& dyc, hiopVector& dyd) = 0;

#ifdef HIOP_DEEPCHECKS
  virtual double errorCompressedLinsys(const hiopVector& rx,  const hiopVector& rd,
                                       const hiopVector& ryc, const hiopVector& ryd,
                                       const hiopVector& dx,  const hiopVector& dd,
                                       const hiopVector& dyc, const hiopVector& dyd);
#endif

protected:
  hiopVector *Dd_;
  hiopVector *rd_tilde_;
protected:
#ifdef HIOP_DEEPCHECKS
  //y=beta*y+alpha*H*x
  virtual void HessianTimesVec_noLogBarrierTerm(double beta, hiopVector& y,
						double alpha, const hiopVector&x)
  {
    Hess_->timesVec(beta, y, alpha, x);
  }
#endif
};

class hiopKKTLinSysLowRank : public hiopKKTLinSysCompressedXYcYd
{
public:
  hiopKKTLinSysLowRank(hiopNlpFormulation* nlp);
  virtual ~hiopKKTLinSysLowRank();

  bool update(const hiopIterate* iter,
	      const hiopVector* grad_f,
	      const hiopMatrix* Jac_c, const hiopMatrix* Jac_d,
	      hiopMatrix* Hess)
  {
    const hiopMatrixDense* Jac_c_ = dynamic_cast<const hiopMatrixDense*>(Jac_c);
    const hiopMatrixDense* Jac_d_ = dynamic_cast<const hiopMatrixDense*>(Jac_d);
    hiopHessianLowRank* Hess_ = dynamic_cast<hiopHessianLowRank*>(Hess);
    if(Jac_c_==NULL || Jac_d_==NULL || Hess_==NULL) {
      assert(false);
      return false;
    }
    return update(iter, grad_f_, Jac_c_, Jac_d_, Hess_);
  }

  virtual bool update(const hiopIterate* iter,
		      const hiopVector* grad_f,
		      const hiopMatrixDense* Jac_c, const hiopMatrixDense* Jac_d,
		      hiopHessianLowRank* Hess);

  virtual bool updateMatrix(const double& delta_wx, const double& delta_wd,
                            const double& delta_cc, const double& delta_cd) {assert(false && "not yet implemented");return false;}

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
  virtual bool solveCompressed(hiopVector& rx, hiopVector& ryc, hiopVector& ryd,
                               hiopVector& dx, hiopVector& dyc, hiopVector& dyd);

  //LAPACK wrappers
  int solve(hiopMatrixDense& M, hiopVector& rhs);
  int solveWithRefin(hiopMatrixDense& M, hiopVector& rhs);
#ifdef HIOP_DEEPCHECKS
  static double solveError(const hiopMatrixDense& M,  const hiopVector& x, hiopVector& rhs);
  double errorCompressedLinsys(const hiopVector& rx, const hiopVector& ryc, const hiopVector& ryd,
			       const hiopVector& dx, const hiopVector& dyc, const hiopVector& dyd);
protected:
  //y=beta*y+alpha*H*x
  void HessianTimesVec_noLogBarrierTerm(double beta, hiopVector& y, double alpha, const hiopVector& x)
  {
    hiopHessianLowRank* HessLowR = dynamic_cast<hiopHessianLowRank*>(Hess_);
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
  hiopVector* _k_vec1;
};


/*
 * Solves KKTLinSysCompressedXYcYd by exploiting the sparse structure
 *
 * In general, the so-called XYcYd system has the form
 * [   H   Jc^T  Jd^T | 0 |  0   0  -I   I   |  0   0   0   0  ] [  dx]   [    rx    ]
 * [  Jc    0     0   | 0 |  0   0   0   0   |  0   0   0   0  ] [ dyc] = [   ryc    ]
 * [  Jd    0     0   |-I |  0   0   0   0   |  0   0   0   0  ] [ dyd]   [   ryd    ]
 * -----------------------------------------------------------------------------------
 * [  0     0    -I   | 0 |  -I  I   0   0   |  0   0   0   0  ] [  dd]   [    rd    ]
 * -----------------------------------------------------------------------------------
 * [  0     0     0   |-I |  0   0   0   0   |  I   0   0   0  ] [ dvl]   [   rdl    ]
 * [  0     0     0   | I |  0   0   0   0   |  0   I   0   0  ] [ dvu]   [   rdu    ]
 * [ -I     0     0   | 0 |  0   0   0   0   |  0   0   I   0  ] [ dzl]   [   rxl    ]
 * [  I     0     0   | 0 |  0   0   0   0   |  0   0   0   I  ] [ dzu]   [   rxu    ]
 * -----------------------------------------------------------------------------------
 * [  0     0     0   | 0 | Sl^d 0   0   0   | Vl   0   0   0  ] [dsdl]   [  rsvl    ]
 * [  0     0     0   | 0 |  0  Su^d 0   0   |  0  Vu   0   0  ] [dsdu]   [  rsvu    ]
 * [  0     0     0   | 0 |  0   0  Sl^x 0   |  0   0  Zl   0  ] [dsxl]   [  rszl    ]
 * [  0     0     0   | 0 |  0   0   0  Su^x |  0   0   0  Zu  ] [dsxu]   [  rszu    ]
 * where
 *  - Jc and Jd present the sparse Jacobians for equalities and inequalities
 *  - H is a sparse Hessian matrix
 *
 */
class hiopKKTLinSysFull: public hiopKKTLinSysCurvCheck
{
public:
  hiopKKTLinSysFull(hiopNlpFormulation* nlp)
    : hiopKKTLinSysCurvCheck(nlp)
  {}

  virtual ~hiopKKTLinSysFull()
  {}

  virtual bool update(const hiopIterate* iter,
                      const hiopVector* grad_f,
                      const hiopMatrix* Jac_c, const hiopMatrix* Jac_d, hiopMatrix* Hess);

  virtual bool computeDirections(const hiopResidual* resid, hiopIterate* direction);

  virtual bool updateMatrix(const double& delta_wx, const double& delta_wd,
                            const double& delta_cc, const double& delta_cd) = 0;  
  
  virtual bool solve( hiopVector& rx, hiopVector& ryc, hiopVector& ryd, hiopVector& rd,
                      hiopVector& rdl, hiopVector& rdu, hiopVector& rxl, hiopVector& rxu,
                      hiopVector& rsvl, hiopVector& rsvu, hiopVector& rszl, hiopVector& rszu,
                      hiopVector& dx, hiopVector& dyc, hiopVector& dyd, hiopVector& dd,
                      hiopVector& dvl, hiopVector& dvu, hiopVector& dzl, hiopVector& dzu,
                      hiopVector& dsdl, hiopVector& dsdu, hiopVector& dsxl, hiopVector& dsxu)=0;
protected:

};

};

#endif
