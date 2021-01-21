// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
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

/**
 * @file hiopDualsUpdater.hpp
 *
 * @author Nai-Yuan Chiang <chiang7@llnl.gov>,  LLNL
 * @author Cosmin G. Petra <petra1@llnl.gov>,  LLNL
 *
 */
 
#ifndef HIOP_DUALSUPDATER
#define HIOP_DUALSUPDATER

#include "hiopNlpFormulation.hpp"
#include "hiopIterate.hpp"
#include "hiopResidual.hpp"
#include "hiopMatrix.hpp"
#include "hiopLinSolver.hpp"

namespace hiop
{

class hiopDualsUpdater
{
public:
  hiopDualsUpdater(hiopNlpFormulation* nlp) : _nlp(nlp) {};
  virtual ~hiopDualsUpdater() {};

  /* The method is called after each iteration to update the duals. Implementations for different 
   * multiplier updating strategies are provided by child classes 
   * - linear (Newton) update in hiopDualsNewtonLinearUpdate
   * - lsq in hiopDualsLsqUpdate
   * The parameters are:
   * - iter: incumbent iterate that is going to be updated with iter_plus by the caller of this method.
   * - iter_plus: [in/out] on return the duals should be updated; primals are already updated, but 
   * the function should not rely on this. If a particular implementation of this method requires
   * accessing primals, it should do so by working with 'iter'. In the algorithm class, iter_plus 
   * corresponds to 'iter_trial'.
   * - f,c,d: fcn evals at  iter_plus 
   * - grad_f, jac_c, jac_d: derivatives at iter_plus
   * - search_dir: search direction (already used to update primals, potentially to be used to 
   * update duals (in linear update))
   * - alpha_primal: step taken for primals (also taken for eq. duals for the linear Newton duals update)
   * - alpha_dual: max step for the duals based on the fraction-to-the-boundary rule (not used
   * by lsq update)
   */
  virtual bool go(const hiopIterate& iter,  hiopIterate& iter_plus,
		  const double& f, const hiopVector& c, const hiopVector& d,
		  const hiopVector& grad_f, const hiopMatrix& jac_c, const hiopMatrix& jac_d,
		  const hiopIterate& search_dir, const double& alpha_primal, const double& alpha_dual,
		  const double& mu, const double& kappa_sigma, const double& infeas_nrm_trial)=0;
protected:
  hiopNlpFormulation* _nlp;
protected: 
  hiopDualsUpdater() {};
private:
  hiopDualsUpdater(const hiopDualsUpdater&) {};
  void operator=(const  hiopDualsUpdater&) {};
  
};

class hiopDualsLsqUpdate : public hiopDualsUpdater
{
public:
  hiopDualsLsqUpdate(hiopNlpFormulation* nlp);
  virtual ~hiopDualsLsqUpdate();

  /** LSQ update of the constraints duals (yc and yd). Source file describe the math. */
  virtual bool go(const hiopIterate& iter,  hiopIterate& iter_plus,
		  const double& f, const hiopVector& c, const hiopVector& d,
		  const hiopVector& grad_f, const hiopMatrix& jac_c, const hiopMatrix& jac_d,
		  const hiopIterate& search_dir, const double& alpha_primal, const double& alpha_dual,
		  const double& mu, const double& kappa_sigma, const double& infeas_nrm_trial);

  /** LSQ-based initialization of the  constraints duals (yc and yd). Source file describes the math. */
  virtual inline bool computeInitialDualsEq(hiopIterate& it_ini,
					    const hiopVector& grad_f,
					    const hiopMatrix& jac_c,
					    const hiopMatrix& jac_d)
  {
    if(augsys_update_)
    {
      _nlp->log->printf(hovSummary,
                        "LSQ Dual Initialization --- Sparse linsys: size %d (%d eq-cons)\n",
                        _nlp->m_eq()+_nlp->m_ineq(), _nlp->m_eq());  

      return LSQInitDualSparse(it_ini,grad_f,jac_c,jac_d);
    }
    assert(augsys_update_==false);
    
    
    _nlp->log->printf(hovSummary,
                      "LSQ Dual Initialization --- Dense linsys: size %d (%d eq-cons)\n",
                      _nlp->m_eq()+_nlp->m_ineq(), _nlp->m_eq());  

    return LSQUpdate(it_ini,grad_f,jac_c,jac_d);
  }
private: //common code 
  virtual bool LSQUpdate(hiopIterate& it,
                         const hiopVector& grad_f,
                         const hiopMatrix& jac_c,
                         const hiopMatrix& jac_d);

  /**
   * @brief LSQ-based initialization for sparse linear algebra.
   *
   * NLPs with sparse Jac/Hes
   * ******************************
   * For NLPs with sparse inputs, the corresponding LSQ problem is solved in augmeted system:
   * [    I    0     Jc^T  Jd^T  ] [ dx]      [ \nabla f(xk) - zk_l + zk_u  ]
   * [    0    I     0     -I    ] [ dd]      [ -vk_l + vk_u ]
   * [    Jc   0     0     0     ] [dyc] =  - [   0    ]
   * [    Jd   -I    0     0     ] [dyd]      [   0    ]         ]
   *
   * The matrix of the above system is stored in the member variable M_ of this class and the
   * right-hand side in 'rhs'.   *
   */
  virtual bool LSQInitDualSparse(hiopIterate& it,
			 const hiopVector& grad_f,
			 const hiopMatrix& jac_c,
			 const hiopMatrix& jac_d);

private:
  hiopMatrix *_mexme_, *_mexmi_, *_mixmi_, *_mxm_;
  hiopMatrix *M_;

  hiopVector *rhs_, *rhsc_, *rhsd_;
  hiopVector *_vec_n_, *_vec_mi_;

  hiopLinSolver* linSys_;
  double lsq_dual_init_max;

#ifdef HIOP_DEEPCHECKS
  hiopMatrix* M_copy_;
  hiopVector *rhs_copy_;
  hiopMatrix* _mixme_;
#endif

  //user options

  /** Do not recompute duals using LSQ unless the primal infeasibilty or constraint violation 
   * is less than this tolerance; default 1e-6
   */
  double recalc_lsq_duals_tol;
  
  int augsys_update_; //0:dense 1:mds 2:sparse
                                
  //helpers
  int factorizeMat(hiopMatrixDense& M);
  int solveWithFactors(hiopMatrixDense& M, hiopVector& r);
private: 
  hiopDualsLsqUpdate() {};
  hiopDualsLsqUpdate(const hiopDualsLsqUpdate&) {};
  void operator=(const  hiopDualsLsqUpdate&) {};
  
};

/** 
 * Performs Newton update for the duals, which is a simple linear update along the dual Newton direction
 * with a given dual step.
 */ 
class hiopDualsNewtonLinearUpdate : public hiopDualsUpdater
{
public:
  hiopDualsNewtonLinearUpdate(hiopNlpFormulation* nlp) : hiopDualsUpdater(nlp) {};
  virtual ~hiopDualsNewtonLinearUpdate() {};

  /* Linear update of step length alpha_primal in eq. duals yc and yd and step length
   * alpha_dual in the (signed or bounds) duals zl, zu, vl, and vu.
   * This is standard in (full) Newton IPMs. Very cheap!
   */
  virtual bool go(const hiopIterate& iter, hiopIterate& iter_plus,
		  const double& f, const hiopVector& c, const hiopVector& d,
		  const hiopVector& grad_f, const hiopMatrix& jac_c, const hiopMatrix& jac_d,
		  const hiopIterate& search_dir, const double& alpha_primal, const double& alpha_dual,
		  const double& mu, const double& kappa_sigma, const double& infeas_nrm_trial)
  { 
    if(!iter_plus.takeStep_duals(iter, search_dir, alpha_primal, alpha_dual)) {
      _nlp->log->printf(hovError, "dual Newton updater: error in standard update of the duals");
      return false;
    }
    return iter_plus.adjustDuals_primalLogHessian(mu,kappa_sigma);
  }

private: 
  hiopDualsNewtonLinearUpdate() {};
  hiopDualsNewtonLinearUpdate(const hiopDualsNewtonLinearUpdate&) {};
  void operator=(const  hiopDualsNewtonLinearUpdate&) {};
  
};

}
#endif
