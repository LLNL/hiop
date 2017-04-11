#ifndef HIOP_DUALSUPDATER
#define HIOP_DUALSUPDATER

#include "hiopNlpFormulation.hpp"
#include "hiopIterate.hpp"
#include "hiopResidual.hpp"
#include "hiopMatrix.hpp"

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

  /** LSQ-based initialization of the  constraints duals (yc and yd). Source file describe the math. */
  virtual inline bool computeInitialDualsEq(hiopIterate& it_ini, const hiopVector& grad_f, const hiopMatrix& jac_c, const hiopMatrix& jac_d)
  {
    return  LSQUpdate(it_ini,grad_f,jac_c,jac_d);
  }
private: //common code 
  virtual bool LSQUpdate(hiopIterate& it, const hiopVector& grad_f, const hiopMatrix& jac_c, const hiopMatrix& jac_d);
private:
  hiopMatrixDense *_mexme, *_mexmi, *_mixmi, *_mxm;
  hiopMatrixDense *M;
  
  hiopVectorPar *rhs, *rhsc, *rhsd;
  hiopVectorPar *_vec_n, *_vec_mi;

#ifdef DEEP_CHECKING
  hiopMatrixDense* M_copy;
  hiopVectorPar *rhs_copy;
  hiopMatrixDense* _mixme;
#endif

  //user options
  double recalc_lsq_duals_tol;  //do not recompute duals using LSQ unless the primal infeasibilty or constraint violation 
                                //is less than this tolerance; default 1e-6

  //helpers
  int factorizeMat(hiopMatrixDense& M);
  int solveWithFactors(hiopMatrixDense& M, hiopVectorPar& r);
private: 
  hiopDualsLsqUpdate() {};
  hiopDualsLsqUpdate(const hiopDualsLsqUpdate&) {};
  void operator=(const  hiopDualsLsqUpdate&) {};
  
};

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
		  const double& mu, const double& kappa_sigma, const double& infeas_nrm_trial) { 
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

};
#endif
