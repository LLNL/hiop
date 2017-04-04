#ifndef HIOP_DUALSUPDATER
#define HIOP_DUALSUPDATER

#include "hiopNlpFormulation.hpp"
#include "hiopIterate.hpp"
#include "hiopResidual.hpp"

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
		  const hiopIterate& search_dir, const double& alpha_primal, const double& alpha_dual)=0;
private:
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
  hiopDualsLsqUpdate(hiopNlpFormulation* nlp) : hiopDualsUpdater(nlp) {};
  virtual ~hiopDualsLsqUpdate() {};

  virtual bool go(const hiopIterate& iter,  hiopIterate& iter_plus,
		  const double& f, const hiopVector& c, const hiopVector& d,
		  const hiopVector& grad_f, const hiopMatrix& jac_c, const hiopMatrix& jac_d,
		  const hiopIterate& search_dir, const double& alpha_primal, const double& alpha_dual) { assert(false); }


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
		  const hiopIterate& search_dir, const double& alpha_primal, const double& alpha_dual) { 
    return iter_plus.takeStep_duals(iter, search_dir, alpha_primal, alpha_dual); 
  }


private: 
  hiopDualsNewtonLinearUpdate() {};
  hiopDualsNewtonLinearUpdate(const hiopDualsNewtonLinearUpdate&) {};
  void operator=(const  hiopDualsNewtonLinearUpdate&) {};
  
};

};
#endif
