#ifndef HIOP_DUALSUPDATER
#define HIOP_DUALSUPDATER

class hiopDualUpdater
{
public:
  hiopDualUpdater(hiopNlpFormulation* nlp) : _nlp(nlp) {};
  virtual ~hiopDualUpdater() {};

  /* The method is called after each iteration to update the duals. Implementations for different 
   * multiplier updating strategies are provided by child classes 
   * - linear (Newton) update in hiopDualsNewtonLinearUpdate
   * - lsq in hiopDualsLsqUpdate
   * The parameters are:
   * - iter_plus: [in/out] on return the duals should be updated; primals are already updated, but 
   * the function should not rely on this.
   * - iter: incumbent iterate that is going to be updated with iter_plus by the caller of this method.
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
		  const hiopIteration& search_dir, const double& alpha_primal, const double& alpha_dual)=0;
	  
private: 
  hiopDualUpdater() {};
  hiopDualUpdater(const hiopDualUpdater&) {};
  void operator=(const  hiopDualUpdater&) {};
  
};

class hiopDualsLsqUpdate : public hiopDualUpdater
{
public:
  hiopDualsLsqUpdate(hiopNlpFormulation* nlp) : hiopDualUpdater(nlp) {};
  virtual ~hiopDualsLsqUpdate() {};

  virtual bool go(const hiopIterate& iter,  hiopIterate& iter_plus,
		  const double& f, const hiopVector& c, const hiopVector& d,
		  const hiopVector& grad_f, const hiopMatrix& jac_c, const hiopMatrix& jac_d,
		  const hiopIteration& search_dir, const double& alpha_primal, const double& alpha_dual) { assert(false); }


private: 
  hiopDualsLsqUpdate() {};
  hiopDualsLsqUpdate(const hiopDualsLsqUpdate&) {};
  void operator=(const  hiopDualsLsqUpdate&) {};
  
};

class hiopDualsNewtonLinearUpdate : public hiopDualUpdater
{
public:
  hiopDualsNewtonLinearUpdate(hiopNlpFormulation* nlp) : hiopDualUpdater(nlp) {};
  virtual ~hiopDualsNewtonLinearUpdate() {};

  /* Linear update of step length alpha_primal in eq. duals yc and yd and step length
   * alpha_dual in the (signed or bounds) duals zl, zu, vl, and vu.
   * This is standard in (full) Newton IPMs. Very cheap!
   */
  virtual bool go(const hiopIterate& iter, hiopIterate& iter_plus,
		  const double& f, const hiopVector& c, const hiopVector& d,
		  const hiopVector& grad_f, const hiopMatrix& jac_c, const hiopMatrix& jac_d,
		  const hiopIteration& search_dir, const double& alpha_primal, const double& alpha_dual) { 
    return it_plus->takeStep_duals(iter, search_dir, alpha_primal, alpha_dual); 
  }


private: 
  hiopDualsNewtonLinearUpdate() {};
  hiopDualsNewtonLinearUpdate(const hiopDualsNewtonLinearUpdate&) {};
  void operator=(const  hiopDualsNewtonLinearUpdate&) {};
  
};


#endif
