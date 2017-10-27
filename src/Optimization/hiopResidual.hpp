#ifndef HIOP_RESIDUAL
#define HIOP_RESIDUAL

#include "hiopNlpFormulation.hpp"
#include "hiopVector.hpp"
#include "hiopIterate.hpp"

#include "hiopLogBarProblem.hpp"

namespace hiop
{

  class hiopResidual
  {
  public:
    hiopResidual(hiopNlpDenseConstraints* nlp);
    virtual ~hiopResidual();
    
    virtual int update(const hiopIterate& it, 
		       const double& f, const hiopVector& c, const hiopVector& d,
		       const hiopVector& gradf, const hiopMatrix& jac_c, const hiopMatrix& jac_d, 
		       const hiopLogBarProblem& logbar);
    
    /* Return the Nlp and Log-bar errors computed at the previous update call. */ 
    inline void getNlpErrors(double& optim, double& feas, double& comple) const
    { optim=nrm_nlp_optim; feas=nrm_nlp_feasib; comple=nrm_nlp_complem;};
    inline void getBarrierErrors(double& optim, double& feas, double& comple) const
    { optim=nrm_bar_optim; feas=nrm_bar_feasib; comple=nrm_bar_complem;};
    /* get the previously computed Infeasibility */
    inline double getInfeasNorm() const { 
      return nrm_nlp_feasib;
    }
    /* evaluate the Infeasibility at the new iterate, which has eq and ineq functions 
     * computed in c_eval and d_eval, respectively. 
     * The method modifies 'this', in particular ryd,ryc, rxl,rxu, rdl, rdu in an attempt
     * to reuse storage/buffers, but does not update the cached nrm_XXX members. */
    virtual double computeNlpInfeasInfNorm(const hiopIterate& iter, 
					   const hiopVector& c_eval, 
					   const hiopVector& d_eval);
    
    /* residual printing function - calls hiopVector::print 
     * prints up to max_elems (by default all), on rank 'rank' (by default on all) */
    virtual void print(FILE*, const char* msg=NULL, int max_elems=-1, int rank=-1) const;
    
  protected:
    hiopVectorPar*rx;           // -\grad f - J_c^t y_c - J_d^t y_d + RieszInv(z_l) - RieszInv(z_u)
    hiopVectorPar*rd;           //  y_d + v_l - v_u
    hiopVectorPar*rxl,*rxu;     //  x - sxl-xl, -x-sxu+xu
    hiopVectorPar*rdl,*rdu;     //  as above but for d
    
    hiopVectorPar*ryc;          // -c(x)   (c(x)=0!//!)
    hiopVectorPar*ryd;          //for d- d(x)
    
    hiopVectorPar*rszl,*rszu;   // \mu e-sxl zl, \mu e - sxu zu
    hiopVectorPar*rsvl,*rsvu;   // \mu e-sdl vl, \mu e - sdu vu
    
    /** storage for the norm of [rx,rd], [rxl,...,rdu,ryc,ryd], and [rszl,...,rsvu]  
     *  for the nlp (\mu=0)
     */
    double nrm_nlp_optim, nrm_nlp_feasib, nrm_nlp_complem; 
    /** storage for the norm of [rx,rd], [rxl,...,rdu,ryc,ryd], and [rszl,...,rsvu]  
     *  for the barrier subproblem
     */
    double nrm_bar_optim, nrm_bar_feasib, nrm_bar_complem; 
    // and associated info from problem formulation
    hiopNlpDenseConstraints * nlp;
  private:
    hiopResidual() {};
    hiopResidual(const hiopResidual&) {};
    hiopResidual& operator=(const hiopResidual& o) { assert(false); return *this;};
    friend class hiopKKTLinSysLowRank;
  };
  
  /** Derived class for backward compatibility (with Ipopt, using inf-norms) */
  class hiopResidualFinDimImpl : public hiopResidual
  {
  public:
    hiopResidualFinDimImpl(hiopNlpDenseConstraints* nlp) : hiopResidual(nlp) {};
    virtual ~hiopResidualFinDimImpl() {};

    virtual int update(const hiopIterate& it, 
		       const double& f, const hiopVector& c, const hiopVector& d,
		       const hiopVector& gradf, const hiopMatrix& jac_c, const hiopMatrix& jac_d, 
		       const hiopLogBarProblem& logbar);
    
    /* evaluate the Infeasibility at the new iterate, which has eq and ineq functions 
     * computed in c_eval and d_eval, respectively. 
     * The method modifies 'this', in particular ryd,ryc, rxl,rxu, rdl, rdu in an attempt
     * to reuse storage/buffers, but does not update the cached nrm_XXX members. */
    virtual double computeNlpInfeasInfNorm(const hiopIterate& iter, 
					   const hiopVector& c_eval, 
					   const hiopVector& d_eval);
  private:
    hiopResidualFinDimImpl & operator=(const hiopResidualFinDimImpl & o) { assert(false); return *this;};
  };
}
#endif
