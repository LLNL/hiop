#ifndef HIOP_RESIDUAL
#define HIOP_RESIDUAL

#include "hiopNlpFormulation.hpp"
#include "hiopVector.hpp"
#include "hiopIterate.hpp"

#include "hiopLogBarProblem.hpp"

class hiopResidual
{
public:
  hiopResidual(const hiopNlpDenseConstraints* nlp);
  virtual ~hiopResidual();

  virtual int update(const hiopIterate& it, 
		     const double& f, const hiopVector& c, const hiopVector& d,
		     const hiopVector& gradf, const hiopMatrix& jac_c, const hiopMatrix& jac_d, 
		     const hiopLogBarProblem& logbar);

  /** Return the Nlp and Log-bar errors computed at the previous update call. */ 
  inline void getNlpErrors(double& optim, double& feas, double& comple) const
  { optim=nrmInf_nlp_optim; feas=nrmInf_nlp_feasib; comple=nrmInf_nlp_complem;};
  inline void getBarrierErrors(double& optim, double& feas, double& comple) const
  { optim=nrmInf_bar_optim; feas=nrmInf_bar_feasib; comple=nrmInf_bar_complem;};

  inline double getNlpInfeasInfNorm() const { return nrmInf_nlp_feasib;}
  /* cloning and copying */
  //hiopResidual* alloc_clone() const;
  //hiopResidual* new_copy() const;

  /* residual printing function - calls hiopVector::print 
   * prints up to max_elems (by default all), on rank 'rank' (by default on all) */
  virtual void print(FILE*, const char* msg=NULL, int max_elems=-1, int rank=-1) const;
private:
  hiopVectorPar*rx;           // -\grad f - J_c^t y_c - J_d^t y_d + z_l - z_u
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
  double nrmInf_nlp_optim, nrmInf_nlp_feasib, nrmInf_nlp_complem; 
  /** storage for the norm of [rx,rd], [rxl,...,rdu,ryc,ryd], and [rszl,...,rsvu]  
   *  for the barrier subproblem
   */
  double nrmInf_bar_optim, nrmInf_bar_feasib, nrmInf_bar_complem; 
  // and associated info from problem formulation
  const hiopNlpDenseConstraints * nlp;
private:
  hiopResidual() {};
  hiopResidual(const hiopResidual&) {};
  hiopResidual& operator=(const hiopResidual& o) {};
  friend class hiopKKTLinSysLowRank;
};


#endif
