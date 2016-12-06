#ifndef HIOP_ITERATE
#define HIOP_ITERATE

#include "hiopVector.hpp"
#include "hiopNlpFormulation.hpp"

class hiopIterate
{
public:
  hiopIterate(const hiopNlpDenseConstraints* nlp);
  virtual ~hiopIterate();

  //virtual void projectPrimalsIntoBounds(double kappa1, double kappa2);
  virtual void projectPrimalsXIntoBounds(double kappa1, double kappa2);
  virtual void projectPrimalsDIntoBounds(double kappa1, double kappa2);
  virtual void setBoundsDualsToConstant(const double& v);
  virtual void setEqualityDualsToConstant(const double& v);
  /** computes the slacks given the primals: sxl=x-xl, sxu=xu-x, and similar 
   *  for sdl and sdu  */
  virtual void determineSlacks();

  /* max{a\in(0,1]| x+ad >=(1-tau)x} */
  bool fractionToTheBdry(const hiopIterate& dir, const double& tau, double& alphaprimal, double& alphadual) const;
  
  /* take the step: this = iter+alpha*dir */
  virtual bool takeStep_primals(const hiopIterate& iter, const hiopIterate& dir, double& alphaprimal, double& alphadual);
  virtual bool takeStep_duals(const hiopIterate& iter, const hiopIterate& dir, double& alphaprimal, double& alphadual);
  //virtual bool updateDualsEq(const hiopIterate& iter, const hiopIterate& dir, double& alphaprimal, double& alphadual);
  //virtual bool updateDualsIneq(const hiopIterate& iter, const hiopIterate& dir, double& alphaprimal, double& alphadual);
  
  /* Adjusts the signed duals to ensure the the logbar primal-dual Hessian is not arbitrarily 
   * far away from the primal counterpart. This is eq. 16 in the filter IPM paper */
  virtual bool adjustDuals_primalLogHessian(const double& mu, const double& kappa_Sigma);
  /* compute the log-barrier term for the primal signed variables */
  virtual double evalLogBarrier() const;

  /* computes the log barrier's linear damping term of the Filter-IPM method of WaectherBiegler (section 3.7) */
  virtual double linearDampingTerm(const double& mu, const double& kappa_d) const;
  /* adds the damping term to the gradient */
  virtual void addLinearDampingTermToGrad_x(const double& mu, const double& kappa_d, const double& beta, hiopVector& grad_x) const;
  virtual void addLinearDampingTermToGrad_d(const double& mu, const double& kappa_d, const double& beta, hiopVector& grad_d) const;

  /** norms for individual parts of the iterate (on demand computation) */
  virtual double normOneOfBoundDuals() const;
  virtual double normOneOfEqualityDuals() const;

  /* cloning and copying */
  hiopIterate* alloc_clone() const;
  //hiopIterate* new_copy() const;
  void copyFrom(const hiopIterate& src);

  /* accessors */
  inline hiopVector* get_x()   const {return x;}
  inline hiopVector* get_d()   const {return d;}
  inline hiopVector* get_sxl() const {return sxl;}

  void print(FILE* f, const char* msg=NULL) const;

  friend class hiopResidual;
  friend class hiopKKTLinSysLowRank;
private:
  /** Primal variables */
  hiopVectorPar*x;         //the original decision x
  hiopVectorPar*d;         //the adtl decisions d, d=d(x)
  hiopVectorPar*sxl,*sxu;  //slacks for x
  hiopVectorPar*sdl,*sdu;  //slacks for d

  /** Dual variables */
  hiopVectorPar*yc;       //for c(x)=crhs
  hiopVectorPar*yd;       //for d(x)-d=0
  hiopVectorPar*zl,*zu;   //for slacks eq. in x: x-sxl=xl, x+sxu=xu
  hiopVectorPar*vl,*vu;   //for slack eq. in d, e.g., d-sdl=dl
private:
  // and associated info from problem formulation
  const hiopNlpDenseConstraints * nlp;
private:
  hiopIterate() {};
  hiopIterate(const hiopIterate&) {};
  hiopIterate& operator=(const hiopIterate& o) {};
};


#endif
