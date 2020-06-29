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

#ifndef HIOP_ITERATE
#define HIOP_ITERATE

#include "hiopVector.hpp"
#include "hiopNlpFormulation.hpp"

namespace hiop
{

class hiopIterate
{
public:
  hiopIterate(const hiopNlpFormulation* nlp);
  virtual ~hiopIterate();

  virtual void projectPrimalsXIntoBounds(double kappa1, double kappa2);
  virtual void projectPrimalsDIntoBounds(double kappa1, double kappa2);
  virtual void setBoundsDualsToConstant(const double& v);
  virtual void setEqualityDualsToConstant(const double& v);
  /** 
   * Computes the slacks given the primals: sxl=x-xl, sxu=xu-x, and similar 
   *  for sdl and sdu.
   */
  virtual void determineSlacks();

  /** 
   * Computes duals for bounds on d, namely vl and vu from vl=mu e/sdl and vu = mu e/sdu. 
   * Assumes sdl and sdu are available/computed previously.
   */
  virtual void determineDualsBounds_d(const double& mu);
  
  /* max{a\in(0,1]| x+ad >=(1-tau)x} */
  bool fractionToTheBdry(const hiopIterate& dir, const double& tau,
			 double& alphaprimal, double& alphadual) const;
  
  /* take the step: this = iter+alpha*dir */
  virtual bool takeStep_primals(const hiopIterate& iter, const hiopIterate& dir,
				const double& alphaprimal, const double& alphadual);
  virtual bool takeStep_duals(const hiopIterate& iter, const hiopIterate& dir,
			      const double& alphaprimal, const double& alphadual);
  
  /**
   * Adjusts the signed duals to ensure the the logbar primal-dual Hessian is not arbitrarily 
   * far away from the primal counterpart. This is eq. 16 in the filter IPM paper 
   */
  virtual bool adjustDuals_primalLogHessian(const double& mu, const double& kappa_Sigma);
  /* compute the log-barrier term for the primal signed variables */
  virtual double evalLogBarrier() const;
  /* add the derivative of the log-barier terms*/
  virtual void addLogBarGrad_x(const double& mu, hiopVector& gradx) const;
  virtual void addLogBarGrad_d(const double& mu, hiopVector& gradd) const;

  /**
   * Computes the log barrier's linear damping term of the Filter-IPM method of WaectherBiegler (section 3.7) 
   */
  virtual double linearDampingTerm(const double& mu, const double& kappa_d) const;
  /* adds the damping term to the gradient */
  virtual void addLinearDampingTermToGrad_x(const double& mu, const double& kappa_d, const double& beta,
					    hiopVector& grad_x) const;
  virtual void addLinearDampingTermToGrad_d(const double& mu, const double& kappa_d, const double& beta,
					    hiopVector& grad_d) const;

  /** norms for individual parts of the iterate (on demand computation) */
  virtual double normOneOfBoundDuals() const;
  virtual double normOneOfEqualityDuals() const;
  /* same as above but computed in one shot to save on communication and computation */
  virtual void   normOneOfDuals(double& nrm1Eq, double& nrm1Bnd) const;

  /* cloning and copying */
  hiopIterate* alloc_clone() const;
  hiopIterate* new_copy() const;
  void copyFrom(const hiopIterate& src);

  /* accessors */
  inline hiopVector* get_x()   const {return x;}
  inline hiopVector* get_d()   const {return d;}
  inline hiopVector* get_sxl() const {return sxl;}
  inline hiopVector* get_yc()  const {return yc;}
  inline hiopVector* get_yd()  const {return yd;}
  inline hiopVector* get_zl()  const {return zl;}
  inline hiopVector* get_zu()  const {return zu;}
  inline hiopVector* get_vl()  const {return vl;}
  inline hiopVector* get_vu()  const {return vu;}

  void print(FILE* f, const char* msg=NULL) const;

  friend class hiopResidual;
  friend class hiopKKTLinSys;
  friend class hiopKKTLinSysCompressedXYcYd;
  friend class hiopKKTLinSysCompressedXDYcYd;
  friend class hiopKKTLinSysDenseXYcYd;
  friend class hiopKKTLinSysDenseXDYcYd;
  friend class hiopKKTLinSysLowRank;
  friend class hiopHessianLowRank;
  friend class hiopKKTLinSysCompressedMDSXYcYd;
  friend class hiopHessianInvLowRank_obsolette;
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
  //associated info from problem formulation
  const hiopNlpFormulation * nlp;
private:
  hiopIterate() {};
  hiopIterate(const hiopIterate&) {};
  hiopIterate& operator=(const hiopIterate& o) {return *this;}
};

}
#endif
