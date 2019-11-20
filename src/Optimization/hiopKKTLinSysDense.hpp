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

#ifndef HIOP_KKTLINSYSY_DENSE
#define HIOP_KKTLINSYSY_DENSE

#include "hiopKKTLinSys.hpp"
#include "hiopLinSolver.hpp"

namespace hiop
{

/** KKT system treated as dense; used for developement/testing purposes mainly */
class hiopKKTLinSysDense : public hiopKKTLinSysCompressedXYcYd
{
public:
  hiopKKTLinSysDense(hiopNlpFormulation* nlp_)
    : hiopKKTLinSysCompressedXYcYd(nlp_), linSys(NULL), rhsXYcYd(NULL)
  {
  }
  virtual ~hiopKKTLinSysDense()
  {
    delete linSys;
    delete rhsXYcYd;
  }

  /* updates the parts in KKT system that are dependent on the iterate. 
   * Triggers a refactorization for the dense linear system */

  bool update(const hiopIterate* iter_, 
	      const hiopVector* grad_f_, 
	      const hiopMatrix* Jac_c_, const hiopMatrix* Jac_d_, 
	      hiopMatrix* Hess_)
  {
    nlp->runStats.tmSolverInternal.start();

    iter = iter_;   
    grad_f = dynamic_cast<const hiopVectorPar*>(grad_f_);
    Jac_c = Jac_c_; Jac_d = Jac_d_;

    Hess=Hess_;

    int nx  = Hess->m(); assert(nx==Hess->n()); assert(nx==Jac_c->n()); assert(nx==Jac_d->n()); 
    int neq = Jac_c->m(), nineq = Jac_d->m();
    
    if(NULL==linSys) {
      int n=Jac_c->m() + Jac_d->m() + Hess->m();
      linSys = new hiopLinSolverIndefDense(n, nlp);
    }

    //update linSys system matrix
    {
      hiopMatrixDense& Msys = linSys->sysMatrix();
      Msys.setToZero();
      
      int alpha = 1.;
      Hess->addUpperTriangleToSymDenseMatrixUpperTriangle(0, alpha, Msys);
      
      Jac_c->transAddToSymDenseMatrixUpperTriangle(0, nx,     alpha, Msys);
      Jac_d->transAddToSymDenseMatrixUpperTriangle(0, nx+neq, alpha, Msys);
      
      //compute and put the barrier diagonals in
      //Dx=(Sxl)^{-1}Zl + (Sxu)^{-1}Zu
      Dx->setToZero();
      Dx->axdzpy_w_pattern(1.0, *iter->zl, *iter->sxl, nlp->get_ixl());
      Dx->axdzpy_w_pattern(1.0, *iter->zu, *iter->sxu, nlp->get_ixu());
      nlp->log->write("Dx in KKT", *Dx, hovMatrices);
      
      //Dd=(Sdl)^{-1}Vu + (Sdu)^{-1}Vu
      Dd_inv->setToZero();
      Dd_inv->axdzpy_w_pattern(1.0, *iter->vl, *iter->sdl, nlp->get_idl());
      Dd_inv->axdzpy_w_pattern(1.0, *iter->vu, *iter->sdu, nlp->get_idu());
#ifdef HIOP_DEEPCHECKS
      assert(true==Dd_inv->allPositive());
#endif 
      Dd_inv->invert();

      Msys.addSubDiagonal(0, *Dx);
      Msys.addSubDiagonal(nx+neq, *Dd_inv);
    }
    
    linSys->matrixChanged();

    nlp->runStats.tmSolverInternal.stop();
    return true;
  }

  virtual void solveCompressed(hiopVectorPar& rx, hiopVectorPar& ryc, hiopVectorPar& ryd,
			       hiopVectorPar& dx, hiopVectorPar& dyc, hiopVectorPar& dyd)
  {
    int nx=rx.get_size(), nyc=ryc.get_size(), nyd=ryd.get_size();
    if(rhsXYcYd == NULL) rhsXYcYd = new hiopVectorPar(nx+nyc+nyd);

    rx. copyToStarting(*rhsXYcYd, 0);
    ryc.copyToStarting(*rhsXYcYd, nx);
    ryd.copyToStarting(*rhsXYcYd, nx+nyc);
    
    linSys->solve(*rhsXYcYd);

    rhsXYcYd->copyToStarting(0,      dx);
    rhsXYcYd->copyToStarting(nx,     dyc);
    rhsXYcYd->copyToStarting(nx+nyc, dyd);
  }

#ifdef HIOP_DEEPCHECKS
  double errorCompressedLinsys(const hiopVectorPar& rx, const hiopVectorPar& ryc, const hiopVectorPar& ryd,
			       const hiopVectorPar& dx, const hiopVectorPar& dyc, const hiopVectorPar& dyd)
  {
    nlp->log->printf(hovLinAlgScalars, "hiopKKTLinSysLowRank::errorCompressedLinsys residuals norm:\n");
    
    double derr=1e20, aux;
    hiopVectorPar *RX=rx.new_copy();
    //RX=rx-H*dx-J'c*dyc-J'*dyd
    Hess->timesVec(1.0, *RX, -1.0, dx);
    RX->axzpy(-1.0,*Dx,dx);

    Jac_c->transTimesVec(1.0, *RX, -1.0, dyc);
    Jac_d->transTimesVec(1.0, *RX, -1.0, dyd);
    aux=RX->twonorm();
    derr=fmax(derr,aux);
    nlp->log->printf(hovLinAlgScalars, "  >>>  rx=%g\n", aux);
    delete RX; RX=NULL;

    hiopVectorPar* RC=ryc.new_copy();
    Jac_c->timesVec(1.0,*RC, -1.0,dx);
    aux = RC->twonorm();
    derr=fmax(derr,aux);
    nlp->log->printf(hovLinAlgScalars, "  >>> ryc=%g\n", aux);
    delete RC; RC=NULL;
    
    hiopVectorPar* RD=ryd.new_copy();
    Jac_d->timesVec(1.0,*RD, -1.0, dx);
    RD->axzpy(1.0, *Dd_inv, dyd);
    aux = RD->twonorm();
    derr=fmax(derr,aux);
    nlp->log->printf(hovLinAlgScalars, "  >>> ryd=%g\n", aux);
    delete RD; RD=NULL;
    
    return derr;
  }
#endif

protected:
  hiopLinSolverIndefDense* linSys;
  hiopVectorPar* rhsXYcYd;
private:
  hiopKKTLinSysDense() :  hiopKKTLinSysCompressedXYcYd(NULL), linSys(NULL) { assert(false); }
};


} //end of namespace

#endif
