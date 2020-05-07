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

#include "hiopCSR_IO.hpp"

namespace hiop
{

/* KKT system treated as dense; used for developement/testing purposes mainly 
 * updates the parts in KKT system that are dependent on the iterate. 
 * Triggers a refactorization for the dense linear system 
 * Forms the linear system
 * [  H  +  Dx   Jc^T   Jd^T   ] [ dx]   [ rx_tilde ]
 * [    Jc        0       0    ] [dyc] = [   ryc    ]
 * [    Jd        0   -Dd^{-1} ] [dyd]   [   ryd    ]  
 */ 
class hiopKKTLinSysDenseXYcYd : public hiopKKTLinSysCompressedXYcYd
{
public:
  hiopKKTLinSysDenseXYcYd(hiopNlpFormulation* nlp_)
    : hiopKKTLinSysCompressedXYcYd(nlp_), linSys(NULL), rhsXYcYd(NULL), 
      write_linsys_counter(-1), csr_writer(nlp_)
  {
  }
  virtual ~hiopKKTLinSysDenseXYcYd()
  {
    delete linSys;
    delete rhsXYcYd;
  }

  /* updates the parts in KKT system that are dependent on the iterate. 
   * Triggers a refactorization for the dense linear system */

  bool update(const hiopIterate* iter_, 
	      const hiopVector* grad_f_, 
	      const hiopMatrix* Jac_c_,
	      const hiopMatrix* Jac_d_, 
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

      if(nlp->options->GetString("compute_mode")=="hybrid") {
#ifdef HIOP_USE_MAGMA
	linSys = new hiopLinSolverIndefDenseMagma(n, nlp);
	nlp->log->printf(hovScalars, "LinSysDenseXYcYd: Magma for a matrix of size %d\n", n);
#else
	linSys = new hiopLinSolverIndefDenseLapack(n, nlp);
	nlp->log->printf(hovScalars, "LinSysDenseXYcYd: Lapack for a matrix of size %d\n", n);
#endif
      } else {
	linSys = new hiopLinSolverIndefDenseLapack(n, nlp);
	nlp->log->printf(hovScalars, "LinSysDenseXYcYd: Lapack for a matrix of size %d\n", n);
      }
    }
    
    hiopMatrixDense& Msys = linSys->sysMatrix();
    //update linSys system matrix
    {
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
      Msys.addSubDiagonal(alpha, 0, *Dx);
      
      //Dd=(Sdl)^{-1}Vu + (Sdu)^{-1}Vu
      Dd_inv->setToZero();
      Dd_inv->axdzpy_w_pattern(1.0, *iter->vl, *iter->sdl, nlp->get_idl());
      Dd_inv->axdzpy_w_pattern(1.0, *iter->vu, *iter->sdu, nlp->get_idu());
#ifdef HIOP_DEEPCHECKS
      assert(true==Dd_inv->allPositive());
#endif 
      Dd_inv->invert();

      alpha=-1.;
      Msys.addSubDiagonal(alpha, nx+neq, *Dd_inv);

      nlp->log->write("KKT Linsys:", Msys, hovMatrices);
    }

    //write matrix to file if requested
    if(nlp->options->GetString("write_kkt") == "yes") write_linsys_counter++;
    if(write_linsys_counter>=0) csr_writer.writeMatToFile(Msys, write_linsys_counter); 

    //
    //factorization + inertia correction if needed
    //
    perturb_calc_->initialize(nlp);
    const size_t max_ic_cor = 10;
    size_t num_ic_cor = 0;
    while(num_ic_cor<=max_ic_cor) {
      //factorize the matrix
      int n_neg_eig = linSys->matrixChanged();
      
      if(Jac_c->m()+Jac_d->m()>0) {
	if(n_neg_eig < 0) {
	  //matrix singular
	  nlp->log->printf(hovScalars, "XDycYd linsys is singular.\n");
	  
	} else if(n_neg_eig != Jac_c->m()+Jac_d->m()) {
	  //wrong inertia
	  nlp->log->printf(hovScalars,
			   "XDycYd linsys negative eigs mismatch: has %d expected %d.\n",
			   Jac_c->m()+Jac_d->m(), n_neg_eig);
	  
	} else {
	  //all is good
	  break;
	}
      } else if(n_neg_eig != 0) {
	//correct for wrong intertia
	nlp->log->printf(hovScalars,
			 "XDycYd linsys has wrong inertia (no constraints): factoriz ret code %d\n.",
			 n_neg_eig);
      } else {
	//all is good
	break;
      }
   
      //will do an inertia correction
      num_ic_cor++;
    }
    if(num_ic_cor>max_ic_cor) {
      
      nlp->log->printf(hovError,
		       "Reached max number (%d) of inertia corrections within an outer iteration.\n",
		       max_ic_cor);
      return false;
    }
    nlp->runStats.tmSolverInternal.stop();
    return true;
  }

  virtual void solveCompressed(hiopVectorPar& rx, hiopVectorPar& ryc, hiopVectorPar& ryd,
			       hiopVectorPar& dx, hiopVectorPar& dyc, hiopVectorPar& dyd)
  {
    int nx=rx.get_size(), nyc=ryc.get_size(), nyd=ryd.get_size();
    if(rhsXYcYd == NULL) rhsXYcYd = new hiopVectorPar(nx+nyc+nyd);

    nlp->log->write("RHS KKT XDycYd rx: ", rx,  hovIteration);
    nlp->log->write("RHS KKT XDycYd ryc:", ryc, hovIteration);
    nlp->log->write("RHS KKT XDycYd ryd:", ryd, hovIteration);

    rx. copyToStarting(*rhsXYcYd, 0);
    ryc.copyToStarting(*rhsXYcYd, nx);
    ryd.copyToStarting(*rhsXYcYd, nx+nyc);

    if(write_linsys_counter>=0) csr_writer.writeRhsToFile(*rhsXYcYd, write_linsys_counter);

    //to do: iterative refinement
    linSys->solve(*rhsXYcYd);

    if(write_linsys_counter>=0) csr_writer.writeSolToFile(*rhsXYcYd, write_linsys_counter);

    rhsXYcYd->copyToStarting(0,      dx);
    rhsXYcYd->copyToStarting(nx,     dyc);
    rhsXYcYd->copyToStarting(nx+nyc, dyd);

    nlp->log->write("SOL KKT XYcYd dx: ", dx,  hovMatrices);
    nlp->log->write("SOL KKT XYcYd dyc:", dyc, hovMatrices);
    nlp->log->write("SOL KKT XYcYd dyd:", dyd, hovMatrices);
  }

protected:
  hiopLinSolverIndefDense* linSys;
  hiopVectorPar* rhsXYcYd;
  //-1 when disabled; otherwise acts like a counter, 0,1,... incremented each time 'solveCompressed' is called
  //depends on the 'write_kkt' option
  int write_linsys_counter; 
  hiopCSR_IO csr_writer;
private:
  hiopKKTLinSysDenseXYcYd() 
    :  hiopKKTLinSysCompressedXYcYd(NULL), linSys(NULL), 
       write_linsys_counter(-1), csr_writer(NULL)
  { 
    assert(false); 
  }
};

/** KKT system treated as dense; used for developement/testing purposes mainly */
class hiopKKTLinSysDenseXDYcYd : public hiopKKTLinSysCompressedXDYcYd
{
public:
  hiopKKTLinSysDenseXDYcYd(hiopNlpFormulation* nlp_)
    : hiopKKTLinSysCompressedXDYcYd(nlp_), linSys(NULL), rhsXDYcYd(NULL),
      write_linsys_counter(-1), csr_writer(nlp_)
  {
  }
  virtual ~hiopKKTLinSysDenseXDYcYd()
  {
    delete linSys;
    delete rhsXDYcYd;
  }

  /* updates the parts in KKT system that are dependent on the iterate. 
   * Triggers a refactorization for the dense linear system 
   * Forms the linear system
   * [  H  +  Dx    0    Jc^T  Jd^T   ] [ dx]   [ rx_tilde ]
   * [    0         Dd    0     -I    ] [ dd]   [ rd_tilde ]
   * [    Jc        0     0      0    ] [dyc] = [   ryc    ]
   * [    Jd       -I     0      0    ] [dyd]   [   ryd    ]  
   */ 
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
      int n=nx+neq+2*nineq;

      if(nlp->options->GetString("compute_mode")=="hybrid") {
#ifdef HIOP_USE_MAGMA
	nlp->log->printf(hovScalars, "LinSysDenseDXYcYd: Magma for a matrix of size %d\n", n);
	linSys = new hiopLinSolverIndefDenseMagma(n, nlp);
#else
	nlp->log->printf(hovScalars, "LinSysDenseXDYcYd: Lapack for a matrix of size %d\n", n);
	linSys = new hiopLinSolverIndefDenseLapack(n, nlp);
#endif
      } else {
	nlp->log->printf(hovScalars, "LinSysDenseXDYcYd Lapack for a matrix of size %d\n", n);
	linSys = new hiopLinSolverIndefDenseLapack(n, nlp);
      }	
    }
    hiopMatrixDense& Msys = linSys->sysMatrix();
    //update linSys system matrix
    {
      Msys.setToZero();
      
      const int alpha = 1.;
      Hess->addUpperTriangleToSymDenseMatrixUpperTriangle(0, alpha, Msys);
      
      Jac_c->transAddToSymDenseMatrixUpperTriangle(0, nx+nineq,     alpha, Msys);
      Jac_d->transAddToSymDenseMatrixUpperTriangle(0, nx+nineq+neq, alpha, Msys);
      
      //compute and put the barrier diagonals in
      //Dx=(Sxl)^{-1}Zl + (Sxu)^{-1}Zu
      Dx->setToZero();
      Dx->axdzpy_w_pattern(1.0, *iter->zl, *iter->sxl, nlp->get_ixl());
      Dx->axdzpy_w_pattern(1.0, *iter->zu, *iter->sxu, nlp->get_ixu());
      nlp->log->write("Dx in KKT", *Dx, hovMatrices);
      Msys.addSubDiagonal(alpha, 0, *Dx);
      
      //Dd=(Sdl)^{-1}Vu + (Sdu)^{-1}Vu
      Dd->setToZero();
      Dd->axdzpy_w_pattern(1.0, *iter->vl, *iter->sdl, nlp->get_idl());
      Dd->axdzpy_w_pattern(1.0, *iter->vu, *iter->sdu, nlp->get_idu());
#ifdef HIOP_DEEPCHECKS
      assert(true==Dd->allPositive());
#endif 

      Msys.addSubDiagonal(alpha, nx, *Dd);


      //add -I (of size nineq) starting at index (nx, nx+nineq+neq)
      {
	int col_start = nx+nineq+neq;
	double** MsysM = Msys.local_data();
	for(int i=nx; i<nx+nineq; i++) MsysM[i][col_start++] -= 1.;
      }

      nlp->log->write("KKT XDYcYd Linsys:", Msys, hovMatrices);
    }

    //write matrix to file if requested
    if(nlp->options->GetString("write_kkt") == "yes") write_linsys_counter++;
    if(write_linsys_counter>=0) csr_writer.writeMatToFile(Msys, write_linsys_counter); 

    //factorize
    linSys->matrixChanged();

    nlp->runStats.tmSolverInternal.stop();
    return true;
  }

  virtual void solveCompressed(hiopVectorPar& rx, hiopVectorPar& rd, hiopVectorPar& ryc, hiopVectorPar& ryd,
			       hiopVectorPar& dx, hiopVectorPar& dd, hiopVectorPar& dyc, hiopVectorPar& dyd)
  {
    int nx=rx.get_size(), nyc=ryc.get_size(), nyd=ryd.get_size();
    if(rhsXDYcYd == NULL) rhsXDYcYd = new hiopVectorPar(nx+nyc+2*nyd);

    nlp->log->write("RHS KKT XDycYd rx: ", rx,  hovMatrices);
    nlp->log->write("RHS KKT XDycYd rd: ", rd,  hovMatrices);
    nlp->log->write("RHS KKT XDycYd ryc:", ryc, hovMatrices);
    nlp->log->write("RHS KKT XDycYd ryd:", ryd, hovMatrices);

    

    rx. copyToStarting(*rhsXDYcYd, 0);
    rd. copyToStarting(*rhsXDYcYd, nx);
    ryc.copyToStarting(*rhsXDYcYd, nx+nyd);
    ryd.copyToStarting(*rhsXDYcYd, nx+nyd+nyc);

    if(write_linsys_counter>=0) csr_writer.writeRhsToFile(*rhsXDYcYd, write_linsys_counter);

    linSys->solve(*rhsXDYcYd);

    if(write_linsys_counter>=0) csr_writer.writeSolToFile(*rhsXDYcYd, write_linsys_counter);

    rhsXDYcYd->copyToStarting(0,          dx);
    rhsXDYcYd->copyToStarting(nx,         dd);
    rhsXDYcYd->copyToStarting(nx+nyd,     dyc);
    rhsXDYcYd->copyToStarting(nx+nyd+nyc, dyd);

    nlp->log->write("SOL KKT XDYcYd dx: ", dx,  hovMatrices);
    nlp->log->write("SOL KKT XDYcYd dd: ", dd,  hovMatrices);
    nlp->log->write("SOL KKT XDYcYd dyc:", dyc, hovMatrices);
    nlp->log->write("SOL KKT XDYcYd dyd:", dyd, hovMatrices);
  }

protected:
  hiopLinSolverIndefDense* linSys;
  hiopVectorPar* rhsXDYcYd;
  //-1 when disabled; otherwise acts like a counter, 0,1,... incremented each time 'solveCompressed' is called
  //depends on the 'write_kkt' option
  int write_linsys_counter; 
  hiopCSR_IO csr_writer;
private:
  hiopKKTLinSysDenseXDYcYd() 
    : hiopKKTLinSysCompressedXDYcYd(NULL), linSys(NULL), 
      write_linsys_counter(-1), csr_writer(NULL)
  { 
    assert(false && "not intended to be used"); 
  }
};


} //end of namespace

#endif
