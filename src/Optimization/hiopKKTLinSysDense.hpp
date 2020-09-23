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
#include "hiopLinSolverIndefDenseLapack.hpp"

#ifdef HIOP_USE_MAGMA
#include "hiopLinSolverIndefDenseMagma.hpp"
#endif

#include "hiopCSR_IO.hpp"

namespace hiop
{

/* KKT system treated as dense; used for development/testing purposes mainly
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
  hiopKKTLinSysDenseXYcYd(hiopNlpFormulation* nlp)
    : hiopKKTLinSysCompressedXYcYd(nlp), linSys(NULL), rhsXYcYd(NULL),
      write_linsys_counter(-1), csr_writer(nlp)
  {
  }
  virtual ~hiopKKTLinSysDenseXYcYd()
  {
    delete linSys;
    delete rhsXYcYd;
  }

  /* updates the parts in KKT system that are dependent on the iterate.
   * Triggers a refactorization for the dense linear system */

  bool update(const hiopIterate* iter,
	      const hiopVector* grad_f,
	      const hiopMatrix* Jac_c,
	      const hiopMatrix* Jac_d,
	      hiopMatrix* Hess)
  {
    nlp_->runStats.tmSolverInternal.start();

    iter_ = iter;
    grad_f_ = dynamic_cast<const hiopVectorPar*>(grad_f_);
    Jac_c_ = Jac_c; Jac_d_ = Jac_d;
    Hess_=Hess;

    int nx  = Hess_->m();
    assert(nx==Hess_->n()); assert(nx==Jac_c_->n()); assert(nx==Jac_d_->n());
    int neq = Jac_c_->m(), nineq = Jac_d_->m();

    if(NULL==linSys) {
      int n=Jac_c_->m() + Jac_d_->m() + Hess_->m();

      if(nlp_->options->GetString("compute_mode")=="hybrid") {
#ifdef HIOP_USE_MAGMA
	linSys = new hiopLinSolverIndefDenseMagmaNopiv(n, nlp_);
	nlp_->log->printf(hovScalars,
			  "LinSysDenseXYcYd: instantiating Magma for a matrix of size %d\n",
			  n);
#else
	linSys = new hiopLinSolverIndefDenseLapack(n, nlp_);
	nlp_->log->printf(hovScalars,
			  "LinSysDenseXYcYd: instantiating Lapack for a matrix of size %d\n",
			  n);
#endif
      } else {
	linSys = new hiopLinSolverIndefDenseLapack(n, nlp_);
	nlp_->log->printf(hovScalars,
			  "LinSysDenseXYcYd: instantiating Lapack for a matrix of size %d\n",
			  n);
      }
    }

    //compute and put the barrier diagonals in
    //Dx=(Sxl)^{-1}Zl + (Sxu)^{-1}Zu
    Dx_->setToZero();
    Dx_->axdzpy_w_pattern(1.0, *iter_->zl, *iter_->sxl, nlp_->get_ixl());
    Dx_->axdzpy_w_pattern(1.0, *iter_->zu, *iter_->sxu, nlp_->get_ixu());
    nlp_->log->write("Dx in KKT", *Dx_, hovMatrices);

    // Dd=(Sdl)^{-1}Vu + (Sdu)^{-1}Vu is computed in the IC loop since we need to
    // add delta_wd and then invert

    hiopMatrixDense& Msys = linSys->sysMatrix();

    //
    //factorization + inertia correction if needed
    //
    const size_t max_ic_cor = 10;
    size_t num_ic_cor = 0;

    //nlp_->log->write("KKT XDYcYd Linsys (no perturb):", Msys, hovMatrices);

    double delta_wx, delta_wd, delta_cc, delta_cd;
    if(!perturb_calc_->compute_initial_deltas(delta_wx, delta_wd, delta_cc, delta_cd)) {
      nlp_->log->printf(hovWarning, "XDycYd linsys: IC perturbation on new linsys failed.\n");
      return false;
    }

    while(num_ic_cor<=max_ic_cor) {

      assert(delta_wx == delta_wd && "something went wrong with IC");
      assert(delta_cc == delta_cd && "something went wrong with IC");
      nlp_->log->printf(hovScalars, "XYcYd linsys: delta_w=%12.5e delta_c=%12.5e (ic %d)\n",
			delta_wx, delta_cc, num_ic_cor);

      //
      // update linSys system matrix, including IC perturbations
      //
      Msys.setToZero();

      int alpha = 1.;
      Hess_->addUpperTriangleToSymDenseMatrixUpperTriangle(0, alpha, Msys);

      Jac_c_->transAddToSymDenseMatrixUpperTriangle(0, nx,     alpha, Msys);
      Jac_d_->transAddToSymDenseMatrixUpperTriangle(0, nx+neq, alpha, Msys);

      Msys.addSubDiagonal(alpha, 0, *Dx_);
      Msys.addSubDiagonal(0, nx, delta_wx);

      //Dd=(Sdl)^{-1}Vu + (Sdu)^{-1}Vu + delta_wd*I
      Dd_inv_->setToConstant(delta_wd);
      Dd_inv_->axdzpy_w_pattern(1.0, *iter_->vl, *iter_->sdl, nlp_->get_idl());
      Dd_inv_->axdzpy_w_pattern(1.0, *iter_->vu, *iter_->sdu, nlp_->get_idu());
#ifdef HIOP_DEEPCHECKS
      assert(true==Dd_inv_->allPositive());
#endif
      Dd_inv_->invert();

      alpha=-1.;
      Msys.addSubDiagonal(alpha, nx+neq, *Dd_inv_);

      assert(delta_cc == delta_cd);
      //Msys.addSubDiagonal(nx+nineq, neq, -delta_cc);
      //Msys.addSubDiagonal(nx+nineq+neq, nineq, -delta_cd);
      Msys.addSubDiagonal(nx, neq+nineq, -delta_cd);

      nlp_->log->write("KKT Linsys:", Msys, hovMatrices);

      //write matrix to file if requested
      if(nlp_->options->GetString("write_kkt") == "yes") write_linsys_counter++;
      if(write_linsys_counter>=0) csr_writer.writeMatToFile(Msys, write_linsys_counter);

      int n_neg_eig = linSys->matrixChanged();

      if(Jac_c_->m()+Jac_d_->m()>0) {
	if(n_neg_eig < 0) {
	  //matrix singular
	  nlp_->log->printf(hovScalars, "XYcYd linsys is singular.\n");

	  if(!perturb_calc_->compute_perturb_singularity(delta_wx, delta_wd, delta_cc, delta_cd)) {
	    nlp_->log->printf(hovWarning, "XYcYd linsys: computing singularity perturbation failed.\n");
	    return false;
	  }

	} else if(n_neg_eig != Jac_c_->m()+Jac_d_->m()) {
	  //wrong inertia
	  nlp_->log->printf(hovScalars, "XYcYd linsys negative eigs mismatch: has %d expected %d.\n",
			    n_neg_eig,  Jac_c_->m()+Jac_d_->m());

	  if(!perturb_calc_->compute_perturb_wrong_inertia(delta_wx, delta_wd, delta_cc, delta_cd)) {
	    nlp_->log->printf(hovWarning, "XYcYd linsys: computing inertia perturbation failed.\n");
	    return false;
	  }

	} else {
	  //all is good
	  break;
	}
      } else if(n_neg_eig != 0) {
	//correct for wrong intertia
	nlp_->log->printf(hovScalars,  "XYcYd linsys has wrong inertia (no constraints): factoriz "
			 "ret code %d\n.", n_neg_eig);
	if(!perturb_calc_->compute_perturb_wrong_inertia(delta_wx, delta_wd, delta_cc, delta_cd)) {
	  nlp_->log->printf(hovWarning, "XYcYd linsys: computing inertia perturbation failed (2).\n");
	  return false;
	}

      } else {
	//all is good
	break;
      }

      //will do an inertia correction
      num_ic_cor++;
    } // end of IC loop

    if(num_ic_cor>max_ic_cor) {

      nlp_->log->printf(hovError,
			"Reached max number (%d) of inertia corrections within an outer iteration.\n",
			max_ic_cor);
      return false;
    }

    nlp_->runStats.tmSolverInternal.stop();
    return true;
  }

  virtual bool solveCompressed(hiopVector& rx, hiopVector& ryc, hiopVector& ryd,
			       hiopVector& dx, hiopVector& dyc, hiopVector& dyd)
  {
    int nx=rx.get_size(), nyc=ryc.get_size(), nyd=ryd.get_size();
    if(rhsXYcYd == NULL) rhsXYcYd = LinearAlgebraFactory::createVector(nx+nyc+nyd);

    nlp_->log->write("RHS KKT XDycYd rx: ", rx,  hovIteration);
    nlp_->log->write("RHS KKT XDycYd ryc:", ryc, hovIteration);
    nlp_->log->write("RHS KKT XDycYd ryd:", ryd, hovIteration);

    rx. copyToStarting(*rhsXYcYd, 0);
    ryc.copyToStarting(*rhsXYcYd, nx);
    ryd.copyToStarting(*rhsXYcYd, nx+nyc);

    if(write_linsys_counter>=0) csr_writer.writeRhsToFile(*rhsXYcYd, write_linsys_counter);

    //! todo: iterative refinement
    bool sol_ok = linSys->solve(*rhsXYcYd);

    if(write_linsys_counter>=0) csr_writer.writeSolToFile(*rhsXYcYd, write_linsys_counter);

    if(false==sol_ok) return false;

    rhsXYcYd->copyToStarting(0,      dx);
    rhsXYcYd->copyToStarting(nx,     dyc);
    rhsXYcYd->copyToStarting(nx+nyc, dyd);

    nlp_->log->write("SOL KKT XYcYd dx: ", dx,  hovMatrices);
    nlp_->log->write("SOL KKT XYcYd dyc:", dyc, hovMatrices);
    nlp_->log->write("SOL KKT XYcYd dyd:", dyd, hovMatrices);
    return true;
  }

protected:
  hiopLinSolverIndefDense* linSys;
  hiopVector* rhsXYcYd;

  /** -1 when disabled; otherwise acts like a counter, 0,1,...
   * incremented each time 'solveCompressed' is called depends on the 'write_kkt' option
   */
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
  hiopKKTLinSysDenseXDYcYd(hiopNlpFormulation* nlp)
    : hiopKKTLinSysCompressedXDYcYd(nlp), linSys(NULL), rhsXDYcYd(NULL),
      write_linsys_counter(-1), csr_writer(nlp)
  {
  }
  virtual ~hiopKKTLinSysDenseXDYcYd()
  {
    delete linSys;
    delete rhsXDYcYd;
  }

  /* Updates the parts in KKT system that are dependent on the iterate.
   * Triggers a refactorization for the dense linear system
   * Forms the linear system
   * [  H  +  Dx    0    Jc^T  Jd^T   ] [ dx]   [ rx_tilde ]
   * [    0         Dd    0     -I    ] [ dd]   [ rd_tilde ]
   * [    Jc        0     0      0    ] [dyc] = [   ryc    ]
   * [    Jd       -I     0      0    ] [dyd]   [   ryd    ]
   */
  bool update(const hiopIterate* iter,
	      const hiopVector* grad_f,
	      const hiopMatrix* Jac_c, const hiopMatrix* Jac_d,
	      hiopMatrix* Hess)
  {
    nlp_->runStats.tmSolverInternal.start();

    iter_ = iter;
    grad_f = dynamic_cast<const hiopVectorPar*>(grad_f_);
    Jac_c_ = Jac_c; Jac_d_ = Jac_d;

    Hess_=Hess;

    int nx  = Hess_->m(); assert(nx==Hess_->n()); assert(nx==Jac_c_->n()); assert(nx==Jac_d_->n());
    int neq = Jac_c_->m(), nineq = Jac_d_->m();

    if(NULL==linSys) {
      int n=nx+neq+2*nineq;

      if(nlp_->options->GetString("compute_mode")=="hybrid") {
#ifdef HIOP_USE_MAGMA
	nlp_->log->printf(hovScalars, "LinSysDenseDXYcYd: instantiating Magma for a matrix of size %d\n", n);
	linSys = new hiopLinSolverIndefDenseMagmaNopiv(n, nlp_);
#else
	nlp_->log->printf(hovScalars, "LinSysDenseXDYcYd: instantiating Lapack for a matrix of size %d\n", n);
	linSys = new hiopLinSolverIndefDenseLapack(n, nlp_);
#endif
      } else {
	nlp_->log->printf(hovScalars, "LinSysDenseXDYcYd instantiating Lapack for a matrix of size %d\n", n);
	linSys = new hiopLinSolverIndefDenseLapack(n, nlp_);
      }
    }

    //
    //compute barrier diagonals (these change only between outer optimiz iterations)
    //
    // Dx=(Sxl)^{-1}Zl + (Sxu)^{-1}Zu
    Dx_->setToZero();
    Dx_->axdzpy_w_pattern(1.0, *iter_->zl, *iter_->sxl, nlp_->get_ixl());
    Dx_->axdzpy_w_pattern(1.0, *iter_->zu, *iter_->sxu, nlp_->get_ixu());
    nlp_->log->write("Dx in KKT", *Dx_, hovMatrices);

    // Dd=(Sdl)^{-1}Vu + (Sdu)^{-1}Vu
    Dd_->setToZero();
    Dd_->axdzpy_w_pattern(1.0, *iter_->vl, *iter_->sdl, nlp_->get_idl());
    Dd_->axdzpy_w_pattern(1.0, *iter_->vu, *iter_->sdu, nlp_->get_idu());
    nlp_->log->write("Dd in KKT", *Dd_, hovMatrices);
#ifdef HIOP_DEEPCHECKS
    assert(true==Dd_->allPositive());
#endif

    hiopMatrixDense& Msys = linSys->sysMatrix();

    //
    //factorization + inertia correction if needed
    //
    const size_t max_ic_cor = 10;
    size_t num_ic_cor = 0;

    //nlp_->log->write("KKT XDYcYd Linsys (no perturb):", Msys, hovMatrices);

    double delta_wx, delta_wd, delta_cc, delta_cd;
    if(!perturb_calc_->compute_initial_deltas(delta_wx, delta_wd, delta_cc, delta_cd)) {
      nlp_->log->printf(hovWarning, "XDycYd linsys: IC perturbation on new linsys failed.\n");
      return false;
    }

    while(num_ic_cor<=max_ic_cor) {

      assert(delta_wx == delta_wd && "something went wrong with IC");
      assert(delta_cc == delta_cd && "something went wrong with IC");
      nlp_->log->printf(hovScalars, "XDycYd linsys: delta_w=%12.5e delta_c=%12.5e (ic %d)\n",
		       delta_wx, delta_cc, num_ic_cor);

      //
      // update linSys system matrix, including IC perturbations
      //
      {
	Msys.setToZero();

	const int alpha = 1.;
	Hess_->addUpperTriangleToSymDenseMatrixUpperTriangle(0, alpha, Msys);

	Jac_c_->transAddToSymDenseMatrixUpperTriangle(0, nx+nineq,     alpha, Msys);
	Jac_d_->transAddToSymDenseMatrixUpperTriangle(0, nx+nineq+neq, alpha, Msys);


	//add diagonals and IC perturbations
	Msys.addSubDiagonal(alpha, 0, *Dx_);
	Msys.addSubDiagonal(0, nx, delta_wx);

	Msys.addSubDiagonal(alpha, nx, *Dd_);
	Msys.addSubDiagonal(nx, nineq, delta_wd);

	//add -I (of size nineq) starting at index (nx, nx+nineq+neq)
	int col_start = nx+nineq+neq;
	double** MsysM = Msys.local_data();
	for(int i=nx; i<nx+nineq; i++) MsysM[i][col_start++] -= 1.;

	//add perturbations for IC (singularity)
	assert(delta_cc == delta_cd);
	//Msys.addSubDiagonal(nx+nineq, neq, -delta_cc);
	//Msys.addSubDiagonal(nx+nineq+neq, nineq, -delta_cd);
	Msys.addSubDiagonal(nx+nineq, neq+nineq, -delta_cd);

      } // end of update linSys system matrix

      //write matrix to file if requested
      if(nlp_->options->GetString("write_kkt") == "yes") write_linsys_counter++;
      if(write_linsys_counter>=0) csr_writer.writeMatToFile(Msys, write_linsys_counter);

      nlp_->log->write("KKT XDYcYd Linsys (to be factorized):", Msys, hovMatrices);

      //factorize the matrix (note: 'matrixChanged' returns -1 if null eigenvalues are detected)
      int n_neg_eig = linSys->matrixChanged();

      if(Jac_c_->m()+Jac_d_->m()>0) {
	if(n_neg_eig < 0) {
	  //matrix singular
	  nlp_->log->printf(hovScalars, "XDycYd linsys is singular.\n");

	  if(!perturb_calc_->compute_perturb_singularity(delta_wx, delta_wd, delta_cc, delta_cd)) {
	    nlp_->log->printf(hovWarning, "XDycYd linsys: computing singularity perturbation failed.\n");
	    return false;
	  }

	} else if(n_neg_eig != Jac_c_->m()+Jac_d_->m()) {
	  //wrong inertia
	  nlp_->log->printf(hovScalars, "XDycYd linsys negative eigs mismatch: has %d expected %d.\n",
			    n_neg_eig,  Jac_c_->m()+Jac_d_->m());

	  if(!perturb_calc_->compute_perturb_wrong_inertia(delta_wx, delta_wd, delta_cc, delta_cd)) {
	    nlp_->log->printf(hovWarning, "XDycYd linsys: computing inertia perturbation failed.\n");
	    return false;
	  }

	} else {
	  //all is good
	  //printf("!!!!! all is good\n");
	  break;
	}
      } else if(n_neg_eig != 0) {
	//correct for wrong intertia
	nlp_->log->printf(hovScalars,  "XDycYd linsys has wrong inertia (no constraints): factoriz "
			 "ret code %d\n.", n_neg_eig);
	if(!perturb_calc_->compute_perturb_wrong_inertia(delta_wx, delta_wd, delta_cc, delta_cd)) {
	  nlp_->log->printf(hovWarning, "XDycYd linsys: computing inertia perturbation failed (2).\n");
	  return false;
	}

      } else {
	//all is good
	break;
      }

      //will do an inertia correction
      num_ic_cor++;
    }

    if(num_ic_cor>max_ic_cor) {

      nlp_->log->printf(hovError,
		       "Reached max number (%d) of inertia corrections within an outer iteration.\n",
		       max_ic_cor);
      return false;
    }

    nlp_->runStats.tmSolverInternal.stop();
    return true;
  }

  virtual bool solveCompressed(hiopVector& rx, hiopVector& rd, hiopVector& ryc, hiopVector& ryd,
			       hiopVector& dx, hiopVector& dd, hiopVector& dyc, hiopVector& dyd)
  {
    int nx=rx.get_size(), nyc=ryc.get_size(), nyd=ryd.get_size();
    if(rhsXDYcYd == NULL) rhsXDYcYd = LinearAlgebraFactory::createVector(nx+nyc+2*nyd);

    nlp_->log->write("RHS KKT XDycYd rx: ", rx,  hovMatrices);
    nlp_->log->write("RHS KKT XDycYd rd: ", rd,  hovMatrices);
    nlp_->log->write("RHS KKT XDycYd ryc:", ryc, hovMatrices);
    nlp_->log->write("RHS KKT XDycYd ryd:", ryd, hovMatrices);

    rx. copyToStarting(*rhsXDYcYd, 0);
    rd. copyToStarting(*rhsXDYcYd, nx);
    ryc.copyToStarting(*rhsXDYcYd, nx+nyd);
    ryd.copyToStarting(*rhsXDYcYd, nx+nyd+nyc);

    if(write_linsys_counter>=0) csr_writer.writeRhsToFile(*rhsXDYcYd, write_linsys_counter);

    bool sol_ok = linSys->solve(*rhsXDYcYd);

    if(write_linsys_counter>=0) csr_writer.writeSolToFile(*rhsXDYcYd, write_linsys_counter);

    if(false==sol_ok) return false;

    rhsXDYcYd->copyToStarting(0,          dx);
    rhsXDYcYd->copyToStarting(nx,         dd);
    rhsXDYcYd->copyToStarting(nx+nyd,     dyc);
    rhsXDYcYd->copyToStarting(nx+nyd+nyc, dyd);

    nlp_->log->write("SOL KKT XDYcYd dx: ", dx,  hovMatrices);
    nlp_->log->write("SOL KKT XDYcYd dd: ", dd,  hovMatrices);
    nlp_->log->write("SOL KKT XDYcYd dyc:", dyc, hovMatrices);
    nlp_->log->write("SOL KKT XDYcYd dyd:", dyd, hovMatrices);
    return true;
  }

protected:
  hiopLinSolverIndefDense* linSys;
  hiopVector* rhsXDYcYd;
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
