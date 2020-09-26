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

#include "hiopKKTLinSysSparse.hpp"

//#ifdef HIOP_SPARSE
#include "hiopLinSolverIndefSparseMA57.hpp"
//#endif

namespace hiop
{

  /* *************************************************************************
   * For class hiopKKTLinSysCompressedSparseXYcYd
   * *************************************************************************
   */
  hiopKKTLinSysCompressedSparseXYcYd::hiopKKTLinSysCompressedSparseXYcYd(hiopNlpFormulation* nlp)
    : hiopKKTLinSysCompressedXYcYd(nlp), linSys_(NULL), rhs_(NULL),
      Hx_(NULL), HessSp_(NULL), Jac_cSp_(NULL), Jac_dSp_(NULL),
      write_linsys_counter_(-1), csr_writer_(nlp)
  {
    nlpSp_ = dynamic_cast<hiopNlpSparse*>(nlp_);
    assert(nlpSp_);
  }

  hiopKKTLinSysCompressedSparseXYcYd::~hiopKKTLinSysCompressedSparseXYcYd()
  {
    delete rhs_;
    delete linSys_;
    delete Hx_;
  }

  bool hiopKKTLinSysCompressedSparseXYcYd::update(const hiopIterate* iter,
					       const hiopVector* grad_f,
					       const hiopMatrix* Jac_c,
					       const hiopMatrix* Jac_d,
					       hiopMatrix* Hess)
  {
    if(!nlpSp_) { assert(false); return false; }

    nlp_->runStats.tmSolverInternal.start();
    nlp_->runStats.kkt.tmUpdateInit.start();

    iter_ = iter;
    grad_f_ = dynamic_cast<const hiopVectorPar*>(grad_f);
    Jac_c_ = Jac_c; Jac_d_ = Jac_d; Hess_=Hess;

    HessSp_ = dynamic_cast<hiopMatrixSymSparseTriplet*>(Hess);
    if(!HessSp_) { assert(false); return false; }

    Jac_cSp_ = dynamic_cast<const hiopMatrixSparseTriplet*>(Jac_c);
    if(!Jac_cSp_) { assert(false); return false; }

    Jac_dSp_ = dynamic_cast<const hiopMatrixSparseTriplet*>(Jac_d);
    if(!Jac_dSp_) { assert(false); return false; }

    long long nx = HessSp_->n(), neq=Jac_cSp_->m(), nineq=Jac_dSp_->m();

    int nnz = HessSp_->numberOfNonzeros() + Jac_cSp_->numberOfNonzeros() + Jac_dSp_->numberOfNonzeros() + nx + neq + nineq;

    //
    //based on safe_mode_, decide whether to go with the nopiv (fast) or Bunch-Kaufman (stable) linear solve
    //
    linSys_ = determineAndCreateLinsys(nx, neq, nineq, nnz);

    //
    //update/compute KKT
    //
    //Dx (<-- log-barrier diagonal, for variable x
    assert(Dx_->get_local_size() == nx);
    Dx_->setToZero();
    Dx_->axdzpy_w_pattern(1.0, *iter->zl, *iter->sxl, nlp_->get_ixl());
    Dx_->axdzpy_w_pattern(1.0, *iter->zu, *iter->sxu, nlp_->get_ixu());
    nlp_->log->write("Dx in KKT", *Dx_, hovMatrices);

    hiopMatrixSparseTriplet& Msys = linSys_->sysMatrix();
    if(perf_report_) {
      nlp_->log->printf(hovSummary,
			"KKT_Sparse_XYcYd linsys: Low-level linear system size: %d\n",
			Msys.n());
    }

    //
    //factorization + inertia correction if needed
    //
    const size_t max_ic_cor = 10;
    size_t num_ic_cor = 0;

    double delta_wx, delta_wd, delta_cc, delta_cd;
    if(!perturb_calc_->compute_initial_deltas(delta_wx, delta_wd, delta_cc, delta_cd)) {
      nlp_->log->printf(hovWarning,
			"KKT_Sparse_XYcYd linsys: IC perturbation on new linsys failed.\n");
      return false;
    }

    nlp_->runStats.kkt.tmUpdateInit.stop();

    while(num_ic_cor<=max_ic_cor) {

      assert(delta_wx == delta_wd && "something went wrong with IC");
      assert(delta_cc == delta_cd && "something went wrong with IC");
      nlp_->log->printf(hovScalars,
			"KKT_Sparse_XYcYd linsys: delta_w=%12.5e delta_c=%12.5e (ic %d)\n",
			delta_wx, delta_cc, num_ic_cor);

      //
      //the update of the linear system, including IC perturbations
      //
      nlp_->runStats.kkt.tmUpdateLinsys.start();
      {
      Msys.setToZero();

      //
      // copy Jac and Hes to the full iterate matrix
      //
      long long dest_nnz_st{0};
      Msys.copyRowsFromSrcToDest(*HessSp_,  0,   nx,     0,      dest_nnz_st); dest_nnz_st += HessSp_->numberOfNonzeros();
      Msys.copyRowsFromSrcToDest(*Jac_cSp_, 0,   neq,    nx,     dest_nnz_st); dest_nnz_st += Jac_cSp_->numberOfNonzeros();
      Msys.copyRowsFromSrcToDest(*Jac_dSp_, 0,   nineq,  nx+neq, dest_nnz_st); dest_nnz_st += Jac_dSp_->numberOfNonzeros();

	  //build the diagonal Hx = Dx + delta_wx
	  if(NULL == Hx_) {
	    Hx_ = LinearAlgebraFactory::createVector(nx); assert(Hx_);
	  }
	  Hx_->startingAtCopyFromStartingAt(0, *Dx_, 0);

	  //a good time to add the IC 'delta_wx' perturbation
	  Hx_->addConstant(delta_wx);

      Msys.copySubDiagonalEleFromVec(0, nx, *Hx_, dest_nnz_st); dest_nnz_st += nx;

	  //add -delta_cc to diagonal block linSys starting at (nx, nx)
      Msys.copySubDiagonalEleFromConstant(nx, neq, -delta_cc, dest_nnz_st); dest_nnz_st += neq;

	  /* we've just done above the (1,1) and (2,2) blocks of
      *
	  * [ Hx+Dxd+delta_wx*I           Jcd^T          Jdd^T   ]
	  * [  Jcd                       -delta_cc*I     0       ]
	  * [  Jdd                        0              M_{33} ]
	  *
	  * where
	  * M_{33} = - (Dd+delta_wd)*I^{-1} - delta_cd*I is performed below
	  */

	  // add -{Dd}^{-1}
	  // Dd=(Sdl)^{-1}Vu + (Sdu)^{-1}Vu + delta_wd * I
      Dd_inv_->setToConstant(delta_wd);
	  Dd_inv_->axdzpy_w_pattern(1.0, *iter->vl, *iter->sdl, nlp_->get_idl());
	  Dd_inv_->axdzpy_w_pattern(1.0, *iter->vu, *iter->sdu, nlp_->get_idu());

#ifdef HIOP_DEEPCHECKS
	assert(true==Dd_inv_->allPositive());
#endif
	  Dd_inv_->invert();
      Dd_inv_->addConstant(-delta_cd);

	  Msys.copySubDiagonalEleFromVec(nx+neq, nineq, *Dd_inv_, dest_nnz_st); dest_nnz_st += nineq;


	  nlp_->log->write("KKT_SPARSE_XYcYd linsys:", Msys, hovMatrices);
      } // end of update of the linear system

      nlp_->runStats.kkt.tmUpdateLinsys.stop();

      //write matrix to file if requested
      if(nlp_->options->GetString("write_kkt") == "yes") write_linsys_counter_++;



      nlp_->runStats.linsolv.start_linsolve();
      nlp_->runStats.kkt.tmUpdateInnerFact.start();
      //factorization
      int n_neg_eig = linSys_->matrixChanged();

      nlp_->runStats.kkt.tmUpdateInnerFact.stop();

      if(Jac_cSp_->m()+Jac_dSp_->m()>0) {
        if(n_neg_eig < 0) {
          //matrix singular
          nlp_->log->printf(hovScalars,
			    "KKT_SPARSE_XYcYdlinsys is singular. Regularization will be attempted...\n");

	  if(!perturb_calc_->compute_perturb_singularity(delta_wx, delta_wd, delta_cc, delta_cd)) {
	    nlp_->log->printf(hovWarning,
			      "KKT_SPARSE_XYcYd linsys: computing singularity perturbation failed.\n");
	    return false;
	  }

	} else if(n_neg_eig != Jac_cSp_->m() + Jac_dSp_->m()) {
	  //wrong inertia
	  nlp_->log->printf(hovScalars,
			    "KKT_SPARSE_XYcYd linsys negative eigs mismatch: has %d expected %d.\n",
			    n_neg_eig,  Jac_cSp_->m()+Jac_dSp_->m());


	  if(n_neg_eig < Jac_cSp_->m() + Jac_dSp_->m())
	    nlp_->log->printf(hovWarning, "KKT_SPARSE_XYcYd linsys negative eigs abnormality\n");


	  if(!perturb_calc_->compute_perturb_wrong_inertia(delta_wx, delta_wd, delta_cc, delta_cd)) {
	    nlp_->log->printf(hovWarning,
			      "KKT_SPARSE_XYcYd linsys: computing inertia perturbation failed.\n");
	    return false;
	  }

	} else {
	  //all is good
	  break;
	}
     } else if(n_neg_eig != 0) {
       //correct for wrong intertia
       nlp_->log->printf(hovScalars,
			 "KKT_SPARSE_XYcYd linsys has wrong inertia (no constraints): factoriz "
			 "ret code/num negative eigs %d\n.", n_neg_eig);
       if(!perturb_calc_->compute_perturb_wrong_inertia(delta_wx, delta_wd, delta_cc, delta_cd)) {
	 nlp_->log->printf(hovWarning,
			   "KKT_SPARSE_XYcYd linsys: computing inertia perturbation failed (2).\n");
	 return false;
       }

     } else {
       //all is good
       break;
     }

     //will do an inertia correction
     num_ic_cor++;
     nlp_->runStats.kkt.nUpdateICCorr++;
    } // end of ic while

    if(num_ic_cor>max_ic_cor) {

      nlp_->log->printf(hovError,
			"KKT_SPARSE_XYcYd linsys: max number (%d) of inertia corrections reached.\n",
			max_ic_cor);
      return false;
    }
    nlp_->runStats.tmSolverInternal.stop();
    return true;
  }

  bool hiopKKTLinSysCompressedSparseXYcYd::
  solveCompressed(hiopVector& rx, hiopVector& ryc, hiopVector& ryd,
		  hiopVector& dx, hiopVector& dyc, hiopVector& dyd)
  {
    if(!nlpSp_)   { assert(false); return false; }
    if(!HessSp_)  { assert(false); return false; }
    if(!Jac_cSp_) { assert(false); return false; }
    if(!Jac_dSp_) { assert(false); return false; }

    nlp_->runStats.kkt.tmSolveRhsManip.start();

    int nx=rx.get_size(), nyc=ryc.get_size(), nyd=ryd.get_size();
    int nxsp=Hx_->get_size();
    assert(nxsp==nx);
    if(rhs_ == NULL) rhs_ = LinearAlgebraFactory::createVector(nx+nyc+nyd);

    nlp_->log->write("RHS KKT_SPARSE_XYcYd rx: ", rx,  hovIteration);
    nlp_->log->write("RHS KKT_SPARSE_XYcYd ryc:", ryc, hovIteration);
    nlp_->log->write("RHS KKT_SPARSE_XYcYd ryd:", ryd, hovIteration);

    //
    // form the rhs for the sparse linSys
    //
    rx.copyToStarting(*rhs_, 0);
    ryc.copyToStarting(*rhs_, nx);
    ryd.copyToStarting(*rhs_, nx+nyc);

    if(write_linsys_counter_>=0)
      csr_writer_.writeRhsToFile(*rhs_, write_linsys_counter_);

    nlp_->runStats.kkt.tmSolveRhsManip.stop();

    nlp_->runStats.kkt.tmSolveTriangular.start();
    //
    // solve
    //
    bool linsol_ok = linSys_->solve(*rhs_);
    nlp_->runStats.kkt.tmSolveTriangular.stop();
    nlp_->runStats.linsolv.end_linsolve();

    if(perf_report_) {
      nlp_->log->printf(hovSummary, "(summary for linear solver from KKT_SPARSE_XYcYd)\n%s",
			nlp_->runStats.linsolv.get_summary_last_solve().c_str());
    }

    if(write_linsys_counter_>=0)
      csr_writer_.writeSolToFile(*rhs_, write_linsys_counter_);

    if(false==linsol_ok) return false;

    nlp_->runStats.kkt.tmSolveRhsManip.start();

    //
    // unpack
    //
    rhs_->startingAtCopyToStartingAt(0,      dx,  0);
    rhs_->startingAtCopyToStartingAt(nx,     dyc, 0);
    rhs_->startingAtCopyToStartingAt(nx+nyc, dyd, 0);

    nlp_->log->write("SOL KKT_SPARSE_XYcYd dx: ", dx,  hovMatrices);
    nlp_->log->write("SOL KKT_SPARSE_XYcYd dyc:", dyc, hovMatrices);
    nlp_->log->write("SOL KKT_SPARSE_XYcYd dyd:", dyd, hovMatrices);

    nlp_->runStats.kkt.tmSolveRhsManip.stop();
    return true;
  }

  hiopLinSolverIndefSparse*
  hiopKKTLinSysCompressedSparseXYcYd::determineAndCreateLinsys(int nx, int neq, int nineq, int nnz)
  {

#//ifdef HIOP_SPARSE
    if(safe_mode_) {
      hiopLinSolverIndefSparseMA57* p = dynamic_cast<hiopLinSolverIndefSparseMA57*>(linSys_);
      if(p==NULL) {
        //we have a nopiv linear solver or linear solver has not been created yet
        delete linSys_;
        linSys_ = NULL;
      } else {
        return p;
      }
    }
//#endif

    if(NULL==linSys_) {
      int n = nx + neq + nineq;

      assert(nlp_->options->GetString("compute_mode")=="cpu");
      {
//#ifdef HIOP_SPARSE
        nlp_->log->printf(hovWarning,
			    "KKT_SPARSE_XYcYd linsys: MA57 size %d (%d cons) (safe_mode=%d)\n",
			    n, neq+nineq, safe_mode_);
        if(safe_mode_) {
          linSys_ = new hiopLinSolverIndefSparseMA57(n, nnz, nlp_);
        }else{
	  linSys_ = new hiopLinSolverIndefSparseMA57(n, nnz, nlp_);
	}
//#else
//        assert(0 && "Please provide a sparse indefinite linear package for HiOP with sparse linear system.");
//#endif
      }
    }
    return linSys_;
  }



  /* *************************************************************************
   * For class hiopKKTLinSysCompressedSparseXDYcYd
   * *************************************************************************
   */
  hiopKKTLinSysCompressedSparseXDYcYd::hiopKKTLinSysCompressedSparseXDYcYd(hiopNlpFormulation* nlp)
    : hiopKKTLinSysCompressedXDYcYd(nlp), linSys_{nullptr}, rhs_{nullptr},
      Hx_{nullptr}, Hd_{nullptr}, HessSp_{nullptr}, Jac_cSp_{nullptr}, Jac_dSp_{nullptr},
      write_linsys_counter_(-1), csr_writer_(nlp)
  {
    nlpSp_ = dynamic_cast<hiopNlpSparse*>(nlp_);
    assert(nlpSp_);
  }

  hiopKKTLinSysCompressedSparseXDYcYd::~hiopKKTLinSysCompressedSparseXDYcYd()
  {
    delete rhs_;
    delete linSys_;
    delete Hx_;
    delete Hd_;
  }

  bool hiopKKTLinSysCompressedSparseXDYcYd::update(const hiopIterate* iter,
					       const hiopVector* grad_f,
					       const hiopMatrix* Jac_c,
					       const hiopMatrix* Jac_d,
					       hiopMatrix* Hess)
  {
    if(!nlpSp_) { assert(false); return false; }

    nlp_->runStats.tmSolverInternal.start();
    nlp_->runStats.kkt.tmUpdateInit.start();

    iter_ = iter;
    grad_f_ = dynamic_cast<const hiopVectorPar*>(grad_f);
    Jac_c_ = Jac_c; Jac_d_ = Jac_d; Hess_=Hess;

    HessSp_ = dynamic_cast<hiopMatrixSymSparseTriplet*>(Hess);
    if(!HessSp_) { assert(false); return false; }

    Jac_cSp_ = dynamic_cast<const hiopMatrixSparseTriplet*>(Jac_c);
    if(!Jac_cSp_) { assert(false); return false; }

    Jac_dSp_ = dynamic_cast<const hiopMatrixSparseTriplet*>(Jac_d);
    if(!Jac_dSp_) { assert(false); return false; }

    long long nx = HessSp_->n(), nd=Jac_dSp_->m(), neq=Jac_cSp_->m(), nineq=Jac_dSp_->m();

    int nnz = HessSp_->numberOfNonzeros() + Jac_cSp_->numberOfNonzeros() + Jac_dSp_->numberOfNonzeros() + nd + nx + nd + neq + nineq;

    //
    //based on safe_mode_, decide whether to go with the nopiv (fast) or Bunch-Kaufman (stable) linear solve
    //
    linSys_ = determineAndCreateLinsys(nx, neq, nineq, nnz);

    //
    //update/compute KKT
    //
    //Dx (<-- log-barrier diagonal, for variable x
    assert(Dx_->get_local_size() == nx);
    Dx_->setToZero();
    Dx_->axdzpy_w_pattern(1.0, *iter->zl, *iter->sxl, nlp_->get_ixl());
    Dx_->axdzpy_w_pattern(1.0, *iter->zu, *iter->sxu, nlp_->get_ixu());
    nlp_->log->write("Dx in KKT", *Dx_, hovMatrices);

    // Dd=(Sdl)^{-1}Vu + (Sdu)^{-1}Vu
    Dd_->setToZero();
    Dd_->axdzpy_w_pattern(1.0, *iter_->vl, *iter_->sdl, nlp_->get_idl());
    Dd_->axdzpy_w_pattern(1.0, *iter_->vu, *iter_->sdu, nlp_->get_idu());
    nlp_->log->write("Dd in KKT", *Dd_, hovMatrices);

    hiopMatrixSparseTriplet& Msys = linSys_->sysMatrix();
    if(perf_report_) {
      nlp_->log->printf(hovSummary,
			"KKT_Sparse_XDYcYd linsys: Low-level linear system size: %d\n",
			Msys.n());
    }

    //
    //factorization + inertia correction if needed
    //
    const size_t max_ic_cor = 10;
    size_t num_ic_cor = 0;

    double delta_wx, delta_wd, delta_cc, delta_cd;
    if(!perturb_calc_->compute_initial_deltas(delta_wx, delta_wd, delta_cc, delta_cd)) {
      nlp_->log->printf(hovWarning,
			"KKT_Sparse_XDYcYd linsys: IC perturbation on new linsys failed.\n");
      return false;
    }

    nlp_->runStats.kkt.tmUpdateInit.stop();

    while(num_ic_cor<=max_ic_cor) {

      assert(delta_wx == delta_wd && "something went wrong with IC");
      assert(delta_cc == delta_cd && "something went wrong with IC");
      nlp_->log->printf(hovScalars,
			"KKT_Sparse_XDYcYd linsys: delta_w=%12.5e delta_c=%12.5e (ic %d)\n",
			delta_wx, delta_cc, num_ic_cor);

      //
      //the update of the linear system, including IC perturbations
      //
      nlp_->runStats.kkt.tmUpdateLinsys.start();

      Msys.setToZero();

      //
      // copy Jac and Hes to the full iterate matrix
      //
      long long dest_nnz_st{0};
      Msys.copyRowsFromSrcToDest(*HessSp_,  0,   nx,     0,          dest_nnz_st); dest_nnz_st += HessSp_->numberOfNonzeros();
      Msys.copyRowsFromSrcToDest(*Jac_cSp_, 0,   neq,    nx+nd,      dest_nnz_st); dest_nnz_st += Jac_cSp_->numberOfNonzeros();
      Msys.copyRowsFromSrcToDest(*Jac_dSp_, 0,   nineq,  nx+nd+neq,  dest_nnz_st); dest_nnz_st += Jac_dSp_->numberOfNonzeros();

      // minus identity matrix for slack variables
      Msys.copyDiagMatrixToSubBlock(-1., nineq, nx+nd+neq, nx, dest_nnz_st); dest_nnz_st += nineq;

	  //build the diagonal Hx = Dx + delta_wx
	  if(NULL == Hx_) {
	    Hx_ = LinearAlgebraFactory::createVector(nx); assert(Hx_);
	  }
	  Hx_->startingAtCopyFromStartingAt(0, *Dx_, 0);

	  //a good time to add the IC 'delta_wx' perturbation
	  Hx_->addConstant(delta_wx);

      Msys.copySubDiagonalEleFromVec(0, nx, *Hx_, dest_nnz_st); dest_nnz_st += nx;

	  //build the diagonal Hd = Dd + delta_wd
	  if(NULL == Hd_) {
	    Hd_ = LinearAlgebraFactory::createVector(nd); assert(Hd_);
	  }
	  Hd_->startingAtCopyFromStartingAt(0, *Dd_, 0);
	  Hd_->addConstant(delta_wd);
      Msys.copySubDiagonalEleFromVec(nx, nd, *Hd_, dest_nnz_st); dest_nnz_st += nd;

	  //add -delta_cc to diagonal block linSys starting at (nx+nd, nx+nd)
      Msys.copySubDiagonalEleFromConstant(nx+nd, neq, -delta_cc, dest_nnz_st); dest_nnz_st += neq;

	  //add -delta_cd to diagonal block linSys starting at (nx+nd+nineq, nx+nd+nineq)
      Msys.copySubDiagonalEleFromConstant(nx+nd+neq, nineq, -delta_cd, dest_nnz_st); dest_nnz_st += nineq;

	  /* we've just done
      *
      * [  H+Dx+delta_wx    0          Jc^T    Jd^T     ] [ dx]   [ rx_tilde ]
      * [    0          Dd+delta_wd     0       -I      ] [ dd]   [ rd_tilde ]
      * [    Jc             0        -delta_cc  0       ] [dyc] = [   ryc    ]
      * [    Jd            -I           0    -delta_cd  ] [dyd]   [   ryd    ]
	  */
	  nlp_->log->write("KKT_SPARSE_XDYcYd linsys:", Msys, hovMatrices);

      nlp_->runStats.kkt.tmUpdateLinsys.stop();

      //write matrix to file if requested
      if(nlp_->options->GetString("write_kkt") == "yes") write_linsys_counter_++;

      nlp_->runStats.linsolv.start_linsolve();
      nlp_->runStats.kkt.tmUpdateInnerFact.start();

      //factorization
      int n_neg_eig = linSys_->matrixChanged();

      nlp_->runStats.kkt.tmUpdateInnerFact.stop();

      if(Jac_cSp_->m()+Jac_dSp_->m()>0) {
        if(n_neg_eig < 0) {
          //matrix singular
          nlp_->log->printf(hovScalars,
			    "KKT_SPARSE_XYcYdlinsys is singular. Regularization will be attempted...\n");

          if(!perturb_calc_->compute_perturb_singularity(delta_wx, delta_wd, delta_cc, delta_cd)) {
            nlp_->log->printf(hovWarning,
			      "KKT_SPARSE_XYcYd linsys: computing singularity perturbation failed.\n");
            return false;
          }
        } else if(n_neg_eig != Jac_cSp_->m() + Jac_dSp_->m()) {
          //wrong inertia
          nlp_->log->printf(hovScalars,
			    "KKT_SPARSE_XYcYd linsys negative eigs mismatch: has %d expected %d.\n",
			    n_neg_eig,  Jac_cSp_->m()+Jac_dSp_->m());

          if(n_neg_eig < Jac_cSp_->m() + Jac_dSp_->m())
            nlp_->log->printf(hovWarning, "KKT_SPARSE_XYcYd linsys negative eigs abnormality\n");

          if(!perturb_calc_->compute_perturb_wrong_inertia(delta_wx, delta_wd, delta_cc, delta_cd)) {
            nlp_->log->printf(hovWarning,
			      "KKT_SPARSE_XYcYd linsys: computing inertia perturbation failed.\n");
            return false;
          }

        } else {
          //all is good
          break;
        }
      } else if(n_neg_eig != 0) {
        //correct for wrong intertia
        nlp_->log->printf(hovScalars,
			 "KKT_SPARSE_XYcYd linsys has wrong inertia (no constraints): factoriz "
			 "ret code/num negative eigs %d\n.", n_neg_eig);
        if(!perturb_calc_->compute_perturb_wrong_inertia(delta_wx, delta_wd, delta_cc, delta_cd)) {
          nlp_->log->printf(hovWarning,
			   "KKT_SPARSE_XYcYd linsys: computing inertia perturbation failed (2).\n");
          return false;
        }
      } else {
        //all is good
        break;
      }

      //will do an inertia correction
      num_ic_cor++;
      nlp_->runStats.kkt.nUpdateICCorr++;
    } // end of ic while

    if(num_ic_cor>max_ic_cor) {
      nlp_->log->printf(hovError,
			"KKT_SPARSE_XYcYd linsys: max number (%d) of inertia corrections reached.\n",
			max_ic_cor);
      return false;
    }
    nlp_->runStats.tmSolverInternal.stop();
    return true;
  }

  bool hiopKKTLinSysCompressedSparseXDYcYd::
  solveCompressed( hiopVector& rx, hiopVector& rd, hiopVector& ryc, hiopVector& ryd,
			       hiopVector& dx, hiopVector& dd, hiopVector& dyc, hiopVector& dyd)
  {
    if(!nlpSp_)   { assert(false); return false; }
    if(!HessSp_)  { assert(false); return false; }
    if(!Jac_cSp_) { assert(false); return false; }
    if(!Jac_dSp_) { assert(false); return false; }

    nlp_->runStats.kkt.tmSolveRhsManip.start();

    int nx=rx.get_size(), nd=rd.get_size(), nyc=ryc.get_size(), nyd=ryd.get_size();
    int nxsp=Hx_->get_size();
    assert(nxsp==nx);
    if(rhs_ == NULL) rhs_ = LinearAlgebraFactory::createVector(nx+nd+nyc+nyd);

    nlp_->log->write("RHS KKT_SPARSE_XDYcYd rx: ", rx,  hovIteration);
    nlp_->log->write("RHS KKT_SPARSE_XDYcYd rx: ", rd,  hovIteration);
    nlp_->log->write("RHS KKT_SPARSE_XDYcYd ryc:", ryc, hovIteration);
    nlp_->log->write("RHS KKT_SPARSE_XDYcYd ryd:", ryd, hovIteration);

    //
    // form the rhs for the sparse linSys
    //
    rx.copyToStarting(*rhs_, 0);
    rd.copyToStarting(*rhs_, nx);
    ryc.copyToStarting(*rhs_, nx+nd);
    ryd.copyToStarting(*rhs_, nx+nd+nyc);

    if(write_linsys_counter_>=0)
      csr_writer_.writeRhsToFile(*rhs_, write_linsys_counter_);

    nlp_->runStats.kkt.tmSolveRhsManip.stop();

    nlp_->runStats.kkt.tmSolveTriangular.start();

    //
    // solve
    //
    bool linsol_ok = linSys_->solve(*rhs_);
    nlp_->runStats.kkt.tmSolveTriangular.stop();
    nlp_->runStats.linsolv.end_linsolve();

    if(perf_report_) {
      nlp_->log->printf(hovSummary, "(summary for linear solver from KKT_SPARSE_XDYcYd)\n%s",
			nlp_->runStats.linsolv.get_summary_last_solve().c_str());
    }

    if(write_linsys_counter_>=0)
      csr_writer_.writeSolToFile(*rhs_, write_linsys_counter_);

    if(false==linsol_ok) return false;

    nlp_->runStats.kkt.tmSolveRhsManip.start();

    //
    // unpack
    //
    rhs_->startingAtCopyToStartingAt(0,         dx,  0);
    rhs_->startingAtCopyToStartingAt(nx,        dd,  0);
    rhs_->startingAtCopyToStartingAt(nx+nd,     dyc, 0);
    rhs_->startingAtCopyToStartingAt(nx+nd+nyc, dyd, 0);

    nlp_->log->write("SOL KKT_SPARSE_XDYcYd dx: ", dx,  hovMatrices);
    nlp_->log->write("SOL KKT_SPARSE_XDYcYd dd: ", dd,  hovMatrices);
    nlp_->log->write("SOL KKT_SPARSE_XDYcYd dyc:", dyc, hovMatrices);
    nlp_->log->write("SOL KKT_SPARSE_XDYcYd dyd:", dyd, hovMatrices);

    nlp_->runStats.kkt.tmSolveRhsManip.stop();
    return true;
  }

  hiopLinSolverIndefSparse*
  hiopKKTLinSysCompressedSparseXDYcYd::determineAndCreateLinsys(int nx, int neq, int nineq, int nnz)
  {

//#ifdef HIOP_SPARSE
    if(safe_mode_) {
      hiopLinSolverIndefSparseMA57* p = dynamic_cast<hiopLinSolverIndefSparseMA57*>(linSys_);
      if(p==NULL) {
        //we have a nopiv linear solver or linear solver has not been created yet
        delete linSys_;
        linSys_ = NULL;
      } else {
        return p;
      }
    }
//#endif

    if(NULL==linSys_) {
      int n = nx + nineq + neq + nineq;

      assert(nlp_->options->GetString("compute_mode")=="cpu");
      {
//#ifdef HIOP_SPARSE
        nlp_->log->printf(hovWarning,
			    "KKT_SPARSE_XYcYd linsys: MA57 size %d (%d cons) (safe_mode=%d)\n",
			    n, neq+nineq, safe_mode_);
        if(safe_mode_) {
          linSys_ = new hiopLinSolverIndefSparseMA57(n, nnz, nlp_);
        }else{
	  linSys_ = new hiopLinSolverIndefSparseMA57(n, nnz, nlp_);
	}
//#else
 //       assert(0 && "Please provide a sparse indefinite linear package for HiOP with sparse linear system.");
//#endif
      }
    }
    return linSys_;
  }





} // end of namespace
