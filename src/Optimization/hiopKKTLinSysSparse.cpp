// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// LLNL-CODE-742473. All rights reserved.
//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop. HiOp
// is released under the BSD 3-clause license (https://opensource.org/licenses/BSD-3-Clause).
// Please also read "Additional BSD Notice" below.
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

#ifdef HIOP_SPARSE
#ifdef HIOP_USE_COINHSL
#include "hiopLinSolverIndefSparseMA57.hpp"
#endif
#ifdef HIOP_USE_STRUMPACK
#include "hiopLinSolverSparseSTRUMPACK.hpp"
#endif
#ifdef HIOP_USE_PARDISO
#include "hiopLinSolverSparsePARDISO.hpp"
#endif
#ifdef HIOP_USE_CUSOLVER
#include "hiopLinSolverSparseCUSOLVER.hpp"
#endif
#endif

namespace hiop
{

  /* *************************************************************************
   * For class hiopKKTLinSysCompressedSparseXYcYd
   * *************************************************************************
   */
  hiopKKTLinSysCompressedSparseXYcYd::hiopKKTLinSysCompressedSparseXYcYd(hiopNlpFormulation* nlp)
    : hiopKKTLinSysCompressedXYcYd(nlp), rhs_(NULL),
      Hx_(NULL), HessSp_(NULL), Jac_cSp_(NULL), Jac_dSp_(NULL),
      write_linsys_counter_(-1), csr_writer_(nlp)
  {
    nlpSp_ = dynamic_cast<hiopNlpSparse*>(nlp_);
    assert(nlpSp_);
  }

  hiopKKTLinSysCompressedSparseXYcYd::~hiopKKTLinSysCompressedSparseXYcYd()
  {
    delete rhs_;
    delete Hx_;
  }

  bool hiopKKTLinSysCompressedSparseXYcYd::build_kkt_matrix(const double& delta_wx,
                                                            const double& delta_wd,
                                                            const double& delta_cc,
                                                            const double& delta_cd)
  {
    HessSp_ = dynamic_cast<hiopMatrixSparse*>(Hess_);
    if(!HessSp_) { assert(false); return false; }

    Jac_cSp_ = dynamic_cast<const hiopMatrixSparse*>(Jac_c_);
    if(!Jac_cSp_) { assert(false); return false; }

    Jac_dSp_ = dynamic_cast<const hiopMatrixSparse*>(Jac_d_);
    if(!Jac_dSp_) { assert(false); return false; }

    size_type nx = HessSp_->n(), neq=Jac_cSp_->m(), nineq=Jac_dSp_->m();
    int nnz = HessSp_->numberOfNonzeros() + Jac_cSp_->numberOfNonzeros() + Jac_dSp_->numberOfNonzeros();
    nnz += nx + neq + nineq;

    linSys_ = determineAndCreateLinsys(nx, neq, nineq, nnz);
        
    auto* linSys = dynamic_cast<hiopLinSolverSymSparse*> (linSys_);
    assert(linSys);

    auto* Msys = dynamic_cast<hiopMatrixSparseTriplet*>(linSys->sysMatrix());
    assert(Msys);
    
    if(perf_report_) {
      nlp_->log->printf(hovSummary,
			"KKT_Sparse_XYcYd linsys: Low-level linear system size: %d\n",
			Msys->n());
    }

    // update linSys system matrix, including IC perturbations
    {
      nlp_->runStats.kkt.tmUpdateLinsys.start();
      
      Msys->setToZero();

      // copy Jac and Hes to the full iterate matrix
      size_type dest_nnz_st{0};
      Msys->copyRowsBlockFrom(*HessSp_,  0,   nx,     0,      dest_nnz_st);
      dest_nnz_st += HessSp_->numberOfNonzeros();
      Msys->copyRowsBlockFrom(*Jac_cSp_, 0,   neq,    nx,     dest_nnz_st);
      dest_nnz_st += Jac_cSp_->numberOfNonzeros();
      Msys->copyRowsBlockFrom(*Jac_dSp_, 0,   nineq,  nx+neq, dest_nnz_st);
      dest_nnz_st += Jac_dSp_->numberOfNonzeros();

      //build the diagonal Hx = Dx + delta_wx
      if(NULL == Hx_) {
        Hx_ = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), nx);
        assert(Hx_);
      }
      Hx_->startingAtCopyFromStartingAt(0, *Dx_, 0);

      //a good time to add the IC 'delta_wx' perturbation
      Hx_->addConstant(delta_wx);

      Msys->copySubDiagonalFrom(0, nx, *Hx_, dest_nnz_st); dest_nnz_st += nx;

      //add -delta_cc to diagonal block linSys starting at (nx, nx)
      Msys->setSubDiagonalTo(nx, neq, -delta_cc, dest_nnz_st); dest_nnz_st += neq;

      /* we've just done above the (1,1) and (2,2) blocks of
       *
       * [ Hx+Dxd+delta_wx*I           Jcd^T          Jdd^T   ]
       * [  Jcd                       -delta_cc*I     0       ]
       * [  Jdd                        0              M_{33} ]
       *
       * where
       * M_{33} = - (Dd+delta_wd)*I^{-1} - delta_cd*I = - Dd_inv - delta_cd*I is performed below
       */

      // Dd = (Sdl)^{-1}Vu + (Sdu)^{-1}Vu + delta_wd * I
      Dd_inv_->setToConstant(delta_wd);
      Dd_inv_->axdzpy_w_pattern(1.0, *iter_->vl, *iter_->sdl, nlp_->get_idl());
      Dd_inv_->axdzpy_w_pattern(1.0, *iter_->vu, *iter_->sdu, nlp_->get_idu());

#ifdef HIOP_DEEPCHECKS
      assert(true==Dd_inv_->allPositive());
#endif
      Dd_inv_->invert();
      Dd_inv_->addConstant(delta_cd);

      Msys->copySubDiagonalFrom(nx+neq, nineq, *Dd_inv_, dest_nnz_st, -1); dest_nnz_st += nineq;


      nlp_->log->write("KKT_SPARSE_XYcYd linsys:", *Msys, hovMatrices);
      nlp_->runStats.kkt.tmUpdateLinsys.stop();
    } // end of update of the linear system

    //write matrix to file if requested
    if(nlp_->options->GetString("write_kkt") == "yes") {
      write_linsys_counter_++;
    }
    if(write_linsys_counter_>=0) {
      csr_writer_.writeMatToFile(*Msys, write_linsys_counter_, nx, neq, nineq);
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
    if(rhs_ == NULL) {
      rhs_ = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"),
                                                 nx+nyc+nyd);
    }

    nlp_->log->write("RHS KKT_SPARSE_XYcYd rx: ", rx,  hovIteration);
    nlp_->log->write("RHS KKT_SPARSE_XYcYd ryc:", ryc, hovIteration);
    nlp_->log->write("RHS KKT_SPARSE_XYcYd ryd:", ryd, hovIteration);

    //
    // form the rhs for the sparse linSys
    //
    rx.copyToStarting(*rhs_, 0);
    ryc.copyToStarting(*rhs_, nx);
    ryd.copyToStarting(*rhs_, nx+nyc);

    if(write_linsys_counter_>=0) {
      csr_writer_.writeRhsToFile(*rhs_, write_linsys_counter_);
    }
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

    if(write_linsys_counter_>=0) {
      csr_writer_.writeSolToFile(*rhs_, write_linsys_counter_);
    }
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

  hiopLinSolverSymSparse*
  hiopKKTLinSysCompressedSparseXYcYd::determineAndCreateLinsys(int nx, int neq, int nineq, int nnz)
  {
    if(nullptr==linSys_) {
      int n = nx + neq + nineq;

      if(nlp_->options->GetString("compute_mode") == "cpu")
      {
        auto linear_solver = nlp_->options->GetString("linear_solver_sparse");

        if(linear_solver == "ma57" || linear_solver == "auto") {
#ifdef HIOP_USE_COINHSL
          nlp_->log->printf(hovScalars,
                            "KKT_SPARSE_XYcYd linsys: alloc MA57 with matrix size %d (%d cons)\n",
                            n, neq+nineq);
          linSys_ = new hiopLinSolverIndefSparseMA57(n, nnz, nlp_);
#endif // HIOP_USE_COINHSL
        }

        if( (nullptr == linSys_ && linear_solver == "auto") || linear_solver == "pardiso") {
          //ma57 is not available or user requested pardiso
#ifdef HIOP_USE_PARDISO
          nlp_->log->printf(hovScalars,
                            "KKT_SPARSE_XYcYd linsys: alloc PARDISO with matrix size %d (%d cons)\n",
                            n, neq+nineq);
          linSys_ = new hiopLinSolverIndefSparsePARDISO(n, nnz, nlp_);
#endif  // HIOP_USE_PARDISO          
        }

        if( (nullptr == linSys_ && linear_solver == "auto") || linear_solver == "strumpack") {
          //ma57 and pardiso are not available or user requested strumpack
#ifdef HIOP_USE_STRUMPACK              
          nlp_->log->printf(hovScalars,
                            "KKT_SPARSE_XYcYd linsys: alloc STRUMPACK with matrix size %d (%d cons)\n",
                            n, neq+nineq);
          hiopLinSolverIndefSparseSTRUMPACK *p = new hiopLinSolverIndefSparseSTRUMPACK(n, nnz, nlp_);
          p->setFakeInertia(neq + nineq);
          linSys_ = p;        
#endif  // HIOP_USE_STRUMPACK        
        }
      } else {
        // on device
#ifdef HIOP_USE_CUSOLVER

        hiopLinSolverIndefSparseCUSOLVER *p = new hiopLinSolverIndefSparseCUSOLVER(n, nnz, nlp_);

        //print it as a warning if safe mode is on
        auto verbosity = hovScalars;
        if(safe_mode_) verbosity  = hovWarning;
        nlp_->log->printf(verbosity,
                          "KKT_SPARSE_XYcYd linsys: alloc CUSOLVER size %d (%d cons) (safe_mode=%d)\n",
                          n, neq+nineq, safe_mode_);
        linSys_ = p;
#elif defined(HIOP_USE_STRUMPACK)        
        hiopLinSolverIndefSparseSTRUMPACK *p = new hiopLinSolverIndefSparseSTRUMPACK(n, nnz, nlp_);

        //print it as a warning if safe mode is on
        auto verbosity = safe_mode_ ? hovWarning : hovScalars;

        nlp_->log->printf(verbosity,
                          "KKT_SPARSE_XYcYd linsys: alloc STRUMPACK size %d (%d cons) (safe_mode=%d)\n",
                          n, neq+nineq, safe_mode_);

        p->setFakeInertia(neq + nineq);
        linSys_ = p;
#elif  defined(HIOP_USE_COINHSL)
        nlp_->log->printf(hovScalars,
                          "KKT_SPARSE_XYcYd linsys: alloc MA57 on CPU size %d (%d cons)\n",
                          n, neq+nineq);                             
        linSys_ = new hiopLinSolverIndefSparseMA57(n, nnz, nlp_);

        if(NULL == linSys_) {
#ifdef HIOP_USE_PARDISO
          nlp_->log->printf(hovScalars,
                            "KKT_SPARSE_XYcYd linsys: alloc PARDISO on CPU size %d (%d cons)\n",
                            n, neq+nineq);                             
          linSys_ = new hiopLinSolverIndefSparseMA57(n, nnz, nlp_);
#endif // HIOP_USE_PARDISO
        }
#endif // HIOP_USE_CUSOLVER
      }
      assert(linSys_&& "KKT_SPARSE_XYcYd linsys: cannot instantiate backend linear solver");
    }
    return dynamic_cast<hiopLinSolverSymSparse*> (linSys_);
  }



  /* *************************************************************************
   * For class hiopKKTLinSysCompressedSparseXDYcYd
   * *************************************************************************
   */
  hiopKKTLinSysCompressedSparseXDYcYd::hiopKKTLinSysCompressedSparseXDYcYd(hiopNlpFormulation* nlp)
    : hiopKKTLinSysCompressedXDYcYd(nlp), rhs_{nullptr},
      Hx_{nullptr}, Hd_{nullptr}, HessSp_{nullptr}, Jac_cSp_{nullptr}, Jac_dSp_{nullptr},
      write_linsys_counter_(-1), csr_writer_(nlp)
  {
    nlpSp_ = dynamic_cast<hiopNlpSparse*>(nlp_);
    assert(nlpSp_);
  }

  hiopKKTLinSysCompressedSparseXDYcYd::~hiopKKTLinSysCompressedSparseXDYcYd()
  {
    delete rhs_;
    delete Hx_;
    delete Hd_;
  }

  bool hiopKKTLinSysCompressedSparseXDYcYd::build_kkt_matrix(const double& delta_wx,
                                                             const double& delta_wd,
                                                             const double& delta_cc,
                                                             const double& delta_cd)
  {
    HessSp_ = dynamic_cast<hiopMatrixSymSparseTriplet*>(Hess_);
    if(!HessSp_) { assert(false); return false; }

    Jac_cSp_ = dynamic_cast<const hiopMatrixSparseTriplet*>(Jac_c_);
    if(!Jac_cSp_) { assert(false); return false; }

    Jac_dSp_ = dynamic_cast<const hiopMatrixSparseTriplet*>(Jac_d_);
    if(!Jac_dSp_) { assert(false); return false; }

    size_type nx = HessSp_->n(), nd=Jac_dSp_->m(), neq=Jac_cSp_->m(), nineq=Jac_dSp_->m();
    int nnz = HessSp_->numberOfNonzeros() + Jac_cSp_->numberOfNonzeros() + Jac_dSp_->numberOfNonzeros() + nd + nx + nd + neq + nineq;

    linSys_ = determineAndCreateLinsys(nx, neq, nineq, nnz);
    
    auto* linSys = dynamic_cast<hiopLinSolverSymSparse*> (linSys_);
    assert(linSys);

    auto* Msys = dynamic_cast<hiopMatrixSparseTriplet*>(linSys->sysMatrix());
    assert(Msys);
    if(perf_report_) {
      nlp_->log->printf(hovSummary,
			"KKT_SPARSE_XDYcYd linsys: Low-level linear system size: %d\n",
			Msys->n());
    }

    // update linSys system matrix, including IC perturbations
    {
      nlp_->runStats.kkt.tmUpdateLinsys.start();

      Msys->setToZero();

      // copy Jac and Hes to the full iterate matrix
      size_type dest_nnz_st{0};
      Msys->copyRowsBlockFrom(*HessSp_,  0,   nx,     0,          dest_nnz_st);
      dest_nnz_st += HessSp_->numberOfNonzeros();
      Msys->copyRowsBlockFrom(*Jac_cSp_, 0,   neq,    nx+nd,      dest_nnz_st);
      dest_nnz_st += Jac_cSp_->numberOfNonzeros();
      Msys->copyRowsBlockFrom(*Jac_dSp_, 0,   nineq,  nx+nd+neq,  dest_nnz_st);
      dest_nnz_st += Jac_dSp_->numberOfNonzeros();

      // minus identity matrix for slack variables
      Msys->copyDiagMatrixToSubblock(-1., nx+nd+neq, nx, dest_nnz_st, nineq);
      dest_nnz_st += nineq;

      //build the diagonal Hx = Dx + delta_wx
      if(NULL == Hx_) {
        Hx_ = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), nx);
        assert(Hx_);
      }
      Hx_->startingAtCopyFromStartingAt(0, *Dx_, 0);

      //a good time to add the IC 'delta_wx' perturbation
      Hx_->addConstant(delta_wx);

      Msys->copySubDiagonalFrom(0, nx, *Hx_, dest_nnz_st);
      dest_nnz_st += nx;

      //build the diagonal Hd = Dd + delta_wd
      if(NULL == Hd_) {
        Hd_ = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), nd);
        assert(Hd_);
      }
      Hd_->startingAtCopyFromStartingAt(0, *Dd_, 0);
      Hd_->addConstant(delta_wd);
      Msys->copySubDiagonalFrom(nx, nd, *Hd_, dest_nnz_st);
      dest_nnz_st += nd;

      //add -delta_cc to diagonal block linSys starting at (nx+nd, nx+nd)
      Msys->setSubDiagonalTo(nx+nd, neq, -delta_cc, dest_nnz_st);
      dest_nnz_st += neq;

      //add -delta_cd to diagonal block linSys starting at (nx+nd+neq, nx+nd+neq)
      Msys->setSubDiagonalTo(nx+nd+neq, nineq, -delta_cd, dest_nnz_st);
      dest_nnz_st += nineq;

      /* we've just done
      *
      * [  H+Dx+delta_wx    0          Jc^T    Jd^T     ] [ dx]   [ rx_tilde ]
      * [    0          Dd+delta_wd     0       -I      ] [ dd]   [ rd_tilde ]
      * [    Jc             0        -delta_cc  0       ] [dyc] = [   ryc    ]
      * [    Jd            -I           0    -delta_cd  ] [dyd]   [   ryd    ]
      */
      nlp_->log->write("KKT_SPARSE_XDYcYd linsys:", *Msys, hovMatrices);
      nlp_->runStats.kkt.tmUpdateLinsys.stop();
    }

    //write matrix to file if requested
    if(nlp_->options->GetString("write_kkt") == "yes") {
      write_linsys_counter_++;
    }
    if(write_linsys_counter_>=0) {
      csr_writer_.writeMatToFile(*Msys, write_linsys_counter_, nx, neq, nineq);
    }

    return true;
  }

  bool hiopKKTLinSysCompressedSparseXDYcYd::
  solveCompressed(hiopVector& rx, hiopVector& rd, hiopVector& ryc, hiopVector& ryd,
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
    if(rhs_ == NULL) {
      rhs_ = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"),
                                                 nx+nd+nyc+nyd);
    }

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

    if(write_linsys_counter_>=0) {
      csr_writer_.writeRhsToFile(*rhs_, write_linsys_counter_);
    }
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

    if(write_linsys_counter_>=0) {
      csr_writer_.writeSolToFile(*rhs_, write_linsys_counter_);
    }
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

  hiopLinSolverSymSparse*
  hiopKKTLinSysCompressedSparseXDYcYd::determineAndCreateLinsys(int nx, int neq, int nineq, int nnz)
  {
    if(nullptr==linSys_) {
      int n = nx + nineq + neq + nineq;

      if(nlp_->options->GetString("compute_mode")=="cpu")
      {
        nlp_->log->printf(hovWarning,
                          "KKT_SPARSE_XDYcYd linsys: alloc sparse solver with matrix size %d (%d cons)\n",
                          n, neq+nineq);

        auto linear_solver = nlp_->options->GetString("linear_solver_sparse");

        if(linear_solver == "ma57" || linear_solver == "auto") {
#ifdef HIOP_USE_COINHSL
          linSys_ = new hiopLinSolverIndefSparseMA57(n, nnz, nlp_);
#endif // HIOP_USE_COINHSL
        }

        if( (nullptr == linSys_ && linear_solver == "auto") || linear_solver == "pardiso") {
          //ma57 is not available or user requested pardiso
#ifdef HIOP_USE_PARDISO
          linSys_ = new hiopLinSolverIndefSparsePARDISO(n, nnz, nlp_);
#endif  // HIOP_USE_PARDISO          
        }

        if( (nullptr == linSys_ && linear_solver == "auto") || linear_solver == "strumpack") {
          //ma57 is not available or user requested strumpack
#ifdef HIOP_USE_STRUMPACK              
          hiopLinSolverIndefSparseSTRUMPACK *p = new hiopLinSolverIndefSparseSTRUMPACK(n, nnz, nlp_);
          p->setFakeInertia(neq + nineq);
          linSys_ = p;        
#endif  // HIOP_USE_STRUMPACK        
        }
      } else {
      //we are on the GPU. Our first choice is always cuSolver
#if  defined(HIOP_USE_CUSOLVER)        
        hiopLinSolverIndefSparseCUSOLVER *p = new hiopLinSolverIndefSparseCUSOLVER(n, nnz, nlp_);
        auto verbosity = hovScalars;
        nlp_->log->printf(verbosity,
                          "KKT_SPARSE_XDYcYd linsys: alloc CUSOLVER size %d (%d cons) (safe_mode=%d)\n",
                          n,
                          neq+nineq,
                          safe_mode_);
        if(safe_mode_) verbosity  = hovWarning;
        linSys_ = p;
#else
#if defined(HIOP_USE_STRUMPACK)
        hiopLinSolverIndefSparseSTRUMPACK *p = new hiopLinSolverIndefSparseSTRUMPACK(n, nnz, nlp_);
        auto verbosity = hovScalars;
        nlp_->log->printf(verbosity,
            "KKT_SPARSE_XDYcYd linsys: alloc STRUMPACK size %d (%d cons) (safe_mode=%d)\n",
            n, neq+nineq, safe_mode_);
        if(safe_mode_) verbosity  = hovWarning;

        p->setFakeInertia(neq + nineq);
        linSys_ = p;
#else
        //Return nullptr (and assert) if a GPU sparse linear solver is not present
        assert(linSys_!=nullptr &&
               "HiOp was built without a sparse linear solver for GPU/device and cannot run on the "
               "device as instructed by the 'compute_mode' option. Change the 'compute_mode' to "
               "'cpu' (from hiopKKTLinSysCompressedSparseXDYcYd)"); 
        return nullptr;
#endif
#ifdef HIOP_USE_COINHSL
        nlp_->log->printf(hovScalars,
                          "KKT_SPARSE_XDYcYd linsys: alloc MA57 on CPU size %d (%d cons)\n",
                          n,
                          neq+nineq);                             
        linSys_ = new hiopLinSolverIndefSparseMA57(n, nnz, nlp_);
#endif // HIOP_USE_COINHSL

        if(NULL == linSys_) {
#ifdef HIOP_USE_PARDISO
          nlp_->log->printf(hovScalars,
                            "KKT_SPARSE_XYcYd linsys: alloc PARDISO on CPU size %d (%d cons)\n",
                            n,
                            neq+nineq);                             
          linSys_ = new hiopLinSolverIndefSparseMA57(n, nnz, nlp_);
#endif // HIOP_USE_PARDISO
        }
#endif // HIOP_USE_CUSOLVER/STRUMPACK
      // Add interface to cuSolver/KLU linear solver
      }
      assert(linSys_&& "KKT_SPARSE_XDYcYd linsys: cannot instantiate backend linear solver");
    }
      return dynamic_cast<hiopLinSolverSymSparse*> (linSys_);
  }



  /* *************************************************************************
   * For class hiopKKTLinSysSparseFull
   * *************************************************************************
   */
  hiopKKTLinSysSparseFull::hiopKKTLinSysSparseFull(hiopNlpFormulation* nlp)
    : hiopKKTLinSysFull(nlp), rhs_(nullptr),
      Hx_(nullptr), Hd_(nullptr), HessSp_(nullptr), Jac_cSp_(nullptr), Jac_dSp_(nullptr),
      write_linsys_counter_(-1), csr_writer_(nlp)
  {
    nlpSp_ = dynamic_cast<hiopNlpSparse*>(nlp_);
    assert(nlpSp_);
  }

  hiopKKTLinSysSparseFull::~hiopKKTLinSysSparseFull()
  {
    delete rhs_;
    delete Hx_;
    delete Hd_;
  }

  hiopLinSolverNonSymSparse*
  hiopKKTLinSysSparseFull::determineAndCreateLinsys(const int &n, const int &n_con, const int &nnz)
  {
    if(NULL==linSys_) {
#ifdef HIOP_USE_CUSOLVER
      nlp_->log->printf(hovWarning,
                        "KKT_SPARSE_FULL_KKT linsys: alloc CUSOLVER size %d (%d cons) (safe_mode=%d)\n",
                        n,
                        n_con,
                        safe_mode_);
      hiopLinSolverNonSymSparseCUSOLVER *p = new hiopLinSolverNonSymSparseCUSOLVER(n, nnz, nlp_);
      linSys_ = p;
#elif HIOP_USE_PARDISO
      nlp_->log->printf(hovWarning,
                        "KKT_SPARSE_FULL_KKT linsys: alloc PARDISO size %d (%d cons) (safe_mode=%d)\n",
                        n,
                        n_con,
                        safe_mode_);
      hiopLinSolverNonSymSparsePARDISO *p = new hiopLinSolverNonSymSparsePARDISO(n, nnz, nlp_);
      p->setFakeInertia(n_con);
      linSys_ = p;
#elif defined(HIOP_USE_STRUMPACK)

      hiopLinSolverNonSymSparseSTRUMPACK *p = new hiopLinSolverNonSymSparseSTRUMPACK(n, nnz, nlp_);
      nlp_->log->printf(hovWarning,
                        "KKT_SPARSE_FULL_KKT linsys: alloc STRUMPACK size %d (%d cons) (safe_mode=%d)\n",
                        n,
                        n_con,
                        safe_mode_);
      p->setFakeInertia(n_con);
      linSys_ = p;
#endif // CUSOLVER
      if(NULL==linSys_) {
        nlp_->log->printf(hovError,
                          "KKT_SPARSE_FULL_KKT linsys: cannot instantiate backend linear solver "
                          "because HIOP was not built with STRUMPACK or PARDISO.\n");
        assert(false);
        return nullptr;
      }
    }
    return dynamic_cast<hiopLinSolverNonSymSparse*> (linSys_);
  }

  bool hiopKKTLinSysSparseFull::build_kkt_matrix(const double& delta_wx,
                                                 const double& delta_wd,
                                                 const double& delta_cc,
                                                 const double& delta_cd)
  {
    HessSp_ = dynamic_cast<hiopMatrixSymSparseTriplet*>(Hess_);
    if(!HessSp_) { assert(false); return false; }

    Jac_cSp_ = dynamic_cast<const hiopMatrixSparseTriplet*>(Jac_c_);
    if(!Jac_cSp_) { assert(false); return false; }

    Jac_dSp_ = dynamic_cast<const hiopMatrixSparseTriplet*>(Jac_d_);
    if(!Jac_dSp_) { assert(false); return false; }

    size_type nx = HessSp_->n(); 
    size_type nd = Jac_dSp_->m();
    size_type neq = Jac_cSp_->m();
    size_type nineq=Jac_dSp_->m();
    size_type ndl = nlp_->m_ineq_low();
    size_type ndu = nlp_->m_ineq_upp();
    size_type nxl = nlp_->n_low();
    size_type nxu = nlp_->n_upp();

    // note that hess may be saved as a triangular matrix
    int n1st = 0;
    int n2st = nx + neq + nineq;
    int n3st = n2st + nd;
    int n4st = n3st + ndl + ndu + nxl + nxu; // shortcut for each subbloock
    int n = n4st + ndl + ndu + nxl + nxu;
    int n_reg = n3st;

    int required_num_neg_eig = neq+nineq;
    int nnz = HessSp_->numberOfNonzeros() + HessSp_->numberOfOffDiagNonzeros()
              + 2*Jac_cSp_->numberOfNonzeros() + 2*Jac_dSp_->numberOfNonzeros()
              + 2*(nd + ndl + ndu + nxl + nxu + ndl + ndu + nxl + nxu)
              + ndl + ndu + nxl + nxu
              + n_reg;

    linSys_ = determineAndCreateLinsys(n, required_num_neg_eig, nnz);

    auto* linSys = dynamic_cast<hiopLinSolverNonSymSparse*> (linSys_);
    assert(linSys);   

    auto* Msys = dynamic_cast<hiopMatrixSparseTriplet*>(linSys->sysMatrix());
    assert(Msys);
    if(perf_report_) {
      nlp_->log->printf(hovSummary,
			"KKT_SPARSE_FULL linsys: Low-level linear system size: %d\n",
			Msys->n());
    }

    // update linSys system matrix, including IC perturbations
    {
      nlp_->runStats.kkt.tmUpdateLinsys.start();

      Msys->setToZero();

      // copy Jac and Hes to the full iterate matrix, use Dx_ and Dd_ as temp vector
      size_type dest_nnz_st{0};

      // H is triangular
      // [   H   Jc^T  Jd^T | 0 |  0   0  -I   I   |  0   0   0   0  ] [  dx]   [    rx    ]
      Msys->copySubmatrixFrom(*HessSp_, 0, 0, dest_nnz_st, true);
      dest_nnz_st += HessSp_->numberOfOffDiagNonzeros();
      Msys->copySubmatrixFromTrans(*HessSp_, 0, 0, dest_nnz_st);
      dest_nnz_st += HessSp_->numberOfNonzeros();

      Msys->copySubmatrixFromTrans(*Jac_cSp_, 0, nx, dest_nnz_st);
      dest_nnz_st += Jac_cSp_->numberOfNonzeros();
      Msys->copySubmatrixFromTrans(*Jac_dSp_, 0, nx+neq, dest_nnz_st);
      dest_nnz_st += Jac_dSp_->numberOfNonzeros();
      Msys->setSubmatrixToConstantDiag_w_colpattern(-1., 0, n3st+ndl+ndu, dest_nnz_st, nxl, nlp_->get_ixl());
      dest_nnz_st += nxl;
      Msys->setSubmatrixToConstantDiag_w_colpattern(1., 0, n3st+ndl+ndu+nxl, dest_nnz_st, nxu, nlp_->get_ixu());
      dest_nnz_st += nxu;

      // [  Jc    0     0   | 0 |  0   0   0   0   |  0   0   0   0  ] [ dyc] = [   ryc    ]
      Msys->copySubmatrixFrom(*Jac_cSp_, nx, 0, dest_nnz_st);
      dest_nnz_st += Jac_cSp_->numberOfNonzeros();

      // [  Jd    0     0   |-I |  0   0   0   0   |  0   0   0   0  ] [ dyd]   [   ryd    ]
      Msys->copySubmatrixFrom(*Jac_dSp_, nx+neq, 0, dest_nnz_st);
      dest_nnz_st += Jac_dSp_->numberOfNonzeros();
      Msys->copyDiagMatrixToSubblock(-1., nx+neq, n2st, dest_nnz_st, nd);
      dest_nnz_st += nd;

      // [  0     0    -I   | 0 |  -I  I   0   0   |  0   0   0   0  ] [  dd]   [    rd    ]
      Msys->copyDiagMatrixToSubblock(-1., n2st, nx+neq, dest_nnz_st, nd);
      dest_nnz_st += nd;
      Msys->setSubmatrixToConstantDiag_w_colpattern(-1., n2st, n3st, dest_nnz_st, ndl, nlp_->get_idl());
      dest_nnz_st += ndl;
      Msys->setSubmatrixToConstantDiag_w_colpattern(1., n2st, n3st+ndl, dest_nnz_st, ndu, nlp_->get_idu());
      dest_nnz_st += ndu;

      // part3
      // [  0     0     0   |-I |  0   0   0   0   |  I   0   0   0  ] [ dvl]   [   rvl    ]
      Msys->setSubmatrixToConstantDiag_w_rowpattern(-1., n3st, n2st, dest_nnz_st, ndl, nlp_->get_idl());
      dest_nnz_st += ndl;
      Msys->copyDiagMatrixToSubblock(1., n3st, n4st, dest_nnz_st, ndl);
      dest_nnz_st += ndl;

      // [  0     0     0   | I |  0   0   0   0   |  0   I   0   0  ] [ dvu]   [   rvu    ]
      Msys->setSubmatrixToConstantDiag_w_rowpattern(1., n3st+ndl, n2st, dest_nnz_st, ndu, nlp_->get_idu());
      dest_nnz_st += ndu;
      Msys->copyDiagMatrixToSubblock(1., n3st+ndl, n4st+ndl, dest_nnz_st, ndu);
      dest_nnz_st += ndu;

      // [ -I     0     0   | 0 |  0   0   0   0   |  0   0   I   0  ] [ dzl]   [   rzl    ]
      Msys->setSubmatrixToConstantDiag_w_rowpattern(-1., n3st+ndl+ndu, 0, dest_nnz_st, nxl, nlp_->get_ixl());
      dest_nnz_st += nxl;
      Msys->copyDiagMatrixToSubblock(1., n3st+ndl+ndu, n4st+ndl+ndu, dest_nnz_st, nxl);
      dest_nnz_st += nxl;

      // [  I     0     0   | 0 |  0   0   0   0   |  0   0   0   I  ] [ dzu]   [   rzu    ]
      Msys->setSubmatrixToConstantDiag_w_rowpattern(1., n3st+ndl+ndu+nxl, 0, dest_nnz_st, nxu, nlp_->get_ixu());
      dest_nnz_st += nxu;
      Msys->copyDiagMatrixToSubblock(1., n3st+ndl+ndu+nxl, n4st+ndl+ndu+nxl, dest_nnz_st, nxu);
      dest_nnz_st += nxu;

      // part 4
      // [  0     0     0   | 0 | Sl^d 0   0   0   | Vl   0   0   0  ] [dsdl]   [  rsdl    ]
      Msys->copyDiagMatrixToSubblock_w_pattern(*iter_->sdl, n4st, n3st, dest_nnz_st, ndl, nlp_->get_idl());
      dest_nnz_st += ndl;
      Msys->copyDiagMatrixToSubblock_w_pattern(*iter_->vl, n4st, n4st, dest_nnz_st, ndl, nlp_->get_idl());
      dest_nnz_st += ndl;

      // [  0     0     0   | 0 |  0  Su^d 0   0   |  0  Vu   0   0  ] [dsdu]   [  rsdu    ]
      Msys->copyDiagMatrixToSubblock_w_pattern(*iter_->sdu, n4st+ndl, n3st+ndl, dest_nnz_st, ndu, nlp_->get_idu());
      dest_nnz_st += ndu;
      Msys->copyDiagMatrixToSubblock_w_pattern(*iter_->vu, n4st+ndl, n4st+ndl, dest_nnz_st, ndu, nlp_->get_idu());
      dest_nnz_st += ndu;

      // [  0     0     0   | 0 |  0   0  Sl^x 0   |  0   0  Zl   0  ] [dsxl]   [  rsxl    ]
      Msys->copyDiagMatrixToSubblock_w_pattern(*iter_->sxl, n4st+ndl+ndu, n3st+ndl+ndu, dest_nnz_st, nxl, nlp_->get_ixl());
      dest_nnz_st += nxl;
      Msys->copyDiagMatrixToSubblock_w_pattern(*iter_->zl, n4st+ndl+ndu, n4st+ndl+ndu, dest_nnz_st, nxl, nlp_->get_ixl());
      dest_nnz_st += nxl;

      // [  0     0     0   | 0 |  0   0   0  Su^x |  0   0   0  Zu  ] [dsxu]   [  rsxu    ]
      Msys->copyDiagMatrixToSubblock_w_pattern(*iter_->sxu,
                                               n4st+ndl+ndu+nxl,
                                               n3st+ndl+ndu+nxl,
                                               dest_nnz_st,
                                               nxu,
                                               nlp_->get_ixu());
      dest_nnz_st += nxu;
      Msys->copyDiagMatrixToSubblock_w_pattern(*iter_->zu,
                                               n4st+ndl+ndu+nxl,
                                               n4st+ndl+ndu+nxl,
                                               dest_nnz_st,
                                               nxu,
                                               nlp_->get_ixu());
      dest_nnz_st += nxu;

      //build the diagonal Hx = delta_wx
      if(nullptr == Hx_) {
        Hx_ = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), nx);
        assert(Hx_);
      }
      Hx_->setToConstant(delta_wx);
      Msys->copySubDiagonalFrom(0, nx, *Hx_, dest_nnz_st); dest_nnz_st += nx;

      //build the diagonal Hd = delta_wd
      if(nullptr == Hd_) {
        Hd_ = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), nd);
        assert(Hd_);
      }

      Hd_->setToConstant(delta_wd);
      Msys->copySubDiagonalFrom(n2st, nd, *Hd_, dest_nnz_st);
      dest_nnz_st += nd;

      //add -delta_cc to diagonal block linSys starting at (nx, nx)
      Msys->setSubDiagonalTo(nx, neq, -delta_cc, dest_nnz_st);
      dest_nnz_st += neq;

      //add -delta_cd to diagonal block linSys starting at (nx+neq, nx+neq)
      Msys->setSubDiagonalTo(nx+neq, nineq, -delta_cd, dest_nnz_st);
      dest_nnz_st += nineq;

      assert(dest_nnz_st==nnz);
      nlp_->log->write("KKT_SPARSE_FULL linsys:", *Msys, hovMatrices);
      nlp_->runStats.kkt.tmUpdateLinsys.stop();
    }

    //write matrix to file if requested
    if(nlp_->options->GetString("write_kkt") == "yes") {
      write_linsys_counter_++;
    }
    if(write_linsys_counter_>=0) {
      csr_writer_.writeMatToFile(*Msys, write_linsys_counter_, nx, neq, nineq);
    }

    return true;
  }


  bool hiopKKTLinSysSparseFull::solve( hiopVector& rx, hiopVector& ryc, hiopVector& ryd, hiopVector& rd,
                                       hiopVector& rvl, hiopVector& rvu, hiopVector& rzl, hiopVector& rzu,
                                       hiopVector& rsdl, hiopVector& rsdu, hiopVector& rsxl, hiopVector& rsxu,
                                       hiopVector& dx, hiopVector& dyc, hiopVector& dyd, hiopVector& dd,
                                       hiopVector& dvl, hiopVector& dvu, hiopVector& dzl, hiopVector& dzu,
                                       hiopVector& dsdl, hiopVector& dsdu, hiopVector& dsxl, hiopVector& dsxu)
  {
    if(!nlpSp_)   { assert(false); return false; }
    if(!HessSp_)  { assert(false); return false; }
    if(!Jac_cSp_) { assert(false); return false; }
    if(!Jac_dSp_) { assert(false); return false; }

    nlp_->runStats.kkt.tmSolveRhsManip.start();

    size_type nx=rx.get_size(), nd=rd.get_size(), neq=ryc.get_size(), nineq=ryd.get_size(),
              ndl = nlp_->m_ineq_low(), ndu = nlp_->m_ineq_upp(), nxl = nlp_->n_low(), nxu = nlp_->n_upp();
    size_type nxsp=Hx_->get_size();
    assert(nxsp==nx);
    int n = nx + neq + nineq + nd + ndl + ndu + nxl + nxu + ndl + ndu + nxl + nxu;

    if(rhs_ == nullptr) {
      rhs_ = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), n);
    }

    {//write to log
      nlp_->log->write("RHS KKT_SPARSE_FULL rx: ", rx,  hovIteration);
      nlp_->log->write("RHS KKT_SPARSE_FULL ryc:", ryc, hovIteration);
      nlp_->log->write("RHS KKT_SPARSE_FULL ryd:", ryd, hovIteration);
      nlp_->log->write("RHS KKT_SPARSE_FULL rd: ", rd,  hovIteration);
      nlp_->log->write("RHS KKT_SPARSE_FULL rvl: ", rvl,  hovIteration);
      nlp_->log->write("RHS KKT_SPARSE_FULL rvu: ", rvu,  hovIteration);
      nlp_->log->write("RHS KKT_SPARSE_FULL rzl: ", rzl,  hovIteration);
      nlp_->log->write("RHS KKT_SPARSE_FULL rzu: ", rzu,  hovIteration);
      nlp_->log->write("RHS KKT_SPARSE_FULL rsdl: ", rsdl,  hovIteration);
      nlp_->log->write("RHS KKT_SPARSE_FULL rsdu: ", rsdu,  hovIteration);
      nlp_->log->write("RHS KKT_SPARSE_FULL rsxl: ", rsxl,  hovIteration);
      nlp_->log->write("RHS KKT_SPARSE_FULL rsxu: ", rsxu,  hovIteration);
    }

    // form the rhs for the sparse linSys
    rx.copyToStarting(*rhs_, 0);
    ryc.copyToStarting(*rhs_, nx);
    ryd.copyToStarting(*rhs_, nx+neq);
    rd.copyToStarting(*rhs_, nx + neq + nineq);
    rvl.copyToStartingAt_w_pattern(*rhs_, nx + neq + nineq + nd, nlp_->get_idl());
    rvu.copyToStartingAt_w_pattern(*rhs_, nx + neq + nineq + nd + ndl, nlp_->get_idu());
    rzl.copyToStartingAt_w_pattern(*rhs_, nx + neq + nineq + nd + ndl + ndu, nlp_->get_ixl());
    rzu.copyToStartingAt_w_pattern(*rhs_, nx + neq + nineq + nd + ndl + ndu + nxl, nlp_->get_ixu());
    rsdl.copyToStartingAt_w_pattern(*rhs_, nx + neq + nineq + nd + ndl + ndu + nxl + nxu, nlp_->get_idl());
    rsdu.copyToStartingAt_w_pattern(*rhs_, nx + neq + nineq + nd + ndl + ndu + nxl + nxu + ndl, nlp_->get_idu());
    rsxl.copyToStartingAt_w_pattern(*rhs_, nx + neq + nineq + nd + ndl + ndu + nxl + nxu + ndl + ndu, nlp_->get_ixl());
    rsxu.copyToStartingAt_w_pattern(*rhs_, nx + neq + nineq + nd + ndl + ndu + nxl + nxu + ndl + ndu + nxl, nlp_->get_ixu());

    if(write_linsys_counter_>=0)
      csr_writer_.writeRhsToFile(*rhs_, write_linsys_counter_);

    nlp_->runStats.kkt.tmSolveRhsManip.stop();

    nlp_->runStats.kkt.tmSolveTriangular.start();

    // solve
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

    // unpack
    rhs_->startingAtCopyToStartingAt(0,          dx,  0);
    rhs_->startingAtCopyToStartingAt(nx,            dyc,  0);
    rhs_->startingAtCopyToStartingAt(nx+neq,           dyd,  0);
    rhs_->startingAtCopyToStartingAt(nx+neq+nineq,         dd,  0);
    rhs_->startingAtCopyToStartingAt_w_pattern(nx+neq+nineq+nd,         dvl,  0, nlp_->get_idl() );
    rhs_->startingAtCopyToStartingAt_w_pattern(nx+neq+nineq+nd+ndl,         dvu,  0, nlp_->get_idu());
    rhs_->startingAtCopyToStartingAt_w_pattern(nx+neq+nineq+nd+ndl+ndu,         dzl,  0, nlp_->get_ixl());
    rhs_->startingAtCopyToStartingAt_w_pattern(nx+neq+nineq+nd+ndl+ndu+nxl,         dzu,  0, nlp_->get_ixu());
    rhs_->startingAtCopyToStartingAt_w_pattern(nx+neq+nineq+nd+ndl+ndu+nxl+nxu,         dsdl,  0, nlp_->get_idl());
    rhs_->startingAtCopyToStartingAt_w_pattern(nx+neq+nineq+nd+ndl+ndu+nxl+nxu+ndl,         dsdu,  0, nlp_->get_idu());
    rhs_->startingAtCopyToStartingAt_w_pattern(nx+neq+nineq+nd+ndl+ndu+nxl+nxu+ndl+ndu,         dsxl,  0, nlp_->get_ixl());
    rhs_->startingAtCopyToStartingAt_w_pattern(nx+neq+nineq+nd+ndl+ndu+nxl+nxu+ndl+ndu+nxl,         dsxu,  0, nlp_->get_ixu());

    {//write to log
      nlp_->log->write("RHS KKT_SPARSE_FULL dx: ", dx,  hovIteration);
      nlp_->log->write("RHS KKT_SPARSE_FULL dyc:", dyc, hovIteration);
      nlp_->log->write("RHS KKT_SPARSE_FULL dyd:", dyd, hovIteration);
      nlp_->log->write("RHS KKT_SPARSE_FULL dd: ", dd,  hovIteration);
      nlp_->log->write("RHS KKT_SPARSE_FULL dvl: ", dvl,  hovIteration);
      nlp_->log->write("RHS KKT_SPARSE_FULL dvu: ", dvu,  hovIteration);
      nlp_->log->write("RHS KKT_SPARSE_FULL dzl: ", dzl,  hovIteration);
      nlp_->log->write("RHS KKT_SPARSE_FULL dzu: ", dzu,  hovIteration);
      nlp_->log->write("RHS KKT_SPARSE_FULL dsdl: ", dsdl,  hovIteration);
      nlp_->log->write("RHS KKT_SPARSE_FULL dsdu: ", dsdu,  hovIteration);
      nlp_->log->write("RHS KKT_SPARSE_FULL dsxl: ", dsxl,  hovIteration);
      nlp_->log->write("RHS KKT_SPARSE_FULL dsxu: ", dsxu,  hovIteration);
    }

    nlp_->runStats.kkt.tmSolveRhsManip.stop();
    return true;
  }



} // end of namespace
