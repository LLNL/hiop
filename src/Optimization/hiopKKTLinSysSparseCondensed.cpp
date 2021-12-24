// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
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

#include "hiopKKTLinSysSparseCondensed.hpp"
#ifdef HIOP_USE_COINHSL
#include "hiopLinSolverIndefSparseMA57.hpp"
#endif

#ifdef HIOP_USE_CUDA
#include "hiopLinSolverCholCuSparse.hpp"
#endif

//#include "/ccsopen/home/petra1/eigen-3.3.9/_install/include/eigen3/Eigen/Sparse"
//#include "/ccsopen/home/petra1/eigen-3.3.9/_install/include/eigen3/Eigen/Core"
//#include "/g/g15/petra1/eigen-3.3.9/_install/include/eigen3/Eigen/Core"
//#include "/g/g15/petra1/eigen-3.3.9/_install/include/eigen3/Eigen/Sparse"
#include "/home/petra1/work/installs/eigen-3.3.9/_install/include/eigen3/Eigen/Core"
#include "/home/petra1/work/installs/eigen-3.3.9/_install/include/eigen3/Eigen/Sparse"

namespace hiop
{

 
hiopKKTLinSysCondensedSparse::hiopKKTLinSysCondensedSparse(hiopNlpFormulation* nlp)
  : hiopKKTLinSysCompressedSparseXDYcYd(nlp),
    dd_pert_(nullptr)
{
}

hiopKKTLinSysCondensedSparse::~hiopKKTLinSysCondensedSparse()
{
  delete dd_pert_;
}
  
bool hiopKKTLinSysCondensedSparse::build_kkt_matrix(const double& delta_wx_in,
                                                    const double& delta_wd_in,
                                                    const double& dcc,
                                                    const double& dcd)
{
  auto delta_wx = delta_wx_in;
  auto delta_wd = delta_wd_in;
  if(dcc>0) {
    delta_wx += dcc;
    delta_wd += dcc;
  }
  
  HessSp_ = dynamic_cast<hiopMatrixSymSparseTriplet*>(Hess_);
  Jac_dSp_ = dynamic_cast<const hiopMatrixSparseTriplet*>(Jac_d_);
  Jac_cSp_ = nullptr; //not used by this class
  assert(HessSp_ && Jac_dSp_);
  if(nullptr==Jac_dSp_ || nullptr==HessSp_) {
    return false;
  }

  assert(0 == Jac_c_->m() &&
         "Detected NLP with equality constraints. Please use hiopNlpSparseIneq formulation");
  
  size_type nx = HessSp_->n();
  size_type nineq = Jac_dSp_->m();
  assert(nineq == Dd_->get_size());
  assert(nx = Dx_->get_size());
  
  nlp_->runStats.kkt.tmUpdateLinsys.start();

  hiopTimer t;
  
  t.start();
  SparseMatrixCSC JacD(nineq, nx);
  {
    std::vector<Triplet> tripletList;
    tripletList.reserve(Jac_dSp_->numberOfNonzeros());
    for(int i = 0; i < Jac_dSp_->numberOfNonzeros(); i++) {
      tripletList.push_back(Triplet(Jac_dSp_->i_row()[i],
                                    Jac_dSp_->j_col()[i],
                                    Jac_dSp_->M()[i]));
    }
    
    JacD.setFromTriplets(tripletList.begin(), tripletList.end());
  }
  t.stop();
  if(perf_report_) nlp_->log->printf(hovSummary, "JacD took        %.3f sec\n", t.getElapsedTime());

  t.reset(); t.start();
  SparseMatrixCSC JacD_trans =  SparseMatrixCSC(JacD.transpose());
  t.stop();
  if(perf_report_) nlp_->log->printf(hovSummary, "JacD trans took       %.3f sec\n", t.getElapsedTime());

  t.reset(); t.start();
  if(nullptr == dd_pert_) {
    dd_pert_ = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), nineq);
  }
  dd_pert_->copyFrom(*Dd_);
  dd_pert_->addConstant(delta_wd);
  t.stop(); 
  if(perf_report_) nlp_->log->printf(hovSummary, "DD vec ops        %.3f sec\n", t.getElapsedTime());

  
  t.reset(); t.start();
  SparseMatrixCSC Dd_mat(nineq, nineq);
  {
    std::vector<Triplet> tripletList;
    tripletList.reserve(Dd_->get_size());
    for(int i=0; i<Dd_->get_size(); i++) {
      tripletList.push_back(Triplet(i, i, dd_pert_->local_data()[i]));
    }
    Dd_mat.setFromTriplets(tripletList.begin(), tripletList.end());
  }
  t.stop(); 
  if(perf_report_) nlp_->log->printf(hovSummary, "Ddmat took       %.3f sec\n", t.getElapsedTime());

  
  t.reset(); t.start();
  SparseMatrixCSC DdxJ = Dd_mat * JacD;
  t.stop();
  if(perf_report_) nlp_->log->printf(hovSummary, "DdxJ took         %.3f sec\n", t.getElapsedTime());

  t.reset(); t.start();  
  SparseMatrixCSC JtxDdxJ = JacD_trans * DdxJ;
  t.stop();
  if(perf_report_) nlp_->log->printf(hovSummary, "JtxDdxJ took       %.3f sec\n", t.getElapsedTime());


  t.reset(); t.start();
  SparseMatrixCSC H(nx, nx);
  {
    std::vector<Triplet> tripletList;
    tripletList.reserve(HessSp_->numberOfNonzeros());
    for(int i = 0; i < HessSp_->numberOfNonzeros(); i++) {
      tripletList.push_back(Triplet(HessSp_->i_row()[i],
                                    HessSp_->j_col()[i],
                                    HessSp_->M()[i]));
    }
    H.setFromTriplets(tripletList.begin(), tripletList.end());
  }
  t.stop();
  if(perf_report_) nlp_->log->printf(hovSummary, "H took        %.3f sec\n", t.getElapsedTime());

  t.reset(); t.start(); 
  SparseMatrixCSC KKTmat = H + JtxDdxJ;
  t.stop();
  if(perf_report_) nlp_->log->printf(hovSummary, "H  + JtxDdxJ took         %.3f sec\n", t.getElapsedTime());
 
  t.reset(); t.start(); 
  SparseMatrixCSC Dx_mat(nx, nx);
  {
    assert(Dx_->get_size() == nx);
    std::vector<Triplet> tripletList;
    tripletList.reserve(nx);
    for(int i=0; i<nx; i++) {
      tripletList.push_back(Triplet(i, i, Dx_->local_data()[i]+delta_wx));
    }
    Dx_mat.setFromTriplets(tripletList.begin(), tripletList.end());
  }
  t.stop();
  if(perf_report_) nlp_->log->printf(hovSummary, "Dx_mat took         %.3f sec\n", t.getElapsedTime());

  t.reset(); t.start(); 
  KKTmat += Dx_mat;
  t.stop();
  if(perf_report_) nlp_->log->printf(hovSummary, "KKTmat += Dx_mat took         %.3f sec\n", t.getElapsedTime());

  t.reset(); t.start(); 
  KKTmat.makeCompressed();
  t.stop();
  if(perf_report_) nlp_->log->printf(hovSummary, "makeCompressed         %.3f sec\n", t.getElapsedTime());

  t.reset(); t.start(); 

  
  bool is_cusolver_on = nlp_->options->GetString("compute_mode") == "cpu" ? false : true;
  
  //count nnz in the lower triangle
  size_type nnz_KKT_lowertri = -1;

  if(!is_cusolver_on) {
    nnz_KKT_lowertri = 0;
    for(int k=0; k<KKTmat.outerSize(); ++k) {
      for(Eigen::SparseMatrix<double>::InnerIterator it(KKTmat, k); it; ++it) {
        //it.value();
        //it.row();   // row index
        //it.col();   // col index (here it is equal to k)
        //it.index(); // inner index, here it is equal to it.row()
        if(it.row() >= it.col()) {
          nnz_KKT_lowertri++;
        }
      }
    }
    t.stop();
    if(perf_report_) nlp_->log->printf(hovSummary, "nnz_KKT_lowertri         %.3f sec\n", t.getElapsedTime());

    linSys_ = determine_and_create_linsys(nx, nineq, nnz_KKT_lowertri);
  } else {

    linSys_ = determine_and_create_linsys(nx, nineq, KKTmat.nonZeros());
  }

  hiopLinSolverIndefSparse* linSys = dynamic_cast<hiopLinSolverIndefSparse*> (linSys_);
  assert(linSys);

  hiopMatrixSparseTriplet& Msys = linSys->sysMatrix();

  //populate Msys
  if(!is_cusolver_on) {
    assert(Msys.numberOfNonzeros() == nnz_KKT_lowertri);
    
    t.reset(); t.start(); 
    
    size_type it_nnz = 0;
    for(int k=0; k<KKTmat.outerSize(); ++k) {
      for(Eigen::SparseMatrix<double>::InnerIterator it(KKTmat, k); it; ++it) {
        if(it.row() >= it.col()) {
          Msys.i_row()[it_nnz] = it.row();
          Msys.j_col()[it_nnz] = it.col();
          Msys.M()[it_nnz++] = it.value();
        }
      }
    }
    t.stop();
    if(perf_report_) nlp_->log->printf(hovSummary, "copy lowertri         %.3f sec\n", t.getElapsedTime());
  } else {
    hiopLinSolverCholCuSparse* linSys_cusolver = dynamic_cast<hiopLinSolverCholCuSparse*>(linSys);
    assert(linSys_cusolver);
    linSys_cusolver->set_sys_mat(KKTmat);
    
  }


  nlp_->log->write("KKT_SPARSE_Condensed linsys:", Msys, hovMatrices);
  nlp_->runStats.kkt.tmUpdateLinsys.stop();
  
  if(perf_report_) {
    nlp_->log->printf(hovSummary,
                      "KKT_SPARSE_Condensed linsys: Low-level linear system size %d nnz %d\n",
                      Msys.n(),
                      Msys.numberOfNonzeros());
  }

  //write matrix to file if requested
  if(nlp_->options->GetString("write_kkt") == "yes") {
    write_linsys_counter_++;
  }
  if(write_linsys_counter_>=0) {
    csr_writer_.writeMatToFile(Msys, write_linsys_counter_, nx, 0, nineq);
  }
  return true; //xxx hiopKKTLinSysCompressedSparseXDYcYd::build_kkt_matrix(dwx, dwd, dcc, dcd);
}
  
bool hiopKKTLinSysCondensedSparse::solveCompressed(hiopVector& rx,
                                                   hiopVector& rd,
                                                   hiopVector& ryc,
                                                   hiopVector& ryd,
                                                   hiopVector& dx,
                                                   hiopVector& dd,
                                                   hiopVector& dyc,
                                                   hiopVector& dyd)
{
  assert(nlpSp_);
  assert(HessSp_);
  assert(Jac_dSp_);
  assert(0 == ryc.get_size() && "this KKT does not support equality constraints");

  size_type nx = rx.get_size();
  size_type nd = rd.get_size();
  size_type nyd = ryd.get_size();

  nlp_->runStats.kkt.tmSolveRhsManip.start();
  
  if(rhs_ == NULL) {
    rhs_ = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), nx);
  }
  assert(rhs_->get_size() == nx);

  nlp_->log->write("RHS KKT_SPARSE_Condensed rx: ", rx,  hovIteration);
  nlp_->log->write("RHS KKT_SPARSE_Condensed rx: ", rd,  hovIteration);
  nlp_->log->write("RHS KKT_SPARSE_Condensed ryc:", ryc, hovIteration);
  nlp_->log->write("RHS KKT_SPARSE_Condensed ryd:", ryd, hovIteration);

  /* (H+Dx+Jd^T*(Dd+delta_wd*I)*Jd)dx = rx + Jd^T*Dd*ryd + Jd^T*rd
   * dd = Jd*dx - ryd
   * dyd = (Dd+delta_wd*I)*dd - rd = (Dd+delta_wd*I)*Jd*dx - (Dd+delta_wd*I)*ryd - rd
   */
  rhs_->copyFrom(rx);

  //working buffers in the size of nineq/nd using output as storage
  hiopVector& Dd_x_ryd = dyd;
  Dd_x_ryd.copyFrom(ryd);
  Dd_x_ryd.componentMult(*dd_pert_);

  hiopVector& DD_x_ryd_plus_rd = Dd_x_ryd;
  DD_x_ryd_plus_rd.axpy(1.0, rd);

  Jac_dSp_->transTimesVec(1.0, *rhs_, 1.0, DD_x_ryd_plus_rd);

  nlp_->runStats.kkt.tmSolveRhsManip.stop();
  nlp_->runStats.kkt.tmSolveTriangular.start();
  
  //
  // solve
  //
  bool linsol_ok = linSys_->solve(*rhs_);
  nlp_->runStats.kkt.tmSolveTriangular.stop();
  nlp_->runStats.linsolv.end_linsolve();
  
  if(perf_report_) {
    nlp_->log->printf(hovSummary, "(summary for linear solver from KKT_SPARSE_Condensed)\n%s",
                      nlp_->runStats.linsolv.get_summary_last_solve().c_str());
  }
  
  if(write_linsys_counter_>=0)
    csr_writer_.writeSolToFile(*rhs_, write_linsys_counter_);
  
  if(false==linsol_ok) return false;

  nlp_->runStats.kkt.tmSolveRhsManip.start();
  
  dx.copyFrom(*rhs_);

  dd.copyFrom(ryd);
  Jac_dSp_->timesVec(-1.0, dd, 1.0, dx);

  dyd.copyFrom(dd);
  dyd.componentMult(*dd_pert_);
  dyd.axpy(-1.0, rd);

  nlp_->runStats.kkt.tmSolveRhsManip.stop();

  nlp_->log->write("SOL KKT_SPARSE_Condensed dx: ", dx,  hovMatrices);
  nlp_->log->write("SOL KKT_SPARSE_Condensed dd: ", dd,  hovMatrices);
  nlp_->log->write("SOL KKT_SPARSE_Condensed dyc:", dyc, hovMatrices);
  nlp_->log->write("SOL KKT_SPARSE_Condensed dyd:", dyd, hovMatrices);
  return true;
}


hiopLinSolverIndefSparse*
hiopKKTLinSysCondensedSparse::determine_and_create_linsys(size_type nx, size_type nineq, size_type nnz)
{   
  if(linSys_) {
    return dynamic_cast<hiopLinSolverIndefSparse*> (linSys_);
  }
  
  int n = nx;

  if(nlp_->options->GetString("compute_mode") == "cpu") {
    //auto linear_solver = nlp_->options->GetString("linear_solver_sparse");

    //TODO:
    //add support for linear_solver == "cholmod"
    // maybe add pardiso as an option in the future
    //
    
#ifdef HIOP_USE_COINHSL
    nlp_->log->printf(hovWarning,
                      "KKT_SPARSE_Condensed linsys: alloc MA57 for matrix of size %d (0 cons)\n", n);
    linSys_ = new hiopLinSolverIndefSparseMA57(n, nnz, nlp_);
#else
    assert(false && "HiOp was built without a sparse linear solver needed by the condensed KKT approach");
#endif // HIOP_USE_COINHSL
    
  } else {
    //
    // on device: compute_mode is hybrid, auto, or gpu
    //
    assert(nullptr==linSys_);
    
#ifdef HIOP_USE_CUDA
    nlp_->log->printf(hovWarning,
                      "KKT_SPARSE_Condensed linsys: alloc cuSOLVER-chol matrix size %d\n", n);
    linSys_ = new hiopLinSolverCholCuSparse(n, nnz, nlp_);
#endif    
    
    //Return NULL (and assert) if a GPU sparse linear solver is not present
    assert(linSys_!=nullptr &&
           "HiOp was built without a sparse linear solver for GPU/device and cannot run on the "
           "device as instructed by the 'compute_mode' option. Change the 'compute_mode' to 'cpu'");
  }
  
  assert(linSys_&& "KKT_SPARSE_XYcYd linsys: cannot instantiate backend linear solver");
  return dynamic_cast<hiopLinSolverIndefSparse*> (linSys_);
}
  
}
