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


// #include "/home/petra1/work/installs/eigen-3.3.9/_install/include/eigen3/Eigen/Core"
// #include "/home/petra1/work/installs/eigen-3.3.9/_install/include/eigen3/Eigen/Sparse"

// // type alias
// using Scalar = double;
// using SparseMatrixCSC = Eigen::SparseMatrix<Scalar, Eigen::StorageOptions::ColMajor>;
// using SparseMatrixCSR = Eigen::SparseMatrix<Scalar, Eigen::StorageOptions::RowMajor>;
// using Triplet = Eigen::Triplet<Scalar>;
// using VectorR = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

namespace hiop
{

hiopKKTLinSysCondensedSparse::hiopKKTLinSysCondensedSparse(hiopNlpFormulation* nlp)
  : hiopKKTLinSysCompressedSparseXDYcYd(nlp)
{
}

hiopKKTLinSysCondensedSparse::~hiopKKTLinSysCondensedSparse()
{
  
}
  
bool hiopKKTLinSysCondensedSparse::build_kkt_matrix(const double& dwx,
                                                    const double& dwd,
                                                    const double& dcc,
                                                    const double& dcd)
{
  assert(0 == Jac_cSp_->m() &&
         "Detected NLP with equality constraints. Please use hiopNlpSparseIneq formulation");
  
  HessSp_ = dynamic_cast<hiopMatrixSymSparseTriplet*>(Hess_);
  Jac_dSp_ = dynamic_cast<const hiopMatrixSparseTriplet*>(Jac_d_);
  Jac_cSp_ = nullptr; //not used by this class
  assert(HessSp_ && Jac_dSp_);
  if(nullptr==Jac_dSp_ || nullptr==HessSp_) {
    return false;
  }
  
  size_type nx = HessSp_->n();
  size_type nineq = Jac_dSp_->m();
  int nnz = HessSp_->numberOfNonzeros() + Jac_cSp_->numberOfNonzeros() + Jac_dSp_->numberOfNonzeros();
  nnz += nx + nineq;
  
//  linSys_ = determineAndCreateLinsys(nx, neq, nineq, nnz);
  return hiopKKTLinSysCompressedSparseXDYcYd::build_kkt_matrix(dwx, dwd, dcc, dcd);
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
  return hiopKKTLinSysCompressedSparseXDYcYd::solveCompressed(rx, rd, ryc, ryd, dx, dd, dyc, dyd);
}


hiopLinSolverIndefSparse*
hiopKKTLinSysCondensedSparse::determine_and_create_linsys(size_type nx, size_type nineq, size_type nnz)
{   
  if(linSys_) {
    return dynamic_cast<hiopLinSolverIndefSparse*> (linSys_);
  }
  
  int n = nx + nineq;
  
  if(nlp_->options->GetString("compute_mode") == "cpu") {
    auto linear_solver = nlp_->options->GetString("linear_solver_sparse");

    //use CHOLDMOD, if not present use ma57
    
    if(linear_solver == "cholmod" || linear_solver == "auto") {
      assert(false && "to be implemented");
    } else {
#ifdef HIOP_USE_COINHSL
      nlp_->log->printf(hovScalars,
                        "KKT_SPARSE_Condensed linsys: alloc MA57 for matrix of size %d (0 cons)\n", n);
      linSys_ = new hiopLinSolverIndefSparseMA57(n, nnz, nlp_);
#else
      assert(false && "HiOp was built without a sparse linear solver needed by the condensed KKT approach");
#endif // HIOP_USE_COINHSL
    }

    //TODO: maybe add pardiso as an option in the future
    
  } else {
    //
    // on device: compute_mode is hybrid, auto, or gpu
    //
    assert(nullptr==linSys_);
    
    //TODO: add cuSparse Cholesky
    
    //Return NULL (and assert) if a GPU sparse linear solver is not present
    assert(linSys_!=nullptr &&
           "HiOp was built without a sparse linear solver for GPU/device and cannot run on the "
           "device as instructed by the 'compute_mode' option. Change the 'compute_mode' to "
           " 'cpu' (from hiopKKTLinSysCompressedSparseXYcYd)"); 
    return nullptr;
  }
  
  assert(linSys_&& "KKT_SPARSE_XYcYd linsys: cannot instantiate backend linear solver");
  return dynamic_cast<hiopLinSolverIndefSparse*> (linSys_);
}
  
}
