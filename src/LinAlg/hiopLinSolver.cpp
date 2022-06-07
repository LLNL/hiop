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

/**
 * @file hiopLinSolver.hpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 * @author Nai-Yuan Chiang <chiang7@llnl.gov>, LLNL
 *
 */

#include "hiopLinSolver.hpp"

#include "hiopOptions.hpp"
#include "hiopLinAlgFactory.hpp"

namespace hiop {
  hiopLinSolver::hiopLinSolver()
    : nlp_(NULL), perf_report_(false)
  {
  }
  hiopLinSolver::~hiopLinSolver()
  {
  }

  /// Constructor allocates dense system matrix
  hiopLinSolverSymDense::hiopLinSolverSymDense(int n, hiopNlpFormulation* nlp)
  {
    nlp_ = nlp;
    perf_report_ = "on"==hiop::tolower(nlp_->options->GetString("time_kkt"));
    M_ = LinearAlgebraFactory::create_matrix_dense(nlp_->options->GetString("mem_space"), n, n);
  }

  /// Default constructor is protected and should fail when called
  hiopLinSolverSymDense::hiopLinSolverSymDense()
    : M_(nullptr)
  {
    assert(false);
  }

  /// Destructor deletes the system matrix
  hiopLinSolverSymDense::~hiopLinSolverSymDense()
  {
    delete M_;
  }

  /// Method to return reference to the system matrix
  hiopMatrixDense& hiopLinSolverSymDense::sysMatrix()
  {
    return *M_;
  }

  hiopLinSolverSymSparse::hiopLinSolverSymSparse(int n, int nnz, hiopNlpFormulation* nlp)
  {
    //we default to triplet matrix for now; derived classes using CSR matrices will not call
    //this constructor (will call the 1-parameter constructor below) so they avoid creating
    //the triplet matrix
    M_ = new hiopMatrixSparseTriplet(n, n, nnz);
    //this class will own `M_`
    sys_mat_owned_ = true;
    nlp_ = nlp;
    perf_report_ = "on"==hiop::tolower(nlp->options->GetString("time_kkt"));
  }

  hiopLinSolverSymSparse::hiopLinSolverSymSparse(hiopNlpFormulation* nlp)
  {
    M_ = nullptr;
    sys_mat_owned_ = false;
    nlp_ = nlp;
    perf_report_ = "on"==hiop::tolower(nlp->options->GetString("time_kkt"));
  }

  hiopLinSolverSymSparse::hiopLinSolverSymSparse(hiopMatrixSparse* M, hiopNlpFormulation* nlp)
  {
    M_ = M;
    sys_mat_owned_ = false;
    nlp_ = nlp;
    perf_report_ = "on"==hiop::tolower(nlp->options->GetString("time_kkt"));
  }
  
  hiopLinSolverNonSymSparse::hiopLinSolverNonSymSparse(int n, int nnz, hiopNlpFormulation* nlp)
  {
    M_ = new hiopMatrixSparseTriplet(n, n, nnz);
    sys_mat_owned_ = false;
    nlp_ = nlp;
    perf_report_ = "on"==hiop::tolower(nlp->options->GetString("time_kkt"));
  }

} // namespace hiop
