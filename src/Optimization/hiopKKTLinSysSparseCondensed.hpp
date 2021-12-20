// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// Written by Nai-Yuan Chiang, chiang7@llnl.gov and Cosmin G. Petra, petra1@llnl.gov.
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

#ifndef HIOP_KKTLINSYSSPARSECONDENSED
#define HIOP_KKTLINSYSSPARSECONDENSED

#include "hiopKKTLinSysSparse.hpp"

/*
 * Solves a sparse KKT linear system by exploiting the sparse structure, namely reduces 
 * the so-called XYcYd KKT system 
 * [  H  +  Dx    0    Jd^T ] [ dx]   [ rx_tilde ]
 * [    0         Dd   -I   ] [ dd] = [ rd_tilde ]
 * [    Jd       -I     0   ] [dyd]   [   ryd    ]
 * into the condensed KKT system
 * (H+Dx+Jd^T*Dd*Jd)dx = rx_tilde + Jd^T*Dd*ryd + Jd^T*rd_tilde
 * dd = Jd*dx - ryd
 * dyd = Dd*dd - rd_tilde = Dd*Jd*dx - Dd*ryd - rd_tilde

 * where Jd is sparse Jacobians for inequalities, H is a sparse Hessian matrix, Dx is 
 * log-barrier diagonal corresponding to x variables, Dd is the log-barrier diagonal 
 * corresponding to the inequality slacks, and I is the identity matrix. 
 *
 * @note: the NLP is assumed to have no equality constraints (or have been relaxed to 
 * two-sided inequality constraints).
 *
 */

namespace hiop
{

class hiopKKTLinSysCondensedSparse : public hiopKKTLinSysCompressedSparseXDYcYd
{
public:
  hiopKKTLinSysCondensedSparse(hiopNlpFormulation* nlp);
  virtual ~hiopKKTLinSysCondensedSparse();

  virtual bool build_kkt_matrix(const double& delta_wx,
                                const double& delta_wd,
                                const double& delta_cc,
                                const double& delta_cd);

  virtual bool solveCompressed(hiopVector& rx, hiopVector& rd, hiopVector& ryc, hiopVector& ryd,
                               hiopVector& dx, hiopVector& dd, hiopVector& dyc, hiopVector& dyd);

protected:
  //
  //from the parent class and its parents we also use
  //

  //right-hand side [rx_tilde, rd_tilde, ((ryc->empty)), ryd]
  //  hiopVector *rhs_; 

  
  //  hiopVectorPar *Dd;
  //  hiopVectorPar *ryd_tilde;

  //from the parent's parent class (hiopKKTLinSysCompressed) we also use
  //  hiopVectorPar *Dx;
  //  hiopVectorPar *rx_tilde;

  //keep Hx = Dx (Dx=log-barrier diagonal for x) + regularization
  //keep Hd = Dd (Dd=log-barrier diagonal for slack variable) + regularization
  //  hiopVector *Hx_, *Hd_;

  //
  //  hiopNlpSparse* nlpSp_;
  //  hiopMatrixSparse* HessSp_;
  //  const hiopMatrixSparse* Jac_cSp_;
  //  const hiopMatrixSparse* Jac_dSp_;

  // int write_linsys_counter_;
  //  hiopCSR_IO csr_writer_;

private:
  //placeholder for the code that decides which linear solver to used based on safe_mode_
  hiopLinSolverIndefSparse* determine_and_create_linsys(size_type nxd,  size_type nineq, size_type nnz);
};

} // end of namespace

#endif
