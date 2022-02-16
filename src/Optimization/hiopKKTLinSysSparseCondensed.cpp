// Copyrightf (c) 2017, Lawrence Livermore National Security, LLC.
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

/**
 * @file hiopKKTLinSysSparseCondensed.cpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 */

#include "hiopKKTLinSysSparseCondensed.hpp"
#ifdef HIOP_USE_COINHSL
#include "hiopLinSolverIndefSparseMA57.hpp"
#endif

#ifdef HIOP_USE_CUDA
#include "hiopLinSolverCholCuSparse.hpp"
#endif

#include "hiopMatrixSparseTripletStorage.hpp"
namespace hiop
{

 
hiopKKTLinSysCondensedSparse::hiopKKTLinSysCondensedSparse(hiopNlpFormulation* nlp)
  : hiopKKTLinSysCompressedSparseXDYcYd(nlp),
    JacD_(nullptr),
    JacDt_(nullptr),
    JtDiagJsto_(nullptr),
    JtDiagJ_(nullptr),
    Hess_lower_csr_(nullptr),
    Hess_upper_csr_(nullptr),
    Hess_csr_(nullptr),
    M_condensedsto_(nullptr),
    M_condensed_(nullptr),
    delta_wx_(0.),
    krylov_mat_opr_(nullptr),
    krylov_prec_opr_(nullptr),
    krylov_rhs_xdycyd_(nullptr),
    bicgstab_(nullptr)
{
}

hiopKKTLinSysCondensedSparse::~hiopKKTLinSysCondensedSparse()
{
  delete bicgstab_;
  delete krylov_rhs_xdycyd_;
  delete krylov_prec_opr_;
  delete krylov_mat_opr_;
  delete M_condensedsto_;
  delete M_condensed_;
  delete JtDiagJsto_;
  delete JtDiagJ_;
  delete JacDt_;
  delete JacD_;
  delete Hess_csr_;
  delete Hess_upper_csr_;
  delete Hess_lower_csr_;
}

bool hiopKKTLinSysCondensedSparse::build_kkt_matrix(const double& delta_wx_in,
                                                    const double& delta_wd_in,
                                                    const double& dcc,
                                                    const double& dcd)
{
  nlp_->runStats.kkt.tmUpdateInit.start();
  
  auto delta_wx = delta_wx_in;
  auto delta_wd = delta_wd_in;
  if(dcc!=0) {
    //nlp_->log->printf(hovWarning, "LinSysCondensed: dual reg. %.6e primal %.6e %.6e\n", dcc, delta_wx, delta_wd);
    assert(dcc == dcd);
    delta_wx += fabs(dcc);
    delta_wd += fabs(dcc);
  }
  delta_wx_ = delta_wx;

  hiopMatrixSymSparseTriplet* Hess_triplet = dynamic_cast<hiopMatrixSymSparseTriplet*>(Hess_);
  HessSp_ = Hess_triplet; //dynamic_cast<hiopMatrixSymSparseTriplet*>(Hess_);
  
  Jac_cSp_ = nullptr; //not used by this class

  const hiopMatrixSparseTriplet* Jac_triplet = dynamic_cast<const hiopMatrixSparseTriplet*>(Jac_d_);
  Jac_dSp_ = Jac_triplet;
  
  assert(HessSp_ && Jac_dSp_);
  if(nullptr==Jac_dSp_ || nullptr==HessSp_) {
    //incorrect linear algebra objects were provided to this class
    return false;
  }

  assert(0 == Jac_c_->m() &&
         "Detected NLP with equality constraints. Please use hiopNlpSparseIneq formulation");
  
  size_type nx = HessSp_->n();
  size_type nineq = Jac_dSp_->m();
  assert(nineq == Dd_->get_size());
  assert(nx == Dx_->get_size());
  
  

  if(nullptr == Hd_) {
    Hd_ = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), nineq);
  }
  Hd_->copyFrom(*Dd_);  
  Hd_->addConstant(delta_wd);

  nlp_->runStats.kkt.tmUpdateInit.stop();
  nlp_->runStats.kkt.tmUpdateLinsys.start();
  
#define USE_OLD_CODE 0
#define USE_NEW_CODE 1  
  
  //
  // compute condensed linear system J'*D*J + H + Dx + delta_wx*I
  //

  hiopTimer t;

#if USE_OLD_CODE  
  t.reset(); t.start();
  hiopMatrixSparseCSRStorage JacD;
  JacD.form_from(*Jac_triplet);
  t.stop(); printf("JacD-stor    took %.5f\n", t.getElapsedTime());
  //Jac_triplet->print();

  t.reset(); t.start();
  hiopMatrixSparseCSRStorage JacDt;
  JacDt.form_transpose_from(*Jac_triplet);
  t.stop(); printf("JacDt-stor   took %.5f\n", t.getElapsedTime());

  // compute J'*D*J
  t.reset(); t.start();
  if(nullptr == JtDiagJsto_) {
    //perform the initial allocation 
    JtDiagJsto_ = JacDt.times_diag_times_mat_init(JacD);
    //perform symbolic (determine sparsity pattern) and numerical multiplication
    JacDt.times_diag_times_mat(*Hd_, JacD, *JtDiagJsto_);
    
  } else {
    //perform numerical multiplication
    JacDt.times_diag_times_mat_numeric(*Hd_, JacD, *JtDiagJsto_);
  }
  t.stop(); printf("J*D*J'-stor  took %.5f\n", t.getElapsedTime());

  t.reset(); t.start();
  // compute J'*D*J + H + Dx + delta_wx*I
  if(nullptr == M_condensedsto_) {
    M_condensedsto_ = add_matrices_init(*JtDiagJsto_, *Hess_triplet, *Dx_, delta_wx_);
  }
  add_matrices(*JtDiagJsto_, *Hess_triplet, *Dx_, delta_wx_, *M_condensedsto_);
  t.stop(); printf("add     took %.5f\n", t.getElapsedTime());
#endif //OLD_CODE

#if USE_NEW_CODE  
  // symbolic conversion from triplet to CSR
  if(nullptr == JacD_) {
    t.reset(); t.start();
    JacD_ = new hiopMatrixSparseCSR();
    JacD_->form_from_symbolic(*Jac_triplet);
    //JacD_.print();

    assert(nullptr == JacDt_);
    JacDt_ = new hiopMatrixSparseCSR();
    JacDt_->form_transpose_from_symbolic(*Jac_triplet);
    //t.stop(); printf("JacD JacDt-symb    took %.5f\n", t.getElapsedTime());
  }

  // numeric conversion from triplet to CSR
  t.reset(); t.start();
  JacD_->form_from_numeric(*Jac_triplet);
  JacDt_->form_transpose_from_numeric(*Jac_triplet);  
  //t.stop(); printf("JacD JacDt-nume    took %.5f\n", t.getElapsedTime());
  
  //symbolic multiplication for JacD'*D*J
  if(nullptr == JtDiagJ_) {
    t.reset(); t.start();
    
    // D * J
    //nothing to do symbolically since we just numerically scale columns of Jt by D 
  
    // Jt* (D*J)  (D is not used since it does not change the sparsity pattern)
    JtDiagJ_ = JacDt_->times_mat_alloc(*JacD_);
    JacDt_->times_mat_symbolic(*JtDiagJ_, *JacD_);
    //t.stop(); printf("J*D*J'-symb  took %.5f\n", t.getElapsedTime());
#if USE_OLD_CODE
    assert(JtDiagJ_->numberOfNonzeros() == JtDiagJsto_->nnz());
#endif    
  }
  
  //numeric multiplication for JacD'*D*J
  t.reset(); t.start();
  // Jt * D
  JacD_->scale_rows(*Hd_);
  // (Jt*D) * J
  JacDt_->times_mat_numeric(0.0, *JtDiagJ_, 1.0, *JacD_);
  //t.stop(); printf("J*D*J'-nume  took %.5f\n", t.getElapsedTime());
  
#ifdef HIOP_DEEPCHECKS
  JtDiagJ_->check_csr_is_ordered();
#endif
  
  if(nullptr == M_condensed_) {
    assert(nullptr == Hess_upper_csr_);
    Hess_upper_csr_ = new hiopMatrixSparseCSR();
    Hess_upper_csr_->form_from_symbolic(*Hess_triplet);

    assert(nullptr == Hess_lower_csr_);
    Hess_lower_csr_ = new hiopMatrixSparseCSR();;
    Hess_lower_csr_->form_transpose_from_symbolic(*Hess_triplet);

    assert(Hess_lower_csr_->numberOfNonzeros() == Hess_upper_csr_->numberOfNonzeros());

    assert(nullptr == Hess_csr_);
    Hess_csr_ = Hess_lower_csr_->add_matrix_alloc(*Hess_upper_csr_);
    Hess_lower_csr_->add_matrix_symbolic(*Hess_csr_, *Hess_upper_csr_);

    
    //a temporary matrix needed to form sparsity pattern of M_condensed_
    auto* M_condensed_tmp = Hess_csr_->add_matrix_alloc(*JtDiagJ_);
    Hess_csr_->add_matrix_symbolic(*M_condensed_tmp, *JtDiagJ_);
    
    //ensure storage for nonzeros diagonal is allocated by adding (symbolically)
    //a diagonal matrix
    hiopMatrixSparseCSR Diag;
    Diag.form_diag_from_symbolic(*Dx_);

    M_condensed_ = M_condensed_tmp->add_matrix_alloc(Diag);
    M_condensed_tmp->add_matrix_symbolic(*M_condensed_, Diag);
    delete M_condensed_tmp;

    //t.stop(); printf("ADD-symb  took %.5f\n", t.getElapsedTime());
  }


  t.reset(); t.start();
  Hess_upper_csr_->form_from_numeric(*Hess_triplet);
  Hess_upper_csr_->set_diagonal(0.0);
  Hess_lower_csr_->form_transpose_from_numeric(*Hess_triplet);
  Hess_lower_csr_->add_matrix_numeric(0.0, *Hess_csr_, 1.0, *Hess_upper_csr_, 1.0);

  M_condensed_->setToZero();
  if(delta_wx>0) {
    M_condensed_->set_diagonal(delta_wx);
  }
  M_condensed_->addDiagonal(1.0, *Dx_);

  Hess_csr_->add_matrix_numeric(1.0, *M_condensed_, 1.0, *JtDiagJ_, 1.0);  
  
  //t.stop(); printf("ADD-nume  took %.5f\n", t.getElapsedTime());
#endif  //NEW_CODE


#if USE_OLD_CODE  
  int nnz_condensed = M_condensedsto_->nnz();
#else
  int nnz_condensed = M_condensed_->numberOfNonzeros();
#endif

  //toggle on (use values from new code) or off (use values from old)  
#if 1
  nnz_condensed = M_condensed_->numberOfNonzeros();
  delete M_condensedsto_;
  M_condensedsto_ = new hiopMatrixSparseCSRStorage(M_condensed_->m(), M_condensed_->m(), nnz_condensed);

  memcpy(M_condensedsto_->irowptr(), M_condensed_->i_row(), (1+M_condensed_->m())*sizeof(index_type));
  memcpy(M_condensedsto_->jcolind(), M_condensed_->j_col(), M_condensed_->numberOfNonzeros()*sizeof(index_type));
  memcpy(M_condensedsto_->values(), M_condensed_->M(), M_condensed_->numberOfNonzeros()*sizeof(double));
#endif
  //
  // linear system matrix update
  //

  // TODO work directly with
  // hiopMatrixSparseTriplet& Msys = linSys->sysMatrix();
  // TODO should have same code for different compute modes (remove is_cusolver_on)
  
  bool is_cusolver_on = nlp_->options->GetString("compute_mode") == "cpu" ? false : true;
#ifndef HIOP_USE_CUDA
  if(is_cusolver_on) {
    nlp_->log->printf(hovWarning,
                      "hiopKKTLinSysCondensedSparse: HiOp was built without CUDA and will use a CPU "
                      "linear solver (MA57)\n");
    is_cusolver_on = false;
  }
#endif  
  if(is_cusolver_on) {
    linSys_ = determine_and_create_linsys(nx, nineq, M_condensedsto_->nnz());

#ifdef HIOP_USE_CUDA    
    hiopLinSolverCholCuSparse* linSys_cusolver = dynamic_cast<hiopLinSolverCholCuSparse*>(linSys_);
    assert(linSys_cusolver);
    linSys_cusolver->set_linsys_mat(M_condensedsto_);
#endif    
    
  } else {
    //compute mode cpu -> use update MA57 linear solver's matrix
    
    if(nullptr == linSys_) {
      
      index_type itnz = 0;
      for(index_type i=0; i<M_condensedsto_->m(); ++i) {

        for(index_type p=M_condensedsto_->irowptr()[i]; p<M_condensedsto_->irowptr()[i+1]; ++p) {
          const index_type j = M_condensedsto_->jcolind()[p];
          if(i<=j) {
            itnz++; 
          }
        }
      }
      linSys_ = determine_and_create_linsys(nx, nineq, itnz);
    }

    assert(linSys_);
    auto* linSys = dynamic_cast<hiopLinSolverSymSparse*> (linSys_);
    auto* Msys = dynamic_cast<hiopMatrixSparseTriplet*>(linSys->sysMatrix());
    assert(Msys);
    assert(Msys->m() == M_condensedsto_->m());

    index_type itnz = 0;
    for(index_type i=0; i<Msys->m(); ++i) {
      for(index_type p=M_condensedsto_->irowptr()[i]; p<M_condensedsto_->irowptr()[i+1]; ++p) {
        const index_type j = M_condensedsto_->jcolind()[p];
        if(i<=j) {
          Msys->i_row()[itnz] = i;
          Msys->j_col()[itnz] = j;
          Msys->M()[itnz] = M_condensedsto_->values()[p];
          itnz++; 
        }
      }
    }
  }
  nlp_->runStats.kkt.tmUpdateLinsys.stop();
  
  if(perf_report_) {
    nlp_->log->printf(hovSummary,
                      "KKT_SPARSE_Condensed linsys: Low-level linear system size %d nnz %d\n",
                      nx, 
                      M_condensedsto_->nnz());
  }

  //write matrix to file if requested
  if(nlp_->options->GetString("write_kkt") == "yes") {
    write_linsys_counter_++;
  }
  if(write_linsys_counter_>=0) {
    // TODO csr_writer_.writeMatToFile(Msys, write_linsys_counter_, nx, 0, nineq);
  }
  return true; 
}



hiopMatrixSparseCSRStorage* hiopKKTLinSysCondensedSparse::
add_matrices_init(hiopMatrixSparseCSRStorage& JtDiagJ,
                  hiopMatrixSymSparseTriplet& Hess,
                  hiopVector& Dx,
                  double delta_wx)
{
  size_type m = JtDiagJ.m();
  assert(Hess.m() == m);
  assert(Dx.get_size() == m);
  size_type M_nnz = 0;
  
  size_type JtDJ_nnz = JtDiagJ.nnz();
  M_nnz += JtDJ_nnz;
  M_nnz += m;
  size_type H_nnz = Hess.numberOfNonzeros();
  M_nnz += H_nnz;
  // since Hess contains only the upper triangle, add the elements that would correspond to
  // the strict lower triangle
  index_type* H_irow = Hess.i_row();
  index_type* H_jcol = Hess.j_col();
  size_type H_nnz_lowtr = 0;
  for(index_type it=0; it<H_nnz; ++it) {
    assert(H_irow[it] <= H_jcol[it]);
    if(H_irow[it]<H_jcol[it]) {
      H_nnz_lowtr++;
    } 
  }
  M_nnz += H_nnz_lowtr;

  //build an array with indexes in H sparse arrays of the elements in the strictly lower triangle
  map_H_lowtr_idxs_.resize(H_nnz_lowtr);
  H_nnz_lowtr = 0;
  for(index_type it=0; it<H_nnz; ++it) {
    if(H_irow[it]<H_jcol[it]) {
      map_H_lowtr_idxs_[H_nnz_lowtr++] = it;
    } 
  }
  
  //allocate space for the result M = JtDiagJ+Hess+Dx+delta_wx*I
  index_type* M_irow = new index_type[M_nnz];
  index_type* M_jcol = new index_type[M_nnz];

  //
  //populate (i,j) for M
  //
  int itnnz=0;

  //populate with JtDiagJ
  index_type* JtDJ_irowptr = JtDiagJ.irowptr();
  index_type* JtDJ_jcolind = JtDiagJ.jcolind();
  for(index_type i=0; i<m; i++) {
    for(index_type p=JtDJ_irowptr[i]; p<JtDJ_irowptr[i+1]; ++p) {
      assert(itnnz<JtDJ_nnz);
      const index_type j = JtDJ_jcolind[p];
      M_irow[itnnz] = i;
      M_jcol[itnnz] = j;
      //M_values[itnnz] = JtDJ_values[p];
      itnnz++;
    }
  }
  assert(itnnz == JtDiagJ.nnz());

  //populate with H (upper triangle)
  memcpy(M_irow+itnnz, H_irow, H_nnz*sizeof(index_type));
  memcpy(M_jcol+itnnz, H_jcol, H_nnz*sizeof(index_type));
  //memcpy(M_values+itnnz, H_values, H_nnz*sizeof(double));
  itnnz += H_nnz;

  //populate with H (strictly lower triangle)
  for(auto idx : map_H_lowtr_idxs_) {
    //flip row with col to put in the strictly lower triangle
    M_irow[itnnz] = H_jcol[idx];
    M_jcol[itnnz] = H_irow[idx];
    assert(itnnz <JtDJ_nnz+H_nnz+H_nnz_lowtr);
    itnnz++; 
  }
  assert(itnnz == JtDJ_nnz+H_nnz+H_nnz_lowtr);

  //populate with D and delta_wx*I
  for(int i=0; i<m; i++) {
    M_irow[itnnz] = i;
    M_jcol[itnnz] = i;
    //M_values[itnnz] = Dx[i]+delta_wx;;
    itnnz++; 
  }
  assert(itnnz == M_nnz);
  assert(itnnz == JtDJ_nnz+H_nnz+H_nnz_lowtr+m);

  //
  // sort
  //
  std::vector<index_type> idxs_sorted(M_nnz);
  std::iota(idxs_sorted.begin(), idxs_sorted.end(), 0);
  sort(idxs_sorted.begin(), idxs_sorted.end(), 
       [&](const int& i1, const int& i2) { 
         if(M_irow[i1]<M_irow[i2]) return true;
         if(M_irow[i1]>M_irow[i2]) return false;
         return M_jcol[i1]<M_jcol[i2];
       });
  
  //shuffle elements
  index_type* buff_i = new index_type[M_nnz];

  //shuffle M_irow
  for(int i=0; i<M_nnz; ++i) {
    buff_i[i] = M_irow[idxs_sorted[i]];
  }
  //use original M_irow as buffer for M_jcol
  index_type* buff_j = M_irow;
  //exchange pointers to avoid copying
  M_irow = buff_i;
  
  //shuffle M_jcol
  for(int i=0; i<M_nnz; ++i) {
    buff_j[i] = M_jcol[idxs_sorted[i]];
  }
  //exchange pointers to avoid copying
  delete[] M_jcol;
  M_jcol = buff_j;

  map_idxs_in_sorted_.resize(M_nnz);
 
  //
  // remove duplicates and update map_idxs_in_sorted_
  //

  index_type itleft = 0;
  index_type itright = 0;
  index_type currI = M_irow[0];
  index_type currJ = M_jcol[0];

  while(itright < M_nnz) {
 
    while(itright<M_nnz && M_irow[itright]==currI && M_jcol[itright]==currJ) {
      
      assert(itright<map_idxs_in_sorted_.size());

      //all duplicates will have the indexes in the sorted+unique array equal to itleft
      map_idxs_in_sorted_[idxs_sorted[itright]] = itleft;
      
      itright++;
    }
    //here itright points to the first elem not equal to (currI, currJ)
    
    //left runner set its value to curr (i,j) 
    M_irow[itleft] = currI;
    M_jcol[itleft] = currJ;

    assert(itleft<itright);
    itleft++;

    if(itright>=M_nnz) break;

    //update current (i,j) 
    currI = M_irow[itright];
    currJ = M_jcol[itright];
       
    //before incrementing itright, update map_idxs_in_sorted_ for it    
    map_idxs_in_sorted_[idxs_sorted[itright]] = itleft;

    itright++;
    
    assert(itleft<=M_nnz);
  }
  assert(itright == M_nnz);
  
  const size_type M_nnz_unique = itleft;
  
  hiopMatrixSparseCSRStorage* M = new hiopMatrixSparseCSRStorage(m, m, M_nnz_unique);
  M->form_from(m, m, M_nnz_unique, M_irow, M_jcol);
  
  delete[] M_irow;
  delete[] M_jcol;
  return M;
}

void hiopKKTLinSysCondensedSparse::add_matrices(hiopMatrixSparseCSRStorage& JtDiagJ,
                                                hiopMatrixSymSparseTriplet& Hess,
                                                hiopVector& Dx,
                                                double delta_wx,
                                                hiopMatrixSparseCSRStorage& M)
{
  const size_type M_nnz_dupl = map_idxs_in_sorted_.size();
  const size_type M_nnz = M.nnz();
  assert(M_nnz<=M_nnz_dupl);

  const size_type JtDJ_nnz = JtDiagJ.nnz();
  const size_type H_nnz_lowtr = map_H_lowtr_idxs_.size();
  const size_type H_nnz = Hess.numberOfNonzeros();
  const size_type m = Dx.get_size();
  assert(M_nnz_dupl == JtDJ_nnz+H_nnz+H_nnz_lowtr+m);
  
  //
  //update the values in M
  //
  double* M_values = M.values();
  for(int i=0; i<M_nnz; ++i) {
    M_values[i] = 0.;
  }

  index_type itnnz = 0;

  double* JtDJ_values = JtDiagJ.values();
  for(int it=0; it<JtDJ_nnz; ++it) {
    assert(map_idxs_in_sorted_[itnnz] < M_nnz);
    M_values[map_idxs_in_sorted_[itnnz]] += JtDJ_values[it];
    itnnz++;
  }
  assert(itnnz == JtDJ_nnz);

  double* H_values = Hess.M();
  for(int it=0; it<H_nnz; ++it) {
    assert(map_idxs_in_sorted_[itnnz] < M_nnz);
    M_values[map_idxs_in_sorted_[itnnz]] += H_values[it];
    itnnz++;
  }
  assert(itnnz == JtDJ_nnz+H_nnz);

  assert(H_nnz_lowtr <= H_nnz);
  //strictly lower triangle for H
  for(int it=0; it<H_nnz_lowtr; ++it) {
    assert(map_idxs_in_sorted_[itnnz] < M_nnz);
    M_values[map_idxs_in_sorted_[itnnz]] += H_values[map_H_lowtr_idxs_[it]];
    itnnz++;
  }

  // add D and delta_wx*I
  const double* d_arr = Dx.local_data_const();
  for(int i=0; i<m; i++) {
    assert(map_idxs_in_sorted_[itnnz] < M_nnz);
    M_values[map_idxs_in_sorted_[itnnz]] += (d_arr[i]+delta_wx);
    itnnz++; 
  }
  assert(itnnz == M_nnz_dupl);

  //M.print(stdout);
}
  
bool hiopKKTLinSysCondensedSparse::solve_compressed_direct(hiopVector& rx,
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

  assert(rhs_);
  assert(rhs_->get_size() == nx);

   /* (H+Dx+Jd^T*(Dd+delta_wd*I)*Jd)dx = rx + Jd^T*Dd*ryd + Jd^T*rd
   * dd = Jd*dx - ryd
   * dyd = (Dd+delta_wd*I)*dd - rd = (Dd+delta_wd*I)*Jd*dx - (Dd+delta_wd*I)*ryd - rd
   */
  rhs_->copyFrom(rx);

  //working buffers in the size of nineq/nd using output as storage
  hiopVector& Dd_x_ryd = dyd;
  Dd_x_ryd.copyFrom(ryd);
  Dd_x_ryd.componentMult(*Hd_);

  hiopVector& DD_x_ryd_plus_rd = Dd_x_ryd;
  DD_x_ryd_plus_rd.axpy(1.0, rd);

  Jac_dSp_->transTimesVec(1.0, *rhs_, 1.0, DD_x_ryd_plus_rd);

  //
  // solve
  //
  bool linsol_ok = linSys_->solve(*rhs_);
  
  if(false==linsol_ok) {
    return false;
  }
  dx.copyFrom(*rhs_);

  dd.copyFrom(ryd);
  Jac_dSp_->timesVec(-1.0, dd, 1.0, dx);

  dyd.copyFrom(dd);
  dyd.componentMult(*Hd_);
  dyd.axpy(-1.0, rd);
  return true;
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

  bool bret;

  nlp_->runStats.kkt.tmSolveInner.start();
  
  size_type nx = rx.get_size();
  size_type nd = rd.get_size();
  size_type nyd = ryd.get_size();

  // this is rhs used by the direct "condensed" solve
  if(rhs_ == NULL) {
    rhs_ = LinearAlgebraFactory::create_vector(nlp_->options->GetString("mem_space"), nx);
  }
  assert(rhs_->get_size() == nx);

  nlp_->log->write("RHS KKT_SPARSE_Condensed rx: ", rx,  hovIteration);
  nlp_->log->write("RHS KKT_SPARSE_Condensed rd: ", rd,  hovIteration);
  nlp_->log->write("RHS KKT_SPARSE_Condensed ryc:", ryc, hovIteration);
  nlp_->log->write("RHS KKT_SPARSE_Condensed ryd:", ryd, hovIteration);
#if 0
  bret = solve_compressed_direct(rx, rd, ryc, ryd, dx, dd, dyc, dyd);
  nlp_->runStats.kkt.tmSolveInner.stop();
#else
  
  if(nullptr == krylov_mat_opr_) {
    krylov_mat_opr_ = new hiopKKTMatVecOpr(this);
    krylov_prec_opr_ = new hiopKKTPrecondOpr(this);
    krylov_rhs_xdycyd_ = new hiopVectorPar(nx+nd+nyd);
    //TODO: memory space device
    bicgstab_ = new hiopBiCGStabSolver(nx+nd+nyd, "DEFAULT", krylov_mat_opr_, krylov_prec_opr_);
  }
  
  rx.copyToStarting(*krylov_rhs_xdycyd_, 0);
  rd.copyToStarting(*krylov_rhs_xdycyd_, nx);
  ryd.copyToStarting(*krylov_rhs_xdycyd_, nx+nd);
  
  const double tol_mu = 1e-2;
  double tol = std::min(mu_*tol_mu, 1e-6);
  bicgstab_->set_tol(tol);
  bicgstab_->set_x0(0.0);
  
  bret = bicgstab_->solve(*krylov_rhs_xdycyd_);
  nlp_->runStats.kkt.nIterRefinInner += bicgstab_->get_sol_num_iter();
  nlp_->runStats.kkt.tmSolveInner.stop();
  if(!bret) {
    nlp_->log->printf(hovWarning, "%s", bicgstab_->get_convergence_info().c_str());

    double tola = 10*mu_;
    double tolr = mu_*1e-1;
    
    if(bicgstab_->get_sol_rel_resid()>tolr || bicgstab_->get_sol_abs_resid()>tola) {
      return false;
    }

    //error out if one of the residuals is large
    if(bicgstab_->get_sol_abs_resid()>1e-2 || bicgstab_->get_sol_rel_resid()>1e-2) {
      return false;
    }
    bret = true;
  }

  nlp_->runStats.kkt.tmSolveRhsManip.start();
  dx.startingAtCopyFromStartingAt(0, *krylov_rhs_xdycyd_, 0);
  dd.startingAtCopyFromStartingAt(0, *krylov_rhs_xdycyd_, nx);
  dyd.startingAtCopyFromStartingAt(0, *krylov_rhs_xdycyd_, nx+nd);
  nlp_->runStats.kkt.tmSolveRhsManip.stop();
  
#endif

  
  if(perf_report_) {
    nlp_->log->printf(hovSummary, "(summary for linear solver from KKT_SPARSE_Condensed(direct))\n%s",
                      nlp_->runStats.linsolv.get_summary_last_solve().c_str());
  }

  
  nlp_->log->write("SOL KKT_SPARSE_Condensed dx: ", dx,  hovMatrices);
  nlp_->log->write("SOL KKT_SPARSE_Condensed dd: ", dd,  hovMatrices);
  nlp_->log->write("SOL KKT_SPARSE_Condensed dyc:", dyc, hovMatrices);
  nlp_->log->write("SOL KKT_SPARSE_Condensed dyd:", dyd, hovMatrices);
  return bret;
}


hiopLinSolverSymSparse*
hiopKKTLinSysCondensedSparse::determine_and_create_linsys(size_type nx, size_type nineq, size_type nnz)
{   
  if(linSys_) {
    return dynamic_cast<hiopLinSolverSymSparse*> (linSys_);
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
  return dynamic_cast<hiopLinSolverSymSparse*> (linSys_);
}

#if 0
SparseMatrixCSR* hiopKKTLinSysCondensedSparse::compute_linsys_eigen(const double& delta_wx)
{
  HessSp_ = dynamic_cast<hiopMatrixSymSparseTriplet*>(Hess_);
  Jac_dSp_ = dynamic_cast<const hiopMatrixSparseTriplet*>(Jac_d_);
  Jac_cSp_ = nullptr; //not used by this class
  const hiopMatrixSparseTriplet* JacTriplet = dynamic_cast<const hiopMatrixSparseTriplet*>(Jac_d_);
  size_type nx = HessSp_->n();
  size_type nineq = Jac_dSp_->m();

  hiopTimer t;
  
  t.start();
  SparseMatrixCSR JacD(nineq, nx);
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
  SparseMatrixCSR JacD_trans =  SparseMatrixCSR(JacD.transpose());
  t.stop();
  if(perf_report_) nlp_->log->printf(hovSummary, "JacD trans took       %.3f sec\n", t.getElapsedTime());

  
  t.reset(); t.start();
  SparseMatrixCSR Dd_mat(nineq, nineq);
  {
    std::vector<Triplet> tripletList;
    tripletList.reserve(Dd_->get_size());
    for(int i=0; i<Dd_->get_size(); i++) {
      tripletList.push_back(Triplet(i, i, Hd_->local_data()[i]));
    }
    Dd_mat.setFromTriplets(tripletList.begin(), tripletList.end());
  }
  t.stop(); 
  if(perf_report_) nlp_->log->printf(hovSummary, "Ddmat took       %.3f sec\n", t.getElapsedTime());

  
  t.reset(); t.start();
  SparseMatrixCSR DdxJ = Dd_mat * JacD;
  t.stop();
  if(perf_report_) nlp_->log->printf(hovSummary, "DdxJ took         %.3f sec\n", t.getElapsedTime());

  t.reset(); t.start();  
  SparseMatrixCSR JtxDdxJ = JacD_trans * DdxJ;
  t.stop();
  if(perf_report_) nlp_->log->printf(hovSummary, "JtxDdxJ took      %.3f sec\n", t.getElapsedTime());

  
  t.reset(); t.start();
  SparseMatrixCSR H(nx, nx);
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
  SparseMatrixCSR *KKTmat = new SparseMatrixCSR();
  *KKTmat = H + JtxDdxJ;
  t.stop();
  if(perf_report_) nlp_->log->printf(hovSummary, "H  + JtxDdxJ took         %.3f sec\n", t.getElapsedTime());
 
  t.reset(); t.start(); 
  SparseMatrixCSR Dx_mat(nx, nx);
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
  *KKTmat += Dx_mat;
  t.stop();
  if(perf_report_) nlp_->log->printf(hovSummary, "KKTmat += Dx_mat took         %.3f sec\n", t.getElapsedTime());

  t.reset(); t.start(); 
  KKTmat->makeCompressed();
  t.stop();
  if(perf_report_) nlp_->log->printf(hovSummary, "makeCompressed         %.3f sec\n", t.getElapsedTime());

  return KKTmat;
} 
#endif //HIOP_USE_EIGEN
/*******************************************************************************************************
 * hiopMatrixSparseCSRStorage
 *******************************************************************************************************/
hiopMatrixSparseCSRStorage::hiopMatrixSparseCSRStorage()
  : nrows_(0),
    ncols_(0),
    nnz_(0),
    irowptr_(nullptr),
    jcolind_(nullptr),
    values_(nullptr)
{
  
}
hiopMatrixSparseCSRStorage::hiopMatrixSparseCSRStorage(size_type m, size_type n, size_type nnz)
  : nrows_(m),
    ncols_(n),
    nnz_(nnz),
    irowptr_(nullptr),
    jcolind_(nullptr),
    values_(nullptr)
{
  alloc();
}
hiopMatrixSparseCSRStorage::~hiopMatrixSparseCSRStorage()
{
  dealloc();
}

void hiopMatrixSparseCSRStorage::alloc()
{
  assert(irowptr_ == nullptr);
  assert(jcolind_ == nullptr);
  assert(values_ == nullptr);

  irowptr_ = new index_type[nrows_+1];
  jcolind_ = new index_type[nnz_];
  values_ = new double[nnz_];
}


void hiopMatrixSparseCSRStorage::dealloc()
{
  delete[] irowptr_;
  delete[] jcolind_;
  delete[] values_;
  irowptr_ = nullptr;
  jcolind_ = nullptr;
  values_ = nullptr;
}

/**
 * Forms a CSR matrix from a sparse matrix in triplet format. Assumes input is ordered by
 * rows then by columns.
 */
bool hiopMatrixSparseCSRStorage::form_from(const hiopMatrixSparseTriplet& M)
{
  if(M.m()!=nrows_ || M.n()!=ncols_ || M.numberOfNonzeros()!=nnz_) {
    dealloc();
    
    nrows_ = M.m();
    ncols_ = M.n();
    nnz_ = M.numberOfNonzeros();

    alloc();
  }

  assert(nnz_>=0);
  if(nnz_<=0) {
    return true;
  }
  
  assert(irowptr_);
  assert(jcolind_);
  assert(values_);

  const index_type* Mirow = M.i_row();
  const index_type* Mjcol = M.j_col();
  const double* Mvalues  = M.M();

  //storage the row count
  std::vector<index_type> w(nrows_, 0);
  
  for(int it=0; it<nnz_; ++it) {
    const index_type row_idx = Mirow[it];

#ifndef NDEBUG
    if(it>0) {
      assert(Mirow[it] >= Mirow[it-1] && "row indexes of the triplet format are not ordered.");
      if(Mirow[it] == Mirow[it-1]) {
        assert(Mjcol[it] > Mjcol[it-1] && "col indexes of the triplet format are not ordered or unique.");
      }
    }
#endif
    assert(row_idx<nrows_ && row_idx>=0);
    assert(Mjcol[it]<ncols_ && Mjcol[it]>=0);

    w[row_idx]++;

    jcolind_[it] = Mjcol[it];
    values_[it] = Mvalues[it];
  }

  irowptr_[0] = 0;
  for(int i=0; i<nrows_; i++) {
    irowptr_[i+1] = irowptr_[i] + w[i];
  }
  assert(irowptr_[nrows_] == nnz_);
  return true;
}

bool hiopMatrixSparseCSRStorage::form_from(const size_type m,
                                           const size_type n,
                                           const size_type nnz,
                                           const index_type* Mirow,
                                           const index_type* Mjcol)
{
    if(m!=nrows_ || n!=ncols_ || nnz!=nnz_) {
    dealloc();
    
    nrows_ = m;
    ncols_ = n;
    nnz_ = nnz;

    alloc();
  }

  assert(nnz_>=0);
  if(nnz_<=0) {
    return true;
  }
  
  assert(irowptr_);
  assert(jcolind_);
  assert(values_);

  //storage for the row count
  std::vector<index_type> w(nrows_, 0);
  
  for(int it=0; it<nnz_; ++it) {
    const index_type row_idx = Mirow[it];

#ifndef NDEBUG
    if(it>0) {
      assert(Mirow[it] >= Mirow[it-1] && "row indexes of the triplet format are not ordered.");
      if(Mirow[it] == Mirow[it-1]) {
        assert(Mjcol[it] > Mjcol[it-1] && "col indexes of the triplet format are not ordered or unique.");
      }
    }
#endif
    assert(row_idx<nrows_ && row_idx>=0);
    assert(Mjcol[it]<ncols_ && Mjcol[it]>=0);

    w[row_idx]++;

    jcolind_[it] = Mjcol[it];
    //values_[it] = Mvalues[it];
  }

  irowptr_[0] = 0;
  for(int i=0; i<nrows_; i++) {
    irowptr_[i+1] = irowptr_[i] + w[i];
  }
  assert(irowptr_[nrows_] == nnz_);

  return true;
}

/**
 * Forms a CSR matrix representing the transpose of the sparse matrix in triplet format 
 * passed as argument. Assumes triplet format is ordered by rows then by columns.
 */
bool hiopMatrixSparseCSRStorage::form_transpose_from(const hiopMatrixSparseTriplet& M)
{
  if(M.m()!=ncols_ || M.n()!=nrows_ || M.numberOfNonzeros()!=nnz_) {
    dealloc();
    
    nrows_ = M.n();
    ncols_ = M.m();
    nnz_ = M.numberOfNonzeros();

    alloc();
  }

  assert(nnz_>=0);
  if(nnz_<=0) {
    return true;
  }
  
  assert(irowptr_);
  assert(jcolind_);
  assert(values_);

  const index_type* Mirow = M.i_row();
  const index_type* Mjcol = M.j_col();
  const double* Mvalues  = M.M();

  //keeps counts of nz on each row of this (later will also store row starts)
  index_type w[nrows_];
  
  // initialize nz per row to zero
  for(index_type i=0; i<nrows_; ++i) {
    w[i] = 0;
  }
  // count number of nonzeros in each row
  for(index_type it=0; it<nnz_; ++it) {
    assert(Mjcol[it]<nrows_);
    w[Mjcol[it]]++;
  }
  // cum sum in irowptr_ and set w to the row starts
  irowptr_[0] = 0;
  for(int i=1; i<=nrows_; ++i) {
    irowptr_[i] = irowptr_[i-1] + w[i-1];
    w[i-1] = irowptr_[i-1];
  }
  assert(irowptr_[nrows_] = nnz_);
  
  //populate jcolind_ and values_
  for(index_type it=0; it<nnz_; ++it) {
    const index_type row_idx = Mjcol[it];
    
    //index in nonzeros of this (transposed)
    const auto nz_idx = w[row_idx];
    assert(nz_idx<nnz_);
    
    //assign col and value
    jcolind_[nz_idx] = Mirow[it];
    values_[nz_idx] = Mvalues[it];
    assert(Mirow[it] < ncols_);
    
    //increase start for row 'row_idx'
    w[row_idx]++;

    assert(w[row_idx] <= irowptr_[row_idx+1]);
  }
 
#ifndef NDEBUG
  for(int i=0; i<nrows_; i++) {
    for(int itnz=irowptr_[i]+1; itnz<irowptr_[i+1]; ++itnz) {
      assert(jcolind_[itnz] > jcolind_[itnz-1] && "something wrong: col indexes not sorted or not unique");
    }
  }
#endif

  return true;
}


void hiopMatrixSparseCSRStorage::print(FILE* file, const
                                       char* msg/*=NULL*/,
                                       int maxRows/*=-1*/,
                                       int maxCols/*=-1*/,
                                       int rank/*=-1*/) const
{
  int myrank_ = 0;
  int numranks=1; //this is a local object => always print

  if(file==NULL) file = stdout;

  int max_elems = maxRows>=0 ? maxRows : nnz_;
  max_elems = std::min(max_elems, nnz_);
  
  if(myrank_==rank || rank==-1) {
    std::stringstream ss;
    if(NULL==msg) {
      if(numranks>1) {
        //fprintf(file,
        //        "matrix of size %d %d and nonzeros %d, printing %d elems (on rank=%d)\n",
        //        m(), n(), numberOfNonzeros(), max_elems, myrank_);
        ss << "matrix of size " << m() << " " << n() << " and nonzeros " 
           << nnz() << ", printing " <<  max_elems << " elems (on rank="
           << myrank_ << ")" << std::endl;
      } else {
        ss << "matrix of size " << m() << " " << n() << " and nonzeros " 
           << nnz() << ", printing " <<  max_elems << " elems" << std::endl;
        // fprintf(file,
        //      "matrix of size %d %d and nonzeros %d, printing %d elems\n",
        //      m(), n(), numberOfNonzeros(), max_elems);
      }
    } else {
      ss << msg << " ";
      //fprintf(file, "%s ", msg);
    }

    // using matlab indices
    //fprintf(file, "iRow_=[");
    ss << "iRow_=[";

    index_type itnz=0;
    for(int i=0; itnz<max_elems && i<nrows_; ++i) {
      for(itnz=irowptr_[i]; itnz<irowptr_[i+1] && itnz<max_elems; itnz++) {
        ss << (i+1) << "; ";
      }
    }
    //fprintf(file, "];\n");
    ss << "];" << std::endl;

    //fprintf(file, "jCol_=[");
    ss << "jCol_=[";
    itnz = 0;
    for(int i=0; itnz<max_elems && i<nrows_; ++i) {
      for(itnz=irowptr_[i]; itnz<irowptr_[i+1] && itnz<max_elems; ++itnz) {
        ss << (jcolind_[itnz]+1) << "; ";
      }
    }
    //fprintf(file, "];\n");
    ss << "];" << std::endl;
    
    //fprintf(file, "v=[");
    ss << "v=[";
    ss << std::scientific << std::setprecision(16);
    itnz = 0;
    for(int i=0; itnz<max_elems && i<nrows_; ++i) {
      for(itnz=irowptr_[i]; itnz<irowptr_[i+1] && itnz<max_elems; ++itnz) {
        ss << values_[itnz] << "; ";
      }
    }
    //fprintf(file, "];\n");
    ss << "];" << std::endl;
    
    fprintf(file, "%s", ss.str().c_str());
  }
}

void hiopMatrixSparseCSRStorage::
times_diag_times_mat(const hiopVector& diag,
                     const hiopMatrixSparseCSRStorage& Y,
                     hiopMatrixSparseCSRStorage& M)
{
  const index_type* irowptrY = Y.irowptr();
  const index_type* jcolindY = Y.jcolind();
  const double* valuesY = Y.values();
  
  const index_type* irowptrX = irowptr_;
  const index_type* jcolindX = jcolind_;
  const double* valuesX = values_;

  index_type* irowptrM = M.irowptr();
  index_type* jcolindM = M.jcolind();
  double* valuesM = M.values();
  
  const index_type m = this->m();
  const index_type n = Y.n();

  const index_type K = this->n();
  assert(Y.m() == K);
  assert(diag.get_size() == K);

  const double* d = diag.local_data_const();

  double* W = new double[n];
  char* flag=new char[n];

  for(int it=0; it<n; it++) W[it] = 0.0;

  int nnzM=0;
  for(int i=0; i<m; i++) {
    memset(flag, 0, m);

    assert(nnzM<M.nnz());
    //start row i of M
    irowptrM[i]=nnzM;
    
    for(int px=irowptrX[i]; px<irowptrX[i+1]; px++) { 
      const auto k = jcolindX[px]; //X[i,k] is non-zero
      assert(k<K);
      
      const double val = valuesX[px]*d[k];

      //iterate the row k of Y and scatter the values into W
      for(int py=irowptrY[k]; py<irowptrY[k+1]; py++) {
	const auto j = jcolindY[py];
        assert(j<n);
        
	//we have A[k,j]
	if(flag[j]==0) {
          assert(nnzM<M.nnz());
          
	  jcolindM[nnzM++]=j;
	  flag[j]=1;
	}
	
	W[j] += (valuesY[py]*val);
      }
    }
    //gather the values into the i-th row M
    for(int p=irowptrM[i]; p<nnzM; p++) {
      const auto j = jcolindM[p];
      valuesM[p] = W[j];
      W[j] = 0.0;
    }
  }
  irowptrM[n] = nnzM;
  delete[] W;
  delete[] flag;
}
void hiopMatrixSparseCSRStorage::
times_diag_times_mat_numeric(const hiopVector& diag,
                             const hiopMatrixSparseCSRStorage& Y,
                             hiopMatrixSparseCSRStorage& M)
{
  const index_type* irowptrY = Y.irowptr();
  const index_type* jcolindY = Y.jcolind();
  const double* valuesY = Y.values();
  
  const index_type* irowptrX = irowptr_;
  const index_type* jcolindX = jcolind_;
  const double* valuesX = values_;

  index_type* irowptrM = M.irowptr();
  index_type* jcolindM = M.jcolind();
  double* valuesM = M.values();
  
  const index_type m = this->m();
  const index_type n = Y.n();

  const index_type K = this->n();
  assert(Y.m() == K);
  assert(diag.get_size() == K);

  const double* d = diag.local_data_const();

  double* W = new double[n];

  for(int it=0; it<n; it++) W[it] = 0.0;

  for(int i=0; i<m; i++) {
    for(int px=irowptrX[i]; px<irowptrX[i+1]; px++) { 
      const auto k = jcolindX[px]; //X[i,k] is non-zero
      assert(k<K);
      
      const double val = valuesX[px]*d[k];

      //iterate the row k of Y and scatter the values into W
      for(int py=irowptrY[k]; py<irowptrY[k+1]; py++) {
        assert(jcolindY[py]<n);        
	W[jcolindY[py]] += (valuesY[py]*val);
      }
    }
    //gather the values into the i-th row M
    for(int p=irowptrM[i]; p<irowptrM[i+1]; ++p) {
      const auto j = jcolindM[p];
      valuesM[p] = W[j];
      W[j] = 0.0;
    }
  }
  delete[] W;
}


/**
 *  M = X*D*Y -> computes nnz in M and allocates M 
 * By convention, M is mxn, X is mxK, Y is Kxn, and D is size K.
 * 
 * The algorithm uses the fact that the sparsity pattern of the i-th row of M is
 *           K
 * M_{i*} = sum x_{ik} Y_{j*}   (see Tim Davis book p.17)
 *          k=1
 * Therefore, to get sparsity pattern of the i-th row of M:
 *  1. we iterate over nonzeros (i,k) in the i-th row of X
 *  2. for each such k we iterate over the nonzeros (k,j) in the k-th row of Y and 
 *  3. count (i,j) as nonzero of M 
 */
hiopMatrixSparseCSRStorage* hiopMatrixSparseCSRStorage::
times_diag_times_mat_init(const hiopMatrixSparseCSRStorage& Y)
{
  const index_type* irowptrY = Y.irowptr();
  const index_type* jcolindY = Y.jcolind();

  const index_type* irowptrX = irowptr_;
  const index_type* jcolindX = jcolind_;

  const index_type m = this->m();
  const index_type n = Y.n();

  const index_type K = this->n();
  assert(Y.m() == K);
  
  index_type nnzM = 0;
  // count the number of entries in the result M
  char* flag = new char[m];
  for(int i=0; i<m; i++) {
    //reset flag 
    memset(flag, 0, m*sizeof(char));

    for(int pt=irowptrX[i]; pt<irowptrX[i+1]; pt++) {
      //X[i,k] is nonzero
      const index_type k = jcolindX[pt];
      assert(k<K);

      //add the nonzero pattern of row k of Y to M
      for(int p=irowptrY[k]; p<irowptrY[k+1]; p++) {
	const index_type j = jcolindY[p];
        assert(j<n);
        
        //Y[k,j] is non zero, hence M[i,j] is non zero
	if(flag[j]==0) {
          //only count once
	  nnzM++;
	  flag[j]=1;
	}
      }
    }
  }
  assert(nnzM>=0); //overflow?!?

  delete[] flag;

  //allocate result M
  return new hiopMatrixSparseCSRStorage(m, n, nnzM);
}

/// Extract the diagonal 
void hiopMatrixSparseCSRStorage::get_diagonal(hiopVector& diag) const
{
  auto& D = dynamic_cast<hiopVectorPar&>(diag);
  assert(m() == D.get_size());
  assert(n() == D.get_size());

  double* Da = D.local_data();

  for(index_type i=0; i<m(); i++) {
    Da[i] = 0.0; //just in case there is no nnz for the diag on this row
    for(index_type pt=irowptr_[i]; pt<irowptr_[i+1]; ++pt) {
      if(jcolind_[pt] == i) {
        Da[i] = values_[pt];
      }
    }
  }
}
#if 0
/// Extract the diagonal 
void hiopMatrixSparseCSRStorage::add_diagonal(double val)
{
  assert(m() == n());

  for(index_type i=0; i<m(); i++) {
    for(index_type pt=irowptr_[i]; pt<irowptr_[i+1]; ++pt) {
      if(jcolind_[pt] == i) {
        values_[pt] += val;
      }
    }
  }
}



/// Performs multiplication this = D * this
void hiopMatrixSparseCSRStorage::diag_mult_left(hiopVector& diag)
{
  auto& D = dynamic_cast<hiopVectorPar&>(diag);
  assert(m() == D.get_size());
  assert(n() == D.get_size());

  const double* Da = D.local_data();
  
  for(index_type i=0; i<m(); i++) {
    for(index_type pt=irowptr_[i]; pt<irowptr_[i+1]; ++pt) {
      values_[pt] *= Da[i];
    }
  }
}
/// Performs multiplication this = this * D
void hiopMatrixSparseCSRStorage::diag_mult_right(hiopVector& diag)
{
  auto& D = dynamic_cast<hiopVectorPar&>(diag);
  assert(m() == D.get_size());
  assert(n() == D.get_size());

  const double* Da = D.local_data();
  
  for(index_type i=0; i<m(); i++) {
    for(index_type pt=irowptr_[i]; pt<irowptr_[i+1]; ++pt) {
      values_[pt] *= Da[jcolind_[pt]];
    }
  } 
}
#endif

} // end of namespace
