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


#include "hiopKKTLinSysMDS.hpp"

namespace hiop
{

  hiopKKTLinSysCompressedMDSXYcYd::hiopKKTLinSysCompressedMDSXYcYd(hiopNlpFormulation* nlp)
    : hiopKKTLinSysCompressedXYcYd(nlp), linSys_(NULL), rhs_(NULL), _buff_xs_(NULL),
      Hxs_(NULL), HessMDS_(NULL), Jac_cMDS_(NULL), Jac_dMDS_(NULL),
      write_linsys_counter_(-1), csr_writer_(nlp)
  {
    nlpMDS_ = dynamic_cast<hiopNlpMDS*>(nlp_);
    assert(nlpMDS_);
  }

  hiopKKTLinSysCompressedMDSXYcYd::~hiopKKTLinSysCompressedMDSXYcYd()
  {
    delete rhs_;
    delete linSys_;
    delete _buff_xs_;
    delete Hxs_;
  }

  bool hiopKKTLinSysCompressedMDSXYcYd::update(const hiopIterate* iter, 
					       const hiopVector* grad_f, 
					       const hiopMatrix* Jac_c,
					       const hiopMatrix* Jac_d,
					       hiopMatrix* Hess)
  {
    if(!nlpMDS_) { assert(false); return false; }
    nlp_->runStats.tmSolverInternal.start();

    iter_ = iter;
    grad_f_ = dynamic_cast<const hiopVectorPar*>(grad_f);
    Jac_c = Jac_c; Jac_d_ = Jac_d_; Hess_=Hess;

    HessMDS_ = dynamic_cast<hiopMatrixSymBlockDiagMDS*>(Hess_);
    if(!HessMDS_) { assert(false); return false; }

    Jac_cMDS_ = dynamic_cast<const hiopMatrixMDS*>(Jac_c_);
    if(!Jac_cMDS_) { assert(false); return false; }

    Jac_dMDS_ = dynamic_cast<const hiopMatrixMDS*>(Jac_d_);
    if(!Jac_dMDS_) { assert(false); return false; }

    int nxs = HessMDS_->n_sp(), nxd = HessMDS_->n_de(), nx = HessMDS_->n(); 
    int neq = Jac_cMDS_->m(), nineq = Jac_dMDS_->m();

    assert(nx==nxs+nxd);
    assert(nx==Jac_cMDS_->n_sp()+Jac_cMDS_->n_de());
    assert(nx==Jac_dMDS_->n_sp()+Jac_dMDS_->n_de());

    if(NULL==linSys_) {
      int n = nxd + neq + nineq;

      if(nlp_->options->GetString("compute_mode")=="hybrid") {
#ifdef HIOP_USE_MAGMA
	nlp_->log->printf(hovScalars, "LinSysMDSXYcYd: Magma for a matrix of size %d\n", n);
	linSys_ = new hiopLinSolverIndefDenseMagma(n, nlp_);
#else
	nlp_->log->printf(hovScalars, "LinSysMDSXYcYd: Lapack for a matrix of size %d\n", n);
	linSys_ = new hiopLinSolverIndefDenseLapack(n, nlp_);
#endif
      } else {
	nlp_->log->printf(hovScalars, "LinSysMDSXYcYd: Lapack for a matrix of size %d\n", n);
	linSys_ = new hiopLinSolverIndefDenseLapack(n, nlp_);
      }
    }

    //
    //the actual update of the linear system
    //
    hiopMatrixDense& Msys = linSys_->sysMatrix();
    Msys.setToZero();

    int alpha = 1.;
    HessMDS_->de_mat()->addUpperTriangleToSymDenseMatrixUpperTriangle(0, alpha, Msys);
    Jac_cMDS_->de_mat()->transAddToSymDenseMatrixUpperTriangle(0, nxd,     alpha, Msys);
    Jac_dMDS_->de_mat()->transAddToSymDenseMatrixUpperTriangle(0, nxd+neq, alpha, Msys);

    assert(Dx_->get_local_size() == nxs+nxd);
    Dx_->setToZero();
    Dx_->axdzpy_w_pattern(1.0, *iter->zl, *iter->sxl, nlp_->get_ixl());
    Dx_->axdzpy_w_pattern(1.0, *iter->zu, *iter->sxu, nlp_->get_ixu());
    nlp_->log->write("Dx in KKT", *Dx_, hovMatrices);

    //update -> add Dxd to (1,1) block of KKT matrix (Hd = HessMDS_->de_mat already added above)
    Msys.addSubDiagonal(0, alpha, *Dx_, nxs, nxd);

    //build the diagonal Hxs = Hsparse+Dxs
    if(NULL == Hxs_) Hxs_ = new hiopVectorPar(nxs); assert(Hxs_);
    Hxs_->startingAtCopyFromStartingAt(0, *Dx_, 0);
    HessMDS_->sp_mat()->startingAtAddSubDiagonalToStartingAt(0, alpha, *Hxs_, 0);
    nlp_->log->write("Hxs in KKT", *Hxs_, hovMatrices);

    //add - Jac_c_sp * (Hxs)^{-1} Jac_c_sp^T to diagonal block linSys starting at (nxd, nxd)
    alpha = -1.;
    Jac_cMDS_->sp_mat()->addMDinvMtransToDiagBlockOfSymDeMatUTri(nxd, alpha, *Hxs_, Msys); 

    alpha = -1.;
    //add - Jac_d_sp * (Hxs)^{-1} Jac_d_sp^T to diagonal block linSys starting at (nxd+neq, nxd+neq)
    Jac_dMDS_->sp_mat()->addMDinvMtransToDiagBlockOfSymDeMatUTri(nxd+neq, alpha, *Hxs_, Msys); 

    alpha = -1.;
    Jac_cMDS_->sp_mat()->addMDinvNtransToSymDeMatUTri(nxd, nxd+neq, alpha, *Hxs_, *Jac_dMDS_->sp_mat(), Msys);

    //add -{Dd}^{-1}
    //Dd=(Sdl)^{-1}Vu + (Sdu)^{-1}Vu
    Dd_inv_->setToZero();
    Dd_inv_->axdzpy_w_pattern(1.0, *iter->vl, *iter->sdl, nlp_->get_idl());
    Dd_inv_->axdzpy_w_pattern(1.0, *iter->vu, *iter->sdu, nlp_->get_idu());
#ifdef HIOP_DEEPCHECKS
    assert(true==Dd_inv_->allPositive());
#endif 
    Dd_inv_->invert();
    
    alpha=-1.;
    Msys.addSubDiagonal(alpha, nxd+neq, *Dd_inv_);

    nlp_->log->write("KKT MDS XdenseDYcYd Linsys:", Msys, hovMatrices);

        //write matrix to file if requested
    if(nlp_->options->GetString("write_kkt") == "yes") write_linsys_counter_++;
    if(write_linsys_counter_>=0) csr_writer_.writeMatToFile(Msys, write_linsys_counter_); 

    //factorization
    linSys_->matrixChanged();

    nlp_->runStats.tmSolverInternal.stop();
    return true;
  }

  void hiopKKTLinSysCompressedMDSXYcYd::
  solveCompressed(hiopVectorPar& rx, hiopVectorPar& ryc, hiopVectorPar& ryd,
		  hiopVectorPar& dx, hiopVectorPar& dyc, hiopVectorPar& dyd)
  {
    if(!nlpMDS_)   { assert(false); return; }
    if(!HessMDS_)  { assert(false); return; }
    if(!Jac_cMDS_) { assert(false); return; }
    if(!Jac_dMDS_) { assert(false); return; }

    int nx=rx.get_size(), nyc=ryc.get_size(), nyd=ryd.get_size();
    int nxsp=Hxs_->get_size(); assert(nxsp<=nx);
    int nxde = nlpMDS_->nx_de();
    assert(nxsp+nxde==nx);
    if(rhs_ == NULL) rhs_ = new hiopVectorPar(nxde+nyc+nyd);
    if(_buff_xs_==NULL) _buff_xs_ = new hiopVectorPar(nxsp);

    nlp_->log->write("RHS KKT MDS XDycYd rx: ", rx,  hovIteration);
    nlp_->log->write("RHS KKT MDS XDycYd ryc:", ryc, hovIteration);
    nlp_->log->write("RHS KKT MDS XDycYd ryd:", ryd, hovIteration);

    hiopVectorPar& rxs = *_buff_xs_;
    //rxs = Hxs^{-1} * rx_sparse 
    rx.startingAtCopyToStartingAt(0, rxs, 0, nxsp);
    rxs.componentDiv(*Hxs_);

    //ryc = ryc - Jac_c_sp * Hxs^{-1} * rxs
    //use dyc as working buffer to avoid altering ryc, which refers directly in the hiopResidual class
    assert(dyc.get_size()==ryc.get_size());
    dyc.copyFrom(ryc);
    Jac_cMDS_->sp_mat()->timesVec(1.0, dyc, -1., rxs);

    //ryd = ryd - Jac_d_sp * Hxs^{-1} * rxs
    Jac_dMDS_->sp_mat()->timesVec(1.0, ryd, -1., rxs);

    //
    // form the rhs for the MDS linSys
    //
    //rhs[0:nxde-1] = rx[nxs:(nxsp+nxde-1)]
    rx.startingAtCopyToStartingAt(nxsp, *rhs_, 0, nxde);
    //rhs[nxde:nxde+nyc-1] = ryc
    dyc.copyToStarting(*rhs_, nxde);
    //ths[nxde+nyc:nxde+nyc+nyd-1] = ryd
    ryd.copyToStarting(*rhs_, nxde+nyc);

    if(write_linsys_counter_>=0) csr_writer_.writeRhsToFile(*rhs_, write_linsys_counter_);

    //
    // solve
    //
    linSys_->solve(*rhs_);

    if(write_linsys_counter_>=0) csr_writer_.writeSolToFile(*rhs_, write_linsys_counter_);

    //
    // unpack 
    //
    rhs_->startingAtCopyToStartingAt(0,        dx,  nxsp, nxde);
    rhs_->startingAtCopyToStartingAt(nxde,     dyc, 0);   
    rhs_->startingAtCopyToStartingAt(nxde+nyc, dyd, 0);

    //
    // compute dxs
    //
    hiopVectorPar& dxs = *_buff_xs_;
    // dxs = (Hxs)^{-1} ( rxs - Jac_c_sp^T dyc - Jac_d_sp^T dyd)
    rx.startingAtCopyToStartingAt(0, dxs, 0, nxsp);
    Jac_cMDS_->sp_mat()->transTimesVec(1., dxs, -1., dyc);
    Jac_dMDS_->sp_mat()->transTimesVec(1., dxs, -1., dyd);
    dxs.componentDiv(*Hxs_);
    //copy to dx
    dxs.startingAtCopyToStartingAt(0, dx, 0);

    nlp_->log->write("SOL KKT MDS XYcYd dx: ", dx,  hovMatrices);
    nlp_->log->write("SOL KKT MDS XYcYd dyc:", dyc, hovMatrices);
    nlp_->log->write("SOL KKT MDS XYcYd dyd:", dyd, hovMatrices);
  
  }
} // end of namespace
