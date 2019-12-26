#include "hiopKKTLinSysMDS.hpp"

namespace hiop
{

  hiopKKTLinSysCompressedMDSXYcYd::hiopKKTLinSysCompressedMDSXYcYd(hiopNlpFormulation* nlp_)
    : hiopKKTLinSysCompressedXYcYd(nlp_), linSys(NULL), rhs(NULL), _buff_xs(NULL),
      Hxs(NULL), HessMDS(NULL), Jac_cMDS(NULL), Jac_dMDS(NULL)
  {
    nlpMDS = dynamic_cast<hiopNlpMDS*>(nlp);
    assert(nlpMDS);
  }

  hiopKKTLinSysCompressedMDSXYcYd::~hiopKKTLinSysCompressedMDSXYcYd()
  {
    delete rhs;
    delete linSys;
    delete _buff_xs;
    delete Hxs;
  }

  bool hiopKKTLinSysCompressedMDSXYcYd::update(const hiopIterate* iter_, 
					       const hiopVector* grad_f_, 
					       const hiopMatrix* Jac_c_, const hiopMatrix* Jac_d_, hiopMatrix* Hess_)
  {
    if(!nlpMDS) { assert(false); return false; }
    nlp->runStats.tmSolverInternal.start();

    iter = iter_; grad_f = dynamic_cast<const hiopVectorPar*>(grad_f_); Jac_c = Jac_c_; Jac_d = Jac_d_; Hess=Hess_;

    HessMDS = dynamic_cast<hiopMatrixSymBlockDiagMDS*>(Hess);
    if(!HessMDS) { assert(false); return false; }

    Jac_cMDS = dynamic_cast<const hiopMatrixMDS*>(Jac_c);
    if(!Jac_cMDS) { assert(false); return false; }

    Jac_dMDS = dynamic_cast<const hiopMatrixMDS*>(Jac_d);
    if(!Jac_dMDS) { assert(false); return false; }

    int nxs = HessMDS->n_sp(), nxd = HessMDS->n_de(), nx = HessMDS->n(); 
    int neq = Jac_cMDS->m(), nineq = Jac_dMDS->m();

    assert(nx==nxs+nxd);
    assert(nx==Jac_cMDS->n_sp()+Jac_cMDS->n_de());
    assert(nx==Jac_dMDS->n_sp()+Jac_dMDS->n_de());

    if(NULL==linSys) {
      int n = nxd + neq + nineq;

      if(nlp->options->GetString("compute_mode")=="hybrid") {
#ifdef HIOP_USE_MAGMA
	linSys = new hiopLinSolverIndefDenseMagma(n, nlp);
#else
	linSys = new hiopLinSolverIndefDenseLapack(n, nlp);
#endif
      } else {
	linSys = new hiopLinSolverIndefDenseLapack(n, nlp);
      }
    }

    //
    //the actual update of the linear system
    //
    hiopMatrixDense& Msys = linSys->sysMatrix();
    Msys.setToZero();

    int alpha = 1.;
    HessMDS->de_mat()->addUpperTriangleToSymDenseMatrixUpperTriangle(0, alpha, Msys);
    Jac_cMDS->de_mat()->transAddToSymDenseMatrixUpperTriangle(0, nxd,     alpha, Msys);
    Jac_dMDS->de_mat()->transAddToSymDenseMatrixUpperTriangle(0, nxd+neq, alpha, Msys);

    assert(Dx->get_local_size() == nxs+nxd);
    Dx->setToZero();
    Dx->axdzpy_w_pattern(1.0, *iter->zl, *iter->sxl, nlp->get_ixl());
    Dx->axdzpy_w_pattern(1.0, *iter->zu, *iter->sxu, nlp->get_ixu());
    nlp->log->write("Dx in KKT", *Dx, hovMatrices);

nlp->log->write("KKT XdenseDYcYd Linsys:", Msys, hovMatrices);

    //update -> add Dxd to (1,1) block of KKT matrix (Hd = HessMDS->de_mat already added above)
    Msys.addSubDiagonal(0, alpha, *Dx, nxs, nxd);

nlp->log->write("KKT XdenseDYcYd Linsys:", Msys, hovMatrices);

    //build the diagonal Hxs = Hsparse+Dxs
    if(NULL == Hxs) Hxs = new hiopVectorPar(nxs); assert(Hxs);
    Hxs->startingAtCopyFromStartingAt(0, *Dx, 0);
    HessMDS->sp_mat()->startingAtAddSubDiagonalToStartingAt(0, alpha, *Hxs, 0);
    nlp->log->write("Hxs in KKT", *Hxs, hovMatrices);

    //add - Jac_c_sp * (Hxs)^{-1} Jac_c_sp^T to diagonal block linSys starting at (nxd, nxd)
    alpha=-1;
    //need to remove const since Jac_cMDS->sp_mat() needs to build some index arrays internaly (for fast multiplication)
    hiopMatrixSparseTriplet* Jac_cMDS_spmat = const_cast<hiopMatrixSparseTriplet*>(Jac_cMDS->sp_mat()); assert(Jac_cMDS_spmat);
    Jac_cMDS_spmat->addMatTimesDinvTimesMatTransToDiagBlockOfSymDenseMatrixUpperTriangle(nxd, alpha, *Hxs, Msys); 

nlp->log->write("KKT XdenseDYcYd Linsys:", Msys, hovMatrices);
nlp->log->write("Jac_c:", *Jac_cMDS, hovMatrices);
    //add - Jac_d_sp * (Hxs)^{-1} Jac_d_sp^T to diagonal block linSys starting at (nxd+neq, nxd+neq)
    //as above, remove const-ness
    hiopMatrixSparseTriplet* Jac_dMDS_spmat = const_cast<hiopMatrixSparseTriplet*>(Jac_dMDS->sp_mat()); assert(Jac_dMDS_spmat);
    Jac_dMDS_spmat->addMatTimesDinvTimesMatTransToDiagBlockOfSymDenseMatrixUpperTriangle(nxd+neq, alpha, *Hxs, Msys); 

nlp->log->write("Jac_d:", *Jac_dMDS, hovMatrices);

nlp->log->write("KKT XdenseDYcYd Linsys:", Msys, hovMatrices);

    //add -{Dd}^{-1}
    //Dd=(Sdl)^{-1}Vu + (Sdu)^{-1}Vu
    Dd_inv->setToZero();
    Dd_inv->axdzpy_w_pattern(1.0, *iter->vl, *iter->sdl, nlp->get_idl());
    Dd_inv->axdzpy_w_pattern(1.0, *iter->vu, *iter->sdu, nlp->get_idu());
#ifdef HIOP_DEEPCHECKS
    assert(true==Dd_inv->allPositive());
#endif 
    Dd_inv->invert();
    
    alpha=-1.;
    Msys.addSubDiagonal(alpha, nxd+neq, *Dd_inv);



    nlp->log->write("KKT XdenseDYcYd Linsys:", Msys, hovMatrices);

    linSys->matrixChanged();

    nlp->runStats.tmSolverInternal.stop();
    return true;
  }

  void hiopKKTLinSysCompressedMDSXYcYd::solveCompressed(hiopVectorPar& rx, hiopVectorPar& ryc, hiopVectorPar& ryd,
							hiopVectorPar& dx, hiopVectorPar& dyc, hiopVectorPar& dyd)
  {
    if(!nlpMDS)   { assert(false); return; }
    if(!HessMDS)  { assert(false); return; }
    if(!Jac_cMDS) { assert(false); return; }
    if(!Jac_dMDS) { assert(false); return; }

    int nx=rx.get_size(), nyc=ryc.get_size(), nyd=ryd.get_size();
    int nxsp=Hxs->get_size(); assert(nxsp<=nx);
    int nxde = nlpMDS->nx_de();
    assert(nxsp+nxde==nx);
    if(this->rhs == NULL) rhs = new hiopVectorPar(nxde+nyc+nyd);
    if(this->_buff_xs==NULL) _buff_xs = new hiopVectorPar(nxsp);

    nlp->log->write("RHS KKT XDycYd rx: ", rx,  hovIteration);
    nlp->log->write("RHS KKT XDycYd ryc:", ryc, hovIteration);
    nlp->log->write("RHS KKT XDycYd ryd:", ryd, hovIteration);

    hiopVectorPar& rxs = *_buff_xs;
    //rxs = Hxs^{-1} * rx_sparse 
    rx.startingAtCopyToStartingAt(0, rxs, 0, nxsp);
    rxs.componentDiv(*Hxs);

    //ryc = ryc - Jac_c_sp * Hxs^{-1} * rxs
    Jac_cMDS->sp_mat()->timesVec(1.0, ryc, -1., rxs);
    //ryd = ryd - Jac_d_sp * Hxs^{-1} * rxs
    Jac_dMDS->sp_mat()->timesVec(1.0, ryd, -1., rxs);

    //
    // for the rhs for the MDS linSys
    //
    //rhs[0:nxde-1] = rx[nxs:(nxsp+nxde-1)]
    rx.startingAtCopyToStartingAt(nxsp, *rhs, 0, nxde);
    //rhs[nxde:nxde+nyc-1] = ryc
    ryc.copyToStarting(*rhs, nxde);
    //ths[nxde+nyc:nxde+nyc+nyd-1] = ryd
    ryd.copyToStarting(*rhs, nxde+nyc);

    //
    // the solve
    //
    linSys->solve(*rhs);

    //
    // unpack 
    //
    rhs->startingAtCopyToStartingAt(0,        dx,  nxde);
    rhs->startingAtCopyToStartingAt(nxde,     dyc, 0);   
    rhs->startingAtCopyToStartingAt(nxde+nyc, dyd, 0);

    //
    // compute dxs
    //
    hiopVectorPar& dxs = *_buff_xs;
    // dxs = (Hxs)^{-1} ( rxs - Jac_c_sp^T dyc - Jac_d_sp^T dyd)
    rx.startingAtCopyToStartingAt(0, dxs, 0, nxsp);
    Jac_cMDS->sp_mat()->transTimesVec(1., dxs, -1., dyc);
    Jac_dMDS->sp_mat()->transTimesVec(1., dxs, -1., dyd);
    dxs.componentDiv(*Hxs);
    //copy to dx
    dxs.startingAtCopyToStartingAt(0, dx, 0);

    nlp->log->write("SOL KKT XYcYd dx: ", dx,  hovMatrices);
    nlp->log->write("SOL KKT XYcYd dyc:", dyc, hovMatrices);
    nlp->log->write("SOL KKT XYcYd dyd:", dyd, hovMatrices);
  
  }
} // end of namespace
