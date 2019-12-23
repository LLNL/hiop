#include "hiopKKTLinSysMDS.hpp"


namespace hiop
{

  hiopKKTLinSysCompressedMDSXYcYd::hiopKKTLinSysCompressedMDSXYcYd(hiopNlpFormulation* nlp_)
    : hiopKKTLinSysCompressedXYcYd(nlp_), linSys(NULL), rhs(NULL), Hxs(NULL)
  {
    nlpMDS = dynamic_cast<hiopNlpMDS*>(nlp);
    assert(nlpMDS);
  }

  hiopKKTLinSysCompressedMDSXYcYd::~hiopKKTLinSysCompressedMDSXYcYd()
  {
    delete rhs;
    delete linSys;
  }

  bool hiopKKTLinSysCompressedMDSXYcYd::update(const hiopIterate* iter_, 
					       const hiopVector* grad_f_, 
					       const hiopMatrix* Jac_c_, const hiopMatrix* Jac_d_, hiopMatrix* Hess_)
  {
    if(!nlpMDS) { assert(false); return false; }
    nlp->runStats.tmSolverInternal.start();

    iter = iter_; grad_f = dynamic_cast<const hiopVectorPar*>(grad_f_); Jac_c = Jac_c_; Jac_d = Jac_d_; Hess=Hess_;

    hiopMatrixSymBlockDiagMDS* HessMDS = dynamic_cast<hiopMatrixSymBlockDiagMDS*>(Hess);
    if(!HessMDS) { assert(false); return false; }

    const hiopMatrixMDS* Jac_cMDS = dynamic_cast<const hiopMatrixMDS*>(Jac_c);
    if(!Jac_cMDS) { assert(false); return false; }

    const hiopMatrixMDS* Jac_dMDS = dynamic_cast<const hiopMatrixMDS*>(Jac_d);
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

    //update -> add Dxd to (1,1) block of KKT matrix (Hd = HessMDS->de_mat already added above)
    Msys.addSubDiagonal(0, alpha, *Dx, nxs, nxd);

    //build the diagonal Hxs = Hsparse+Dxs
    if(NULL == Hxs) Hxs = new hiopVectorPar(nxs); assert(Hxs);
    Hxs->startingAtCopyFromStartingAt(0, *Dx, 0);
    HessMDS->sp_mat()->startingAtAddSubDiagonalToStartingAt(0, alpha, *Hxs, 0);

    alpha=-1;
    //need to remove const since Jac_cMDS->sp_mat() needs to build some index arrays internaly (for fast multiplication)
    hiopMatrixSparseTriplet* Jac_cMDS_spmat = const_cast<hiopMatrixSparseTriplet*>(Jac_cMDS->sp_mat()); assert(Jac_cMDS_spmat);
    Jac_cMDS_spmat->addMatTimesDinvTimesMatTransToDiagBlockOfSymDenseMatrixUpperTriangle(nxd, alpha, *Hxs, Msys); 

    //same as above
    hiopMatrixSparseTriplet* Jac_dMDS_spmat = const_cast<hiopMatrixSparseTriplet*>(Jac_dMDS->sp_mat()); assert(Jac_dMDS_spmat);
    Jac_dMDS_spmat->addMatTimesDinvTimesMatTransToDiagBlockOfSymDenseMatrixUpperTriangle(nxd+neq, alpha, *Hxs, Msys); 

    nlp->log->write("KKT XDYcYd Linsys:", Msys, hovMatrices);

    linSys->matrixChanged();

    nlp->runStats.tmSolverInternal.stop();
    return true;
  }

  void hiopKKTLinSysCompressedMDSXYcYd::solveCompressed(hiopVectorPar& rx, hiopVectorPar& ryc, hiopVectorPar& ryd,
							hiopVectorPar& dx, hiopVectorPar& dyc, hiopVectorPar& dyd)
  {
    if(!nlpMDS) { assert(false); return; }

    // int nx=rx.get_size(), nyc=ryc.get_size(), nyd=ryd.get_size();

    // if(rhsXYcYd == NULL) rhsXYcYd = new hiopVectorPar(nx+nyc+nyd);

    // rx. copyToStarting(*rhsXYcYd, 0);
    // ryc.copyToStarting(*rhsXYcYd, nx);
    // ryd.copyToStarting(*rhsXYcYd, nx+nyc);
    
    // linSys->solve(*rhsXYcYd);

    // rhsXYcYd->copyToStarting(0,      dx);
    // rhsXYcYd->copyToStarting(nx,     dyc);
    // rhsXYcYd->copyToStarting(nx+nyc, dyd);

    // nlp->log->write("SOL KKT XYcYd dx: ", dx,  hovMatrices);
    // nlp->log->write("SOL KKT XYcYd dyc:", dyc, hovMatrices);
    // nlp->log->write("SOL KKT XYcYd dyd:", dyd, hovMatrices);
  
  }
} // end of namespace
