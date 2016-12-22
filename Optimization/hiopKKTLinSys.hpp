#ifndef HIOP_KKTLINSYSY
#define HIOP_KKTLINSYSY

#include "hiopIterate.hpp"
#include "hiopResidual.hpp"
#include "hiopHessianLowRank.hpp"

class hiopKKTLinSys 
{
public:
  /* updates the parts in KKT system that are dependent on the iterate. 
   * It may trigger a refactorization for direct linear systems, or it may not do 
   * anything, for example, LowRank linear system */
  virtual bool update(const hiopIterate* iter, 
		      const hiopVector* grad_f, 
		      const hiopMatrixDense* Jac_c, const hiopMatrixDense* Jac_d, 
		      hiopHessianLowRank* Hess)=0;
  virtual bool computeDirections(const hiopResidual* resid, hiopIterate* direction)=0;
  virtual ~hiopKKTLinSys() {}
};

class hiopKKTLinSysLowRank : public hiopKKTLinSys
{
public:
  hiopKKTLinSysLowRank(const hiopNlpFormulation* nlp_);
  virtual ~hiopKKTLinSysLowRank();

  virtual bool update(const hiopIterate* iter, 
		      const hiopVector* grad_f, 
		      const hiopMatrixDense* Jac_c, const hiopMatrixDense* Jac_d, 
		      hiopHessianLowRank* Hess);
  virtual bool computeDirections(const hiopResidual* resid, hiopIterate* direction);

  /* Solves the system corresponding to directions for x, yc, and yd, namely
   * [ H_BFGS + Dx   Jc^T  Jd^T   ] [ dx]   [ rx_tilde ]
   * [    Jc          0     0     ] [dyc] = [   ryc    ]
   * [    Jd          0   -Dd^{-1}] [dyd]   [ ryd_tilde]
   */
  virtual void solveCompressed(hiopVectorPar& rx, hiopVectorPar& ryc, hiopVectorPar& ryd,
			       hiopVectorPar& dx, hiopVectorPar& dyc, hiopVectorPar& dyd);

  static int factorizeMat(hiopMatrixDense& M);
  static int solveWithFactors(hiopMatrixDense& M, hiopVectorPar& r);
#ifdef DEEP_CHECKING
  static double solveError(const hiopMatrixDense& M,  const hiopVectorPar& x, hiopVectorPar& rhs);

  //computes the solve error for the KKT Linear system; used only for correctness checking
  double errorKKT(const hiopResidual* resid, const hiopIterate* sol);
  double errorCompressedLinsys(const hiopVectorPar& rx, const hiopVectorPar& ryc, const hiopVectorPar& ryd,
			       const hiopVectorPar& dx, const hiopVectorPar& dyc, const hiopVectorPar& dyd);
#endif
private:
  const hiopIterate* iter;
  const hiopVectorPar* grad_f;
  const hiopMatrixDense *Jac_c, *Jac_d;
  hiopHessianLowRank* Hess;

  const hiopNlpDenseConstraints* nlp;

  hiopMatrixDense* N; //the kxk reduced matrix
#ifdef DEEP_CHECKING
  hiopMatrixDense* Nmat; //a copy of the above to compute the residual
#endif
  hiopVectorPar *Dx, *Dd_inv;
  //internal buffers
  hiopVectorPar *rx_tilde, *ryd_tilde;
  hiopMatrixDense* _kxn_mat; //!opt (work directly with the Jacobian)
  hiopVectorPar* _k_vec1;
};

#endif
