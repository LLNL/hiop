#include "hiopHessianLowRank.hpp"

#include "blasdefs.hpp"

#ifdef WITH_MPI
#include "mpi.h"
#endif

#include <cassert>
#include <cstring>
#include <cmath>

#define SIGMA_STRATEGY1 1
#define SIGMA_STRATEGY2 2
#define SIGMA_STRATEGY3 3
#define SIGMA_STRATEGY4 4
#define SIGMA_CONSTANT  5
hiopHessianInvLowRank::hiopHessianInvLowRank(const hiopNlpDenseConstraints* nlp_, int max_mem_len)
  : hiopHessianLowRank(nlp_,max_mem_len)
{
  const hiopNlpDenseConstraints* nlp = dynamic_cast<const hiopNlpDenseConstraints*>(nlp_);
  //  assert(nlp==NULL && "only NLPs with a small number of constraints are supported by HessianLowRank");

  H0 = dynamic_cast<hiopVectorPar*>(nlp->alloc_primal_vec());
  St = nlp->alloc_multivector_primal(0,l_max);
  Yt = St->alloc_clone(); //faster than nlp->alloc_multivector_primal(...);
  //these are local
  R  = new hiopMatrixDense(0, 0);
  D  = new hiopVectorPar(0);

  //the previous iteration's objects are set to NULL
  _it_prev=NULL; _grad_f_prev=NULL; _Jac_c_prev=NULL; _Jac_d_prev=NULL;


  //internal buffers for memory pool (none of them should be in n)
#ifdef WITH_MPI
  _buff_kxk = new double[nlp->m() * nlp->m()];
  _buff_lxk = new double[nlp->m() * l_max];
  _buff_lxl = new double[l_max*l_max];
#else
   //not needed in non-MPI mode
  _buff_kxk = NULL;
  _buff_lxk = NULL;
  _buff_lxl = NULL;
#endif

  //auxiliary objects/buffers
  _S1=_Y1=_S3=NULL;
  _DpYtH0Y=NULL;
  _l_vec1 = _l_vec2 = _l_vec3 = NULL;
  _n_vec1 = H0->alloc_clone();
  _n_vec2 = H0->alloc_clone();
  //H0->setToConstant(sigma);

  sigma=sigma0;
  sigma_update_strategy = SIGMA_STRATEGY1;
  sigma_safe_min=1e-8;
  sigma_safe_max=1e+8;
  nlp->log->printf(hovScalars, "Hessian Low Rank: initial sigma is %g\n", sigma);
}
hiopHessianInvLowRank::~hiopHessianInvLowRank()
{
  if(H0) delete H0;
  if(St) delete St;
  if(Yt) delete Yt;
  if(R)  delete R;
  if(D)  delete D;

  if(_it_prev)    delete _it_prev;
  if(_grad_f_prev)delete _grad_f_prev;
  if(_Jac_c_prev) delete _Jac_c_prev;
  if(_Jac_d_prev) delete _Jac_d_prev;

  if(_buff_kxk) delete[] _buff_kxk;
  if(_buff_lxk) delete[] _buff_lxk;
  if(_buff_lxl) delete[] _buff_lxl;
  if(_S1) delete _S1;
  if(_Y1) delete _Y1;
  if(_DpYtH0Y) delete _DpYtH0Y;
  if(_S3) delete _S3;
  if(_l_vec1) delete _l_vec1;
  if(_l_vec2) delete _l_vec2;
  if(_l_vec3) delete _l_vec3;
  if(_n_vec1) delete _n_vec1;
  if(_n_vec2) delete _n_vec2;
}

#include <limits>

bool hiopHessianInvLowRank::
update(const hiopIterate& it_curr, const hiopVector& grad_f_curr_,
       const hiopMatrix& Jac_c_curr_, const hiopMatrix& Jac_d_curr_)
{
  const hiopVectorPar&   grad_f_curr= dynamic_cast<const hiopVectorPar&>(grad_f_curr_);
  const hiopMatrixDense& Jac_c_curr = dynamic_cast<const hiopMatrixDense&>(Jac_c_curr_);
  const hiopMatrixDense& Jac_d_curr = dynamic_cast<const hiopMatrixDense&>(Jac_d_curr_);

#ifdef DEEP_CHECKING
  assert(it_curr.zl->matchesPattern(nlp->get_ixl()));
  assert(it_curr.zu->matchesPattern(nlp->get_ixu()));
  assert(it_curr.sxl->matchesPattern(nlp->get_ixl()));
  assert(it_curr.sxu->matchesPattern(nlp->get_ixu()));
#endif
  return true;//assert(false);

  if(l_curr>0) {
    long long n=grad_f_curr.get_size();
    //compute s_new = x_curr-x_prev
    hiopVectorPar& s_new = new_n_vec1(n);  s_new.copyFrom(*_it_prev->x); s_new.axpy(-1.,*it_curr.x);
    double s_infnorm=s_new.infnorm();
    if(s_infnorm>=100*std::numeric_limits<double>::epsilon()) { //norm of s not too small

      //compute y_new = \grad J(x_curr,\lambda_curr) - \grad J(x_prev, \lambda_curr) (yes, J(x_prev, \lambda_curr))
      //              = graf_f_curr-grad_f_prev + (Jac_c_curr-Jac_c_prev)yc_curr+ (Jac_d_curr-Jac_c_prev)yd_curr - zl_curr*s_new + zu_curr*s_new
      hiopVectorPar& y_new = new_n_vec2(n);
      y_new.copyFrom(grad_f_curr); 
      y_new.axpy(-1., *_grad_f_prev);
      Jac_c_curr.transTimesVec  (1.0, y_new, 1.0, *it_curr.yc);
      _Jac_c_prev->transTimesVec(1.0, y_new,-1.0, *it_curr.yc); //!opt if nlp->Jac_c_isLinear no need for the multiplications
      Jac_d_curr.transTimesVec  (1.0, y_new, 1.0, *it_curr.yd); //!opt same here
      _Jac_d_prev->transTimesVec(1.0, y_new,-1.0, *it_curr.yd);
      y_new.axzpy(-1.0, s_new, *it_curr.zl);
      y_new.axzpy( 1.0, s_new, *it_curr.zu);
      
      double sTy = s_new.dotProductWith(y_new), s_nrm2=s_new.twonorm(), y_nrm2=y_new.twonorm();
      nlp->log->printf(hovLinAlgScalarsVerb, "hiopHessianInvLowRank: s^T*y=%20.14e ||s||=%20.14e ||y||=%20.14e\n", sTy, s_nrm2, y_nrm2);

      if(sTy>s_nrm2*y_nrm2*std::numeric_limits<double>::epsilon()) { //sTy far away from zero
	//compute the new column in R, update S and Y (either augment them or shift cols and add s_new and y_new)
	hiopVectorPar& STy = new_l_vec1(l_curr-1);
	St->timesVec(0.0, STy, 1.0, y_new);
	//update representation
	if(l_curr<l_max) {
	  //just grow/augment the matrices
	  St->appendRow(s_new);
	  Yt->appendRow(y_new);
	  growR(l_curr, l_max, STy, sTy);
	  growD(l_curr, l_max, sTy);
	  l_curr++;
	} else {
	  //shift
	  St->shiftRows(-1);
	  Yt->shiftRows(-1);
	  St->replaceRow(l_max-1, s_new);
	  Yt->replaceRow(l_max-1, y_new);
	  updateR(STy,sTy);
	  updateD(sTy);
	  l_curr=l_max;
	}

	//update B0 (i.e., sigma)
	switch (sigma_update_strategy ) {
	case SIGMA_STRATEGY1:
	  sigma=sTy/(s_nrm2*s_nrm2);
	  break;
	case SIGMA_STRATEGY2:
	  sigma=y_nrm2*y_nrm2/sTy;
	  break;
	case SIGMA_STRATEGY3:
	  sigma=sqrt(s_nrm2*s_nrm2 / y_nrm2 / y_nrm2);
	  break;
	case SIGMA_STRATEGY4:
	  sigma=0.5*(sTy/(s_nrm2*s_nrm2)+y_nrm2*y_nrm2/sTy);
	  break;
	case SIGMA_CONSTANT:
	  sigma=sigma0;
	  break;
	default:
	  assert(false && "Option value for sigma_update_strategy was not recognized.");
	  break;
	} // else of the switch
	//safe guard it
	sigma=fmax(fmin(sigma_safe_min, sigma), sigma_safe_max);
	nlp->log->printf(hovLinAlgScalarsVerb, "hiopHessianInvLowRank: sigma was updated to %16.10e\n", sigma);
      } else { //sTy is too small or negative -> skip
	 nlp->log->printf(hovLinAlgScalarsVerb, "hiopHessianInvLowRank: s^T*y=%12.6e not positive enough... skipping the Hessian update\n", sTy);
      }
    } else {// norm of s_new is too small -> skip
      nlp->log->printf(hovLinAlgScalarsVerb, "hiopHessianInvLowRank: ||s_new||=%12.6e too small... skipping the Hessian update\n", s_infnorm);
    }

    //save this stuff for next update
    _it_prev->copyFrom(it_curr);  _grad_f_prev->copyFrom(grad_f_curr); 
    _Jac_c_prev->copyFrom(Jac_c_curr); _Jac_d_prev->copyFrom(Jac_d_curr);
    nlp->log->printf(hovLinAlgScalarsVerb, "hiopHessianInvLowRank: storing the iteration info as 'previous'\n", s_infnorm);

  } else {
    //this is the first optimization iterate, just save the iterate and exit
    if(NULL==_it_prev)     _it_prev     = it_curr.new_copy();
    if(NULL==_grad_f_prev) _grad_f_prev = grad_f_curr.new_copy();
    if(NULL==_Jac_c_prev)  _Jac_c_prev  = Jac_c_curr.new_copy();
    if(NULL==_Jac_d_prev)  _Jac_d_prev  = Jac_c_curr.new_copy();

    nlp->log->printf(hovLinAlgScalarsVerb, "HessianInvLowRank on first update, just saving iteration");

    l_curr++;
  }
  return true;
}

bool hiopHessianInvLowRank::updateDiagonal(const hiopVector& Dx)
{
  H0->setToConstant(sigma);
  H0->axpy(1.0,Dx);
#ifdef DEEP_CHECKING
  assert(H0->allPositive());
#endif
  H0->invert();
  nlp->log->write("H0 InvHes", *H0, hovMatrices);
  
}


/* Y = beta*Y + alpha*this*X
 * where 'this' is
 * M^{-1} = H0 + [S HO*Y] [ R^{-T}*(D+Y^T*H0*Y)*R^{-1}    -R^{-T} ] [ S^T   ]
 *                        [          -R^{-1}                 0    ] [ Y^T*H0]
 *
 * M is is nxn, S,Y are nxl, R is upper triangular lxl, and X is nx1
 * Remember we store Yt=Y^T and St=S^T
 */  
void hiopHessianInvLowRank::apply(double beta, hiopVector& y_, double alpha, const hiopVector& x_)
{
  hiopVectorPar& y = dynamic_cast<hiopVectorPar&>(y_);
  const hiopVectorPar& x = dynamic_cast<const hiopVectorPar&>(x_);
  long long n=St->n(), l=St->m();
#ifdef DEEP_CHECKING
  assert(y.get_size()==n);
  assert(x.get_size()==n);
  assert(H0->get_size()==n);
#endif
  //0. y = beta*y + alpha*H0*y
  y.scale(beta);
  y.axzpy(alpha,*H0,x);

  //1. stx = S^T*x and ytx = Y^T*H0*x
  hiopVectorPar &stx=new_l_vec1(l), &ytx=new_l_vec2(l), &H0x=new_n_vec1(n);
  St->timesVec(0.0,stx,1.0,x);
  H0x.copyFrom(x); H0x.componentMult(*H0);
  Yt->timesVec(0.0,ytx,1.0,H0x);
  //2.ytx = R^{-T}* [(D+Y^T*H0*Y)*R^{-1} stx - ytx ]
  //  stx = -R^{-1}*stx                                  
  //2.1. stx = R^{-1}*stx 
  triangularSolve(*R, stx);
  //2.2 ytx = -ytx + (D+Y^T*H0*Y)*R^{-1} stx
  hiopMatrixDense& DpYtH0Y = *_DpYtH0Y; // ! this is computed in symmetricTimesMat
  DpYtH0Y.timesVec(-1.0,ytx, 1.0, stx);
  //2.3 ytx = R^{-T}* [(D+Y^T*H0*Y)*R^{-1} stx - ytx ]  and  stx = -R^{-1}*stx         
  triangularSolveTrans(*R,ytx);

  //3. add alpha(S*ytx + H0*Y*stx) to y
  St->transTimesVec(1.0,y,alpha,ytx);
  hiopVectorPar& H0Ystx=new_n_vec1(n);
  //H0Ystx=H0*Y*stx
  Yt->transTimesVec(0.0, H0Ystx, -1.0, stx); //-1.0 since we haven't negated stx in 2.3
  H0Ystx.componentMult(*H0);
  y.axpy(alpha,H0Ystx);
}
void hiopHessianInvLowRank::apply(double beta, hiopMatrix& Y, double alpha, const hiopMatrix& X)
{
  assert(false && "not yet implemented");
}


/* W = beta*W + alpha*X*this*X^T 
 * where 'this' is
 * M^{-1} = H0 + [S HO*Y] [ R^{-T}*(D+Y^T*H0*Y)*R^{-1}    -R^{-T} ] [ S^T   ]
 *                        [          -R^{-1}                 0    ] [ Y^T*H0]
 *
 * W is kxk, H0 is nxn, S,Y are nxl, R is upper triangular lxl, and X is kxn
 * Remember we store Yt=Y^T and St=S^T
 */
void hiopHessianInvLowRank::
symmetricTimesMat(double beta, hiopMatrixDense& W,
		  double alpha, const hiopMatrixDense& X)
{
  long long n=St->n(), l=St->m(), k=W.m();
  assert(n==H0->get_size());
  assert(k==W.n());
  assert(l==Yt->m());
  assert(n==Yt->n()); assert(n==St->n());
  assert(k==X.m()); assert(n==X.n());

  //1.--compute W = beta*W + alpha*X*HO*X^T by calling symmMatTransTimesDiagTimesMat_local
#ifdef WITH_MPI
  int myrank, ierr;
  ierr=MPI_Comm_rank(nlp->get_comm(),&myrank); assert(MPI_SUCCESS==ierr);
  if(0==myrank)
    symmMatTimesDiagTimesMatTrans_local(beta,W,alpha,X,*H0);
  else 
    symmMatTimesDiagTimesMatTrans_local(0.0,W,alpha,X,*H0);
  //W will be MPI_All_reduced later
#else
  symmMatTimesDiagTimesMatTrans_local(beta,W,alpha,X,*H0);
#endif

  //2.--compute S1=S^T*X^T=St*X^T and Y1=Y^T*H0*X^T=Yt*H0*X^T
  hiopMatrixDense& S1 = new_S1(*St,X);
  St->timesMatTrans_local(0.0,S1,1.0,X);
  hiopMatrixDense& Y1 = new_Y1(*Yt,X);
  matTimesDiagTimesMatTrans_local(Y1,*Yt,*H0,X);

  //3.--compute Y^T*H0*Y from D+Y^T*H0*Y
  hiopMatrixDense& DpYtH0Y = new_DpYtH0Y(*Yt);
  symmMatTimesDiagTimesMatTrans_local(0.0,DpYtH0Y, 1.0,*Yt,*H0);
#ifdef WITH_MPI
  //!opt - use one buffer and one reduce call
  ierr=MPI_Allreduce(S1.local_buffer(),      _buff_lxk,l*k, MPI_DOUBLE,MPI_SUM,nlp->get_comm()); assert(ierr==MPI_SUCCESS);
  S1.copyFrom(_buff_lxk);
  ierr=MPI_Allreduce(Y1.local_buffer(),      _buff_lxk,l*k, MPI_DOUBLE,MPI_SUM,nlp->get_comm()); assert(ierr==MPI_SUCCESS);
  Y1.copyFrom(_buff_lxk);
  ierr=MPI_Allreduce(W.local_buffer(),       _buff_kxk,k*k, MPI_DOUBLE,MPI_SUM,nlp->get_comm()); assert(ierr==MPI_SUCCESS);
  W.copyFrom(_buff_kxk);

  ierr=MPI_Allreduce(DpYtH0Y.local_buffer(), _buff_lxl,l*l, MPI_DOUBLE,MPI_SUM,nlp->get_comm()); assert(ierr==MPI_SUCCESS);
  DpYtH0Y.copyFrom(_buff_lxl);
#endif
 //add D to finish calculating D+Y^T*H0*Y
  DpYtH0Y.addDiagonal(*D);

  //now we have W = beta*W+alpha*X*HO*X^T and the remaining term in the form
  // [S1^T Y1^T] [ R^{-T}*DpYtH0Y*R^{-1}  -R^{-T} ] [ S1 ]
  //             [       -R^{-1}            0     ] [ Y1 ]
  // So that W is updated with 
  //    W += (S1^T*R^{-T})*DpYtH0Y*(R^{-1}*S1) - (S1^T*R^{-T})*Y1 - Y1^T*(R^{-1}*S1)
  // or W +=       S2^T   *DpYtH0Y*   S2       -      S2^T    *Y1 -   Y1^T*S2 
  
  //4.-- compute S1 = R\S1
  triangularSolve(*R,S1);

  //5.-- W += S2^T   *DpYtH0Y*   S2
  //S3=DpYtH0Y*S2
  hiopMatrix& S3=new_S3(DpYtH0Y, S1);
  DpYtH0Y.timesMat(0.0,S3,1.0,S1);
  // W += S2^T * S3
  S1.transTimesMat(1.0,W,1.0,S3);

  //6.-- W += -  S2^T    *Y1 -   (S2^T  *Y1)^T  
  // W = W - S2^T*Y1
  S1.transTimesMat(1.0,W,-1.0,Y1);
  // W = W - Y1*S2^T
  Y1.transTimesMat(1.0,W,-1.0,S1);

  // -- done --
#ifdef DEEP_CHECKING
  //W.print();
  assert(W.assertSymmetry());
#endif

}

/* symmetric multiplication W = beta*W + alpha*X*Diag*X^T 
 * W is kxk local, X is kxn distributed and Diag is n, distributed
 * The ops are perform locally. The reduce is done separately/externally to decrease comm
 */
void hiopHessianInvLowRank::
symmMatTimesDiagTimesMatTrans_local(double beta, hiopMatrixDense& W,
				    double alpha, const hiopMatrixDense& X,
				    const hiopVectorPar& d)
{
  long long k=W.m();
  long long n=X.n();
  size_t n_local=X.get_local_size_n();
#ifdef DEEP_CHECKING
  assert(W.n()==k);
  assert(X.m()==k);
  assert(d.get_size()==n);
  assert(d.get_local_size()==n_local);
#endif
  //#define chunk 512; //!opt
  double *xi, *xj, acc;
  double **Wdata=W.local_data(), **Xdata=X.local_data();
  const double* dd=d.local_data_const();
  for(int i=0; i<k; i++) {
    xi=Xdata[i];
    for(size_t j=i; j<k; j++) {
      xj=Xdata[j];
      //compute W[i,j] = sum {X[i,p]*d[p]*X[j,p] : p=1,...,n_local}
      acc=0.0;
      for(size_t p=0; p<n_local; p++)
	acc += xi[p]*dd[p]*xj[p];

      Wdata[i][j]=Wdata[j][i]=beta*Wdata[i][j]+alpha*acc;
    }
  }
}

/* W=S*D*X^T, where S is lxn, D is diag nxn, and X is kxn */
void hiopHessianInvLowRank::
matTimesDiagTimesMatTrans_local(hiopMatrixDense& W, const hiopMatrixDense& S, const hiopVectorPar& d, const hiopMatrixDense& X)
{
#ifdef DEEP_CHECKING
  assert(S.n()==d.get_size());
  assert(S.n()==X.n());
#endif  
  int l=S.m(), n=d.get_local_size(), k=X.m();
  double *Sdi, *Wdi, *Xdj, acc;
  double **Wd=W.local_data(), **Sd=S.local_data(), **Xd=X.local_data(); const double *diag=d.local_data_const();
  //!opt
  for(int i=0;i<l; i++) {
    Sdi=Sd[i]; Wdi=Wd[i];
    for(int j=0; j<k; j++) {
      Xdj=Xd[j];
      acc=0.;
      for(int p=0; p<n; p++) 
	acc += Sdi[p]*diag[p]*Xdj[p];
      //for(int p=0; p<n;p++) {acc += *Si * *diag * *Xd; Si++; diag++; Xd+=n;}
      
      Wdi[j]=acc;
    }
  }
}

/* rhs = R \ rhs, where R is upper triangular lxl and rhs is lxk. R is supposed to be local */
void hiopHessianInvLowRank::triangularSolve(const hiopMatrixDense& R, hiopMatrixDense& rhs)
{
  int l=R.m(), k=rhs.n();
#ifdef DEEP_CHECKING
  assert(R.n()==l);
#endif
  assert(l==rhs.m());
  if(0==l) return; //nothing to solve

  const double *Rbuf = R.local_buffer(); double *rhsbuf=rhs.local_buffer();
  char side  ='l'; //op(A)X=rhs
  char uplo  ='l'; //since we store it row-wise and fortran access it column-wise
  char transA='T'; //to solve with an upper triangular, we force fortran to solve a transpose lower (see above)
  char diag  ='N'; //not a unit triangular
  double one=1.0; int lda=l, ldb=l;
  dtrsm_(&side,&uplo,&transA,&diag,
	 &l, //rows of rhs
	 &k, //columns of rhs
	 &one,
	 Rbuf,&lda,
	 rhsbuf, &ldb);
  
}

void hiopHessianInvLowRank::triangularSolve(const hiopMatrixDense& R, hiopVectorPar& rhs)
{
  int l=R.m();
#ifdef DEEP_CHECKING
  assert(l==R.n());
  assert(rhs.get_size()==l);
#endif
  if(0==l) return; //nothing to solve
  const double* Rbuf=R.local_buffer(); double *rhsbuf=rhs.local_data();

  //we need to solve AX=B but we ask Fortran-style LAPACK to solve A^T X = B
  char side  ='l'; //op(A)X=rhs
  char uplo  ='l'; //we store R row-wise, but fortran access it column-wise, thus we ask LAPACK to access the lower triangular
  char transA='T'; //to solve with an upper triangular, we ask fortran to solve a transpose lower (see above)
  char diag  ='N'; //not a unit triangular
  double one=1.0; int lda=l, ldb=l, k=1;
  dtrsm_(&side,&uplo,&transA,&diag,
	 &l, //rows of rhs
	 &k, //columns of rhs
	 &one,
	 Rbuf,&lda,
	 rhsbuf, &ldb);
}
void hiopHessianInvLowRank::triangularSolveTrans(const hiopMatrixDense& R, hiopVectorPar& rhs)
{
  int l=R.m();
#ifdef DEEP_CHECKING
  assert(l==R.n());
  assert(rhs.get_size()==l);
#endif
  if(0==l) return; //nothing to solve
  const double* Rbuf=R.local_buffer(); double *rhsbuf=rhs.local_data();

  //we need to solve A^TX=B but we ask Fortran-style LAPACK to solve A X = B
  char side  ='l'; //op(A)X=rhs
  char uplo  ='l'; //we store upper triangular R row-wise, but fortran access it column-wise, thus we ask LAPACK to access the lower triangular
  char transA='N'; //to transpose-solve with an upper triangular, we ask fortran to perform a simple lower triangular solve (see above)
  char diag  ='N'; //not a unit triangular
  double one=1.0; int lda=l, ldb=l, k=1;
  dtrsm_(&side,&uplo,&transA,&diag,
	 &l, //rows of rhs
	 &k, //columns of rhs
	 &one,
	 Rbuf,&lda,
	 rhsbuf, &ldb);
}
void hiopHessianInvLowRank::growR(const int& lmem_curr, const int& lmem_max, const hiopVectorPar& STy, const double& sTy)
{
  int l=R->m();
#ifdef DEEP_CHECKING
  assert(l==R->n());
  assert(lmem_curr-1==l);
  assert(lmem_max>l);
#endif
  //newR = [ R S^T*y ]
  //       [ 0 s^T*y ]
  hiopMatrixDense* newR = new hiopMatrixDense(l+1,l+1);
  assert(newR);
  //copy from R to newR
  newR->copyBlockFromMatrix(0,0, *R);

  double** newR_mat=newR->local_data(); //doing the rest here
  const double* STy_vec=STy.local_data_const();
  for(int j=0; j<l; j++) newR_mat[j][l] = STy_vec[j];
  newR_mat[l][l] = sTy;

  //and the zero entries on the last row
  for(int i=0; i<l; i++) newR_mat[i][l] = 0.0;

  //swap the pointers
  delete R;
  R=newR;
}

void hiopHessianInvLowRank::growD(const int& lmem_curr, const int& lmem_max, const double& sTy)
{
  int l=D->get_size();
  assert(l==lmem_curr-1);
  assert(lmem_max>l);

  hiopVectorPar* Dnew=new hiopVectorPar(l+1);
  double* Dnew_vec=Dnew->local_data();
  memcpy(Dnew_vec, D->local_data_const(), l);
  Dnew_vec[l]=sTy;

  delete D;
  D=Dnew;
}

void hiopHessianInvLowRank::updateR(const hiopVectorPar& STy, const double& sTy)
{
  int l=STy.get_size();
#ifdef DEEP_CHECKING
  assert(l==R->m());
  assert(l==R->n());
#endif
  const int lm1=l-1;
  double** R_mat=R->local_data();
  const double* sTy_vec=STy.local_data_const();
  for(int i=0; i<lm1; i++)
    for(int j=i; j<lm1; j++)
      R_mat[i][j] = R_mat[i+1][j+1];
  for(int j=0; j<lm1; j++)
    R_mat[lm1][j]=0.0;
  for(int i=0; i<lm1; i++)
    R_mat[i][lm1]=sTy_vec[i];

  R_mat[lm1][lm1]=sTy;
}
void hiopHessianInvLowRank::updateD(const double& sTy)
{
  int l=D->get_size();
  double* D_vec = D->local_data();
  for(int i=0; i<l-1; i++)
    D_vec[i]=D_vec[i+1];
  D_vec[l-1]=sTy;
}

hiopMatrixDense& hiopHessianInvLowRank::new_S1(const hiopMatrixDense& St, const hiopMatrixDense& X)
{
  //S1 is St*X^T (lxk), where St=S^T is lxn and X is kxn (l BFGS memory size, k number of constraints)
  long long k=X.m(), n=St.n(), l=St.m();
#ifdef DEEP_CHECKING
  assert(n==X.n());
  if(_S1!=NULL) 
    assert(_S1->n()==k);
#endif
  if(NULL!=_S1 && _S1->n()!=l) { delete _S1; _S1=NULL; }
  
  if(NULL==_S1) _S1=new hiopMatrixDense(l,k);

  return *_S1;
}

hiopMatrixDense& hiopHessianInvLowRank::new_Y1(const hiopMatrixDense& Yt, const hiopMatrixDense& X)
{
  //Y1 is Yt*H0*X^T = Y^T*H0*X^T, where Y^T is lxn, H0 is diag nxn, X is kxn
  long long k=X.m(), n=Yt.n(), l=Yt.m();
#ifdef DEEP_CHECKING
  assert(X.n()==n);
  if(_Y1!=NULL) assert(_Y1->n()==k);
#endif

  if(NULL!=_Y1 && _Y1->n()!=l) { delete _Y1; _Y1=NULL; }

  if(NULL==_Y1) _Y1=new hiopMatrixDense(l,k);
  return *_Y1;
}
hiopMatrixDense& hiopHessianInvLowRank::new_DpYtH0Y(const hiopMatrixDense& Yt)
{
  long long l = Yt.m();
#ifdef DEEP_CHECKING
  if(_DpYtH0Y!=NULL) assert(_DpYtH0Y->m()==_DpYtH0Y->n());
#endif
  if(_DpYtH0Y!=NULL && _DpYtH0Y->m()!=l) {delete _DpYtH0Y; _DpYtH0Y=NULL;}
  if(_DpYtH0Y==NULL) _DpYtH0Y=new hiopMatrixDense(l,l);
  return *_DpYtH0Y;
}

/* S3 = DpYtH0H * S2, where S2=R\S1. DpYtH0H is symmetric (!opt) lxl and S2 is lxk */
hiopMatrixDense& hiopHessianInvLowRank::new_S3(const hiopMatrixDense& Left, const hiopMatrixDense& Right)
{
  int l=Left.m(), k=Right.n();
#ifdef DEEP_CHECKING
  assert(Right.m()==l);
  assert(Left.n()==l);
  if(_S3!=NULL) assert(_S3->m()==l);
#endif
  if(_S3!=NULL && _S3->m()!=l) { delete _S3; _S3=NULL;}

  if(_S3==NULL) _S3 = new hiopMatrixDense(l,k);
  return *_S3;
}
hiopVectorPar&  hiopHessianInvLowRank::new_l_vec1(int l)
{
  if(_l_vec1!=NULL && _l_vec1->get_size()==l) return *_l_vec1;
  
  if(_l_vec1!=NULL) {
    delete _l_vec1;
  }
  _l_vec1= new hiopVectorPar(l);
  return *_l_vec1;
}
hiopVectorPar&  hiopHessianInvLowRank::new_l_vec2(int l)
{
  if(_l_vec2!=NULL && _l_vec2->get_size()==l) return *_l_vec2;
  
  if(_l_vec2!=NULL) {
    delete _l_vec2;
  }
  _l_vec2= new hiopVectorPar(l);
  return *_l_vec2;
}
hiopVectorPar&  hiopHessianInvLowRank::new_l_vec3(int l)
{
  if(_l_vec3!=NULL && _l_vec3->get_size()==l) return *_l_vec3;
  
  if(_l_vec3!=NULL) {
    delete _l_vec3;
  }
  _l_vec3= new hiopVectorPar(l);
  return *_l_vec3;
}

#ifdef DEEP_CHECKING
void hiopHessianInvLowRank::timesVec(double beta, hiopVector& y, double alpha, const hiopVector&x) 
{
  //we have B+=B-B*s*B*s'/(s'*B*s)+yy'/(y'*s)
  //B0 is sigma*I (and is NOT this->H0, since this->H0=(B0+Dx)^{-1})
  //hiopVectorPar& aux = new_n_vec1(x.get_size());
  
  //y = beta*y+alpha*sigma*x
  y.scale(beta);
  y.axpy(alpha*sigma,x);
}
#endif

