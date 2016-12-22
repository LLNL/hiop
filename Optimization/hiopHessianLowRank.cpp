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

hiopHessianLowRank::hiopHessianLowRank(const hiopNlpDenseConstraints* nlp_, int max_mem_len)
  : l_max(max_mem_len), l_curr(0), sigma(1.), sigma0(1.), nlp(nlp_), matrixChanged(false)
{
  DhInv = dynamic_cast<hiopVectorPar*>(nlp->alloc_primal_vec());
  St = nlp->alloc_multivector_primal(0,l_max);
  Yt = St->alloc_clone(); //faster than nlp->alloc_multivector_primal(...);
  //these are local
  L  = new hiopMatrixDense(0,0);
  D  = new hiopVectorPar(0);
  V  = new hiopMatrixDense(0,0);

  //the previous iteration's objects are set to NULL
  _it_prev=NULL; _grad_f_prev=NULL; _Jac_c_prev=NULL; _Jac_d_prev=NULL;

  //internal buffers for memory pool (none of them should be in n)
#ifdef WITH_MPI
  _buff_kxk    = new double[nlp->m() * nlp->m()];
  _buff_2lxk   = new double[nlp->m() * 2*l_max];
  _buff1_lxlx3 = new double[3*l_max*l_max];
  _buff2_lxlx3 = new double[3*l_max*l_max];
#else
   //not needed in non-MPI mode
  _buff_kxk  = NULL;
  _buff_2lxk = NULL;
  _buff1_lxlx3 = _buff2_lxlx3;
#endif

  //auxiliary objects/buffers
  _S1=_Y1=NULL;
  _lxl_mat1=_kxl_mat1=_kx2l_mat1=NULL;
  _l_vec1 = _l_vec2 = _2l_vec1 = NULL;
  _n_vec1 = DhInv->alloc_clone();
  _n_vec2 = DhInv->alloc_clone();

  _V_work_vec=new hiopVectorPar(0);
  _V_ipiv_vec=NULL; _V_ipiv_size=-1;

  sigma=sigma0;
  sigma_update_strategy = SIGMA_STRATEGY1;
  sigma_safe_min=1e-8;
  sigma_safe_max=1e+8;
  nlp->log->printf(hovScalars, "Hessian Low Rank: initial sigma is %g\n", sigma);
}  

hiopHessianLowRank::~hiopHessianLowRank()
{
  if(DhInv) delete DhInv;

  if(St) delete St;
  if(Yt) delete Yt;
  if(L)  delete L;
  if(D)  delete D;
  if(V)  delete V;

  if(_it_prev)    delete _it_prev;
  if(_grad_f_prev)delete _grad_f_prev;
  if(_Jac_c_prev) delete _Jac_c_prev;
  if(_Jac_d_prev) delete _Jac_d_prev;

  if(_buff_kxk)    delete[] _buff_kxk;
  if(_buff_2lxk)   delete[] _buff_2lxk;
  if(_buff1_lxlx3) delete[] _buff1_lxlx3;
  if(_buff2_lxlx3) delete[] _buff2_lxlx3;

  if(_S1) delete _S1;
  if(_Y1) delete _Y1;
  if(_lxl_mat1)    delete _lxl_mat1;
  if(_kxl_mat1)    delete _kxl_mat1; 
  if(_kx2l_mat1)   delete _kx2l_mat1;

  if(_l_vec1) delete _l_vec1;
  if(_l_vec2) delete _l_vec2;
  if(_n_vec1) delete _n_vec1;
  if(_n_vec2) delete _n_vec2;

  if(_V_ipiv_vec) delete[] _V_ipiv_vec;
  if(_V_work_vec) delete _V_work_vec;
}


bool hiopHessianLowRank::updateLogBarrierDiagonal(const hiopVector& Dx)
{
  DhInv->setToConstant(sigma);
  DhInv->axpy(1.0,Dx);
#ifdef DEEP_CHECKING
  assert(DhInv->allPositive());
#endif
  DhInv->invert();
  nlp->log->write("hiopHessianLowRank: inverse diag:", *DhInv, hovMatrices);
  matrixChanged=true;
}

#include <limits>

bool hiopHessianLowRank::update(const hiopIterate& it_curr, const hiopVector& grad_f_curr_,
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

  if(l_curr>0) {
    long long n=grad_f_curr.get_size();
    //compute s_new = x_curr-x_prev
    hiopVectorPar& s_new = new_n_vec1(n);  s_new.copyFrom(*it_curr.x); s_new.axpy(-1.,*_it_prev->x);
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
      //y_new.axzpy(-1.0, s_new, *it_curr.zl);
      //y_new.axzpy( 1.0, s_new, *it_curr.zu);
      
      double sTy = s_new.dotProductWith(y_new), s_nrm2=s_new.twonorm(), y_nrm2=y_new.twonorm();
      nlp->log->printf(hovLinAlgScalarsVerb, "hiopHessianInvLowRank_obsolette: s^T*y=%20.14e ||s||=%20.14e ||y||=%20.14e\n", sTy, s_nrm2, y_nrm2);
      nlp->log->write("hiopHessianInvLowRank_obsolette s_new",s_new, hovIteration);
      nlp->log->write("hiopHessianInvLowRank_obsolette y_new",y_new, hovIteration);

      if(sTy>s_nrm2*y_nrm2*std::numeric_limits<double>::epsilon()) { //sTy far away from zero
	//compute the new row in L, update S and Y (either augment them or shift cols and add s_new and y_new)
	hiopVectorPar& YTs = new_l_vec1(l_curr-1);
	Yt->timesVec(0.0, YTs, 1.0, s_new);
	//update representation
	if(l_curr<l_max) {
	  //just grow/augment the matrices
	  St->appendRow(s_new);
	  Yt->appendRow(y_new);
	  growL(l_curr, l_max, YTs);
	  growD(l_curr, l_max, sTy);
	  l_curr++;
	} else {
	  //shift
	  St->shiftRows(-1);
	  Yt->shiftRows(-1);
	  St->replaceRow(l_max-1, s_new);
	  Yt->replaceRow(l_max-1, y_new);
	  //updateL(YTs,sTy);
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
	sigma=fmax(fmin(sigma_safe_max, sigma), sigma_safe_min);
	nlp->log->printf(hovLinAlgScalarsVerb, "hiopHessianInvLowRank_obsolette: sigma was updated to %16.10e\n", sigma);
      } else { //sTy is too small or negative -> skip
	 nlp->log->printf(hovLinAlgScalarsVerb, "hiopHessianInvLowRank_obsolette: s^T*y=%12.6e not positive enough... skipping the Hessian update\n", sTy);
      }
    } else {// norm of s_new is too small -> skip
      nlp->log->printf(hovLinAlgScalarsVerb, "hiopHessianInvLowRank_obsolette: ||s_new||=%12.6e too small... skipping the Hessian update\n", s_infnorm);
    }

    //save this stuff for next update
    _it_prev->copyFrom(it_curr);  _grad_f_prev->copyFrom(grad_f_curr); 
    _Jac_c_prev->copyFrom(Jac_c_curr); _Jac_d_prev->copyFrom(Jac_d_curr);
    nlp->log->printf(hovLinAlgScalarsVerb, "hiopHessianInvLowRank_obsolette: storing the iteration info as 'previous'\n", s_infnorm);

  } else {
    //this is the first optimization iterate, just save the iterate and exit
    if(NULL==_it_prev)     _it_prev     = it_curr.new_copy();
    if(NULL==_grad_f_prev) _grad_f_prev = grad_f_curr.new_copy();
    if(NULL==_Jac_c_prev)  _Jac_c_prev  = Jac_c_curr.new_copy();
    if(NULL==_Jac_d_prev)  _Jac_d_prev  = Jac_d_curr.new_copy();

    nlp->log->printf(hovLinAlgScalarsVerb, "HessianInvLowRank on first update, just saving iteration\n");

    l_curr++;
  }
  return true;
}

/* 
 * The dirty work to bring this^{-1} to the form
 * M = DhInv + DhInv*[B0*S Y] * V^{-1} * [ S^T*B0 ] *DhInv
 *                                       [ Y^T    ]
 * Namely it computes V, a symmetric 2lx2l given by
 *  [S'*B0*(DhInv*B0-I)*S    -L+S'*B0*DhInv*Y ]
 *  [-L'+Y'*Dhinv*B0*S       +D+Y'*Dhinv*Y    ]
 * Caution: an upside-down symmetric permutation of the above form is stored 
 * for numerical stability considerations (the p.d. matrix D+Y'*Dhinv*Y is in (1,1)), namely
 * V = [   D+Y'*Dhinv*Y         -L'+Y'*Dhinv*B0*S  ]
 *     [ -L+S'*B0*DhInv*Y    S'*B0*(DhInv*B0-I)*S  ]
 * In this function V is factorize and it will contain the factors
 * Note that L, D, S, and Y are from the BFGS secant representation and are updated/computed in 'update'
 */
void hiopHessianLowRank::updateInternalBFGSRepresentation()
{
  long long n=St->n(), l=St->m();

  //grow L,D, andV if needed
  if(L->m()!=l) { delete L; L=new hiopMatrixDense(l,l);}
  if(D->get_size()!=l) { delete D; D=new hiopVectorPar(l); }
  if(V->m()!=2*l) {delete V; V=new hiopMatrixDense(2*l,2*l); }

  //-- block (1,1)
  hiopMatrixDense& DpYtDhInvY = new_lxl_mat1(l);
  symmMatTimesDiagTimesMatTrans_local(0.0, DpYtDhInvY, 1.0,*Yt,*DhInv);
#ifdef WITH_MPI
  const size_t buffsize=l*l*sizeof(double);
  memcpy(_buff1_lxlx3, DpYtDhInvY.local_buffer(), buffsize);
#else
  DpYtDhInvY.addDiagonal(*D);
  V->copyBlockFromMatrix(0,0,DpYtDhInvY);
#endif

  //-- block (2,1)
  hiopMatrixDense& StB0DhInvYmL = DpYtDhInvY; //just a rename
  hiopVectorPar& B0DhInv = new_n_vec1(n);
  B0DhInv.copyFrom(*DhInv); B0DhInv.scale(sigma);
  matTimesDiagTimesMatTrans_local(StB0DhInvYmL, *St, B0DhInv, *Yt);
#ifdef WITH_MPI
  memcpy(_buff1_lxlx3+l*l, StB0DhInvYmL.local_buffer(), buffsize);
#else
  //substract L
  StB0DhInvYmL.addMatrix(-1.0, *L);
  // (2,1) in V
  V->copyBlockFromMatrix(l,0,StB0DhInvYmL);
#endif

  //-- block (2,2)
  hiopVectorPar& theDiag = B0DhInv; //just a rename, also reuses values
  theDiag.addConstant(-1.0); //at this point theDiag=DhInv*B0-I
  theDiag.scale(sigma);
  hiopMatrixDense& StDS = DpYtDhInvY; //a rename
  symmMatTimesDiagTimesMatTrans_local(0.0, StDS, 1.0, *St, theDiag);
#ifdef WITH_MPI
  memcpy(_buff1_lxlx3+2*l*l, DpYtDhInvY.local_buffer(), buffsize);
#else
  V->copyBlockFromMatrix(l,l,StDS);
#endif


#ifdef WITH_MPI
  int ierr;
  ierr = MPI_Allreduce(_buff1_lxlx3, _buff2_lxlx3, 3*l*l, MPI_DOUBLE, MPI_SUM, nlp->get_comm()); assert(ierr==MPI_SUCCESS);

  // - block (1,1)
  DpYtDhInvY.copyFrom(_buff2_lxlx3);
  DpYtDhInvY.addDiagonal(*D);
  V->copyBlockFromMatrix(0,0,DpYtDhInvY);

  // - block (2,1)
  StB0DhInvYmL.copyFrom(_buff2_lxlx3+l*l);
  StB0DhInvYmL.addMatrix(-1.0, *L);
  V->copyBlockFromMatrix(l,0,StB0DhInvYmL);

  // - block (2,2)
  StDS.copyFrom(_buff2_lxlx3+2*l*l);
  V->copyBlockFromMatrix(l,l,StDS);
#endif

  //finally, factorize V
  factorizeV();

  matrixChanged=false;
}

/* Solves this*x = res as x = this^{-1}*res
 * where 'this^{-1}' is
 * M = DhInv + DhInv*[B0*S Y] * V^{-1} * [ S^T*B0 ] *DhInv
 *                                       [ Y^T    ]
 *
 * M is is nxn, S,Y are nxl, V is upper triangular 2lx2l, and x is nx1
 * Remember we store Yt=Y^T and St=S^T
 */  
void hiopHessianLowRank::solve(const hiopVector& rhs_, hiopVector& x_)
{
  if(matrixChanged) updateInternalBFGSRepresentation();

  hiopVectorPar& x = dynamic_cast<hiopVectorPar&>(x_);
  const hiopVectorPar& rhsx = dynamic_cast<const hiopVectorPar&>(rhs_);
  long long n=St->n(), l=St->m();
#ifdef DEEP_CHECKING
  assert(rhsx.get_size()==n);
  assert(x.get_size()==n);
  assert(DhInv->get_size()==n);
#endif

  //1. x = DhInv*res
  x.copyFrom(rhsx);
  x.componentMult(*DhInv);

  //2. stx= S^T*B0*DhInv*res and ytx=Y^T*DhInv*res
  hiopVectorPar &stx=new_l_vec1(l), &ytx=new_l_vec2(l);
  Yt->timesVec(0.0,ytx,1.0,x);

  hiopVectorPar& B0DhInvx = new_n_vec1(n);
  B0DhInvx.copyFrom(x); //it contains DhInv*res
  B0DhInvx.scale(sigma); //B0*(DhInv*res) 
  St->timesVec(0.0,stx,1.0,B0DhInvx);

  //3. solve with V
  hiopVectorPar& spart=stx; hiopVectorPar& ypart=ytx;
  solveWithV(spart,ypart);

  //4. multiply with  DhInv*[B0*S Y], namely
  // result = DhInv*(B0*S*spart + Y*ypart)
  hiopVectorPar&  result = new_n_vec1(n);
  St->transTimesVec(0.0, result, 1.0, spart);
  result.scale(sigma);
  Yt->transTimesVec(1.0, result, 1.0, ypart);
  result.componentMult(*DhInv);

  //5. x = first term + second term = x_computed_in_1 + result 
  x.axpy(1.0,result);
}

/* W = beta*W + alpha*X*inverse(this)*X^T (a more efficient version of solve)
 * where 'this^{-1}' is
 * M = DhInv + DhInv*[B0*S Y] * V^{-1} * [ S^T*B0 ] *DhInv
 *                                       [ Y^T    ]
 * W is kxk, S,Y are nxl, DhInv,B0 are n, V is 2lx2l
 * X is kxn
 */ 
void hiopHessianLowRank::
symMatTimesInverseTimesMatTrans(double beta, hiopMatrixDense& W, 
				double alpha, const hiopMatrixDense& X)
{
  if(matrixChanged) updateInternalBFGSRepresentation();

  long long n=St->n(), l=St->m();
  long long k=W.m(); 
  assert(X.m()==k);
  assert(X.n()==n);

  //1. compute W=beta*W + alpha*X*DhInv*X'
#ifdef WITH_MPI
  int myrank, ierr;
  ierr=MPI_Comm_rank(nlp->get_comm(),&myrank); assert(MPI_SUCCESS==ierr);
  if(0==myrank)
    symmMatTimesDiagTimesMatTrans_local(beta,W,alpha,X,*DhInv);
  else
    symmMatTimesDiagTimesMatTrans_local(0.0, W,alpha,X,*DhInv);
  //W will be MPI_All_reduced later
#else
  symmMatTimesDiagTimesMatTrans_local(beta,W,alpha,X,*DhInv);
#endif
  //2. compute S1=X*DhInv*B0*S and Y1=X*DhInv*Y
  hiopMatrixDense &S1=new_S1(X,*St), &Y1=new_Y1(X,*Yt); //both are kxl
  hiopVectorPar& B0DhInv = new_n_vec1(n);
  B0DhInv.copyFrom(*DhInv); B0DhInv.scale(sigma);
  matTimesDiagTimesMatTrans_local(S1, X, B0DhInv, *St);
  matTimesDiagTimesMatTrans_local(Y1, X, *DhInv,  *Yt);

  //3. reduce W, S1, and Y1 (dimensions: kxk, kxl, kxl)
  hiopMatrixDense& S2Y2 = new_kx2l_mat1(k,l);  //Initialy S2Y2 = [Y1 S1]
  //order of Y1 and S1 is changed to match the permutation of V
  S2Y2.copyBlockFromMatrix(0,0,Y1);
  S2Y2.copyBlockFromMatrix(0,l,S1);
#ifdef WITH_MPI
  ierr=MPI_Allreduce(S2Y2.local_buffer(), _buff_2lxk, 2*l*k, MPI_DOUBLE, MPI_SUM, nlp->get_comm()); assert(ierr==MPI_SUCCESS);
  ierr=MPI_Allreduce(W.local_buffer(),    _buff_kxk,  k*k,   MPI_DOUBLE, MPI_SUM, nlp->get_comm()); assert(ierr==MPI_SUCCESS);
  S2Y2.copyFrom(_buff_2lxk);
  W.copyFrom(_buff_kxk);
#endif

  //4. [S2] = V \ [S1^T]
  //   [Y2]       [Y1^T]
  //S2Y2 is exactly [S1^T] when Fortran Lapack looks at it
  //                [Y1^T]
  hiopMatrixDense& RHS_fortran = S2Y2; 
  solveWithV(RHS_fortran);

  //5. W = W+alpha*[S1 Y1]*[S2^T] 
  //                       [Y2^T]
  S2Y2 = RHS_fortran;
  hiopMatrixDense& S2=new_kxl_mat1(k,l);
  S2.copyFromMatrixBlock(S2Y2, 0, l);
  S1.timesMatTrans_local(1.0, W, alpha, S2);
  hiopMatrixDense& Y2=S2;
  Y2.copyFromMatrixBlock(S2Y2, 0,0);
  Y1.timesMatTrans_local(1.0, W, alpha, Y2);
  //we're done here
 
  /*  
  //2. compute S1= S^T*B0*DhInv*X^T and Y1= Y^T*DhInv*X^T
  hiopMatrixDense& S1 = new_S1(*St,X);
  hiopVectorPar& B0DhInv = new_n_vec1(n);
  B0DhInv.copyFrom(*DhInv); B0DhInv.scale(sigma);
  matTimesDiagTimesMatTrans_local(S1,*St, B0DhInv,X);
  hiopMatrixDense& Y1 = new_Y1(*Yt,X);
  matTimesDiagTimesMatTrans_local(Y1,*Yt,*DhInv,X);

  //3. reduce W, S1, and Y1 (dimensions: kxk, lxk, lxk)
  hiopMatrixDense& S2Y2 = new_2lxk_mat1(l,k);
  //order of Y1 and S1 is inverted to match the permutation of V
  S2Y2.copyRowsFrom(Y1, l, 0);
  S2Y2.copyRowsFrom(S1, l, l);
#ifdef WITH_MPI
  ierr=MPI_Allreduce(S2Y2.local_buffer(), _buff_2lxk, 2*l*k, MPI_DOUBLE, MPI_SUM, nlp->get_comm()); assert(ierr==MPI_SUCCESS);
  ierr=MPI_Allreduce(W.local_buffer(),    _buff_kxk,  k*k,   MPI_DOUBLE, MPI_SUM, nlp->get_comm()); assert(ierr==MPI_SUCCESS);
  S2Y2.copyFrom(_buff_2lxk);
  W.copyFrom(_buff_kxk);
#endif
  //4. [S2] = V\[S1]
  //   [Y2]     [Y1]
  solveWithV(S2Y2);

  //5. multiply S2Y2 at the left with S1' and Y1' (no communication, everything is local)
  //5.1 W = W+alpha*S1'*S2
  hiopMatrixDense& X2=new_lxk_mat1(l,k);
  X2.copyFromMatrixBlock(S2Y2, l, 0);
  S1.transTimesMat(1.0, W, alpha, X2);
  //4.2 W = W+alpha*Y1'*Y2
  hiopMatrixDense& Y2=X2;
  Y2.copyFromMatrixBlock(S2Y2, 0, 0);
  Y1.transTimesMat(1.0, W, alpha, Y2);
  */
}


void hiopHessianLowRank::factorizeV()
{
  int N=V->n(), lda=N, info;
  if(N==0) return;

#ifdef DEEP_CHECKING
  nlp->log->write("factorizeV:  V is ", *V, hovMatrices);
#endif

  char uplo='U'; //V is lower in C++ so it's upper in fortran

  if(_V_ipiv_vec==NULL) _V_ipiv_vec=new int[N];
  else if(_V_ipiv_size!=N) { delete[] _V_ipiv_vec; _V_ipiv_vec=new int[N]; _V_ipiv_size=N; }

  int lwork=-1;//inquire sizes
  double Vwork_tmp;
  dsytrf_(&uplo, &N, V->local_buffer(), &lda, _V_ipiv_vec, &Vwork_tmp, &lwork, &info);
  assert(info==0);

  lwork=(int)Vwork_tmp;
  if(lwork != _V_work_vec->get_size()) {
    if(_V_work_vec!=NULL) delete _V_work_vec;  
    _V_work_vec=new hiopVectorPar(lwork);
  } else assert(_V_work_vec);

  dsytrf_(&uplo, &N, V->local_buffer(), &lda, _V_ipiv_vec, _V_work_vec->local_data(), &lwork, &info);
  
  if(info<0)
    nlp->log->printf(hovError, "error: %d argument to dsytrf has an illegal value\n", -info);
  else if(info>0)
    nlp->log->printf(hovError, "error: %d entry in the factorization's diagonal is exactly zero. Division by zero will occur if it a solve is attempted.\n", info);
  assert(info==0);
#ifdef DEEP_CHECKING
  nlp->log->write("factorizeV:  factors of V: ", *V, hovMatrices);
#endif

}

void hiopHessianLowRank::solveWithV(hiopVectorPar& rhs_s, hiopVectorPar& rhs_y)
{
  int N=V->n();
  if(N==0) return;

#ifdef DEEP_CHECKING
  nlp->log->write("solveWithV: RHS IN 's' part: ", rhs_s, hovMatrices);
  nlp->log->write("solveWithV: RHS IN 'y' part: ", rhs_y, hovMatrices);
#endif

  int lda=N, one=1, info;
  int l=rhs_s.get_size();
  char uplo='U'; 
#ifdef DEEP_CHECKING
  assert(N==rhs_s.get_size()+rhs_y.get_size());
#endif
  hiopVectorPar& rhs=new_2l_vec1(l);
  //switch s and y to match the permutation of V
  rhs.copyFromStarting(rhs_y,0);
  rhs.copyFromStarting(rhs_s,l);

  dsytrs_(&uplo, &N, &one, V->local_buffer(), &lda, _V_ipiv_vec, rhs.local_data(), &N, &info);

  if(info<0) nlp->log->printf(hovError, "error: %d argument to dsytrf has an illegal value\n", -info);
  assert(info==0);

  //copy back the solution
  rhs.copyToStarting(rhs_y,0);
  rhs.copyToStarting(rhs_s,l);

#ifdef DEEP_CHECKING
  nlp->log->write("solveWithV: RHS OUT 's' part: ", rhs_s, hovMatrices);
  nlp->log->write("solveWithV: RHS OUT 'y' part: ", rhs_y, hovMatrices);
#endif

}

void hiopHessianLowRank::solveWithV(hiopMatrixDense& rhs)
{
  int N=V->n();
  if(0==N) return;

#ifdef DEEP_CHECKING
  nlp->log->write("solveWithV: RHS IN: ", rhs, hovMatrices);
#endif

  //rhs is transpose in C++

  char uplo='U'; 
  int lda=N, ldb=N, nrhs=rhs.m(), info;
#ifdef DEEP_CHECKING
  assert(N==rhs.n()); 
#endif
  dsytrs_(&uplo, &N, &nrhs, V->local_buffer(), &lda, _V_ipiv_vec, rhs.local_buffer(), &ldb, &info);

  if(info<0) nlp->log->printf(hovError, "error: %d argument to dsytrf has an illegal value\n", -info);
  assert(info==0);
#ifdef DEEP_CHECKING
  nlp->log->write("solveWithV: RHS OUT: ", rhs, hovMatrices);
#endif

}

void hiopHessianLowRank::growL(const int& lmem_curr, const int& lmem_max, const hiopVectorPar& YTs)
{
  int l=L->m();
#ifdef DEEP_CHECKING
  assert(l==L->n());
  assert(lmem_curr-1==l);
  assert(lmem_max>l);
#endif
  //newL = [   L     0]
  //       [ Y^T*s   0]
  hiopMatrixDense* newL = new hiopMatrixDense(l+1,l+1);
  assert(newL);
  //copy from L to newL
  newL->copyBlockFromMatrix(0,0, *L);

  double** newL_mat=newL->local_data(); //doing the rest here
  const double* YTs_vec=YTs.local_data_const();
  for(int j=0; j<l; j++) newL_mat[l][j] = YTs_vec[j];

  //and the zero entries of the last column
  for(int i=0; i<l+1; i++) newL_mat[i][l] = 0.0;

  //swap the pointers
  delete L;
  L=newL;
}

void hiopHessianLowRank::growD(const int& lmem_curr, const int& lmem_max, const double& sTy)
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

void hiopHessianLowRank::updateL(const hiopVectorPar& YTs, const double& sTy)
{
  int l=YTs.get_size();
#ifdef DEEP_CHECKING
  assert(l==L->m());
  assert(l==L->n());
#endif
  const int lm1=l-1;
  double** L_mat=L->local_data();
  const double* yts_vec=YTs.local_data_const();
  for(int i=1; i<lm1; i++)
    for(int j=0; j<i; j++)
      L_mat[i][j] = L_mat[i+1][j+1];

  //is this really needed?
  for(int i=0; i<lm1; i++)
    L_mat[i][lm1]=0.0;

  for(int j=0; j<lm1; j++)
    L_mat[lm1][j]=yts_vec[j];

  L_mat[lm1][lm1]=0.0;
}
void hiopHessianLowRank::updateD(const double& sTy)
{
  int l=D->get_size();
  double* D_vec = D->local_data();
  for(int i=0; i<l-1; i++)
    D_vec[i]=D_vec[i+1];
  D_vec[l-1]=sTy;
}


hiopVectorPar&  hiopHessianLowRank::new_l_vec1(int l)
{
  if(_l_vec1!=NULL && _l_vec1->get_size()==l) return *_l_vec1;
  
  if(_l_vec1!=NULL) {
    delete _l_vec1;
  }
  _l_vec1= new hiopVectorPar(l);
  return *_l_vec1;
}
hiopVectorPar&  hiopHessianLowRank::new_l_vec2(int l)
{
  if(_l_vec2!=NULL && _l_vec2->get_size()==l) return *_l_vec2;
  
  if(_l_vec2!=NULL) {
    delete _l_vec2;
  }
  _l_vec2= new hiopVectorPar(l);
  return *_l_vec2;
}

hiopMatrixDense& hiopHessianLowRank::new_lxl_mat1(int l)
{
  if(_lxl_mat1!=NULL)
    if( l==_lxl_mat1->m() ) {
      return *_lxl_mat1;
    } else {
      delete _lxl_mat1; 
      _lxl_mat1=NULL;
    }
  _lxl_mat1 = new hiopMatrixDense(l,l);
  return *_lxl_mat1;
}
hiopMatrixDense& hiopHessianLowRank::new_kx2l_mat1(int k, int l)
{
  int twol=2*l;
  if(NULL!=_kx2l_mat1) {
    assert(_kx2l_mat1->m()==k);
    if( twol==_kx2l_mat1->n() ) {
      return *_kx2l_mat1;
    } else {
      delete _kx2l_mat1; 
      _kx2l_mat1=NULL;
    }
  }
  _kx2l_mat1 = new hiopMatrixDense(k,twol);
  
  return *_kx2l_mat1;
}
hiopMatrixDense& hiopHessianLowRank::new_kxl_mat1(int k, int l)
{
  if(_kxl_mat1!=NULL) {
    assert(_kxl_mat1->m()==k);
    if( l==_kxl_mat1->n() ) {
      return *_kxl_mat1;
    } else {
      delete _kxl_mat1; 
      _kxl_mat1=NULL;
    }
  }
  _kxl_mat1 = new hiopMatrixDense(k,l);
  return *_kxl_mat1;
}
hiopMatrixDense& hiopHessianLowRank::new_S1(const hiopMatrixDense& X, const hiopMatrixDense& St)
{
  //S1 is X*some_diag*S  (kxl). Here St=S^T is lxn and X is kxn (l BFGS memory size, k number of constraints)
  long long k=X.m(), n=St.n(), l=St.m();
#ifdef DEEP_CHECKING
  assert(n==X.n());
  if(_S1!=NULL) 
    assert(_S1->m()==k);
#endif
  if(NULL!=_S1 && _S1->n()!=l) { delete _S1; _S1=NULL; }
  
  if(NULL==_S1) _S1=new hiopMatrixDense(k,l);

  return *_S1;
}

hiopMatrixDense& hiopHessianLowRank::new_Y1(const hiopMatrixDense& X, const hiopMatrixDense& Yt)
{
  //Y1 is X*somediag*Y (kxl). Here Yt=Y^T is lxn,  X is kxn
  long long k=X.m(), n=Yt.n(), l=Yt.m();
#ifdef DEEP_CHECKING
  assert(X.n()==n);
  if(_Y1!=NULL) assert(_Y1->m()==k);
#endif

  if(NULL!=_Y1 && _Y1->n()!=l) { delete _Y1; _Y1=NULL; }

  if(NULL==_Y1) _Y1=new hiopMatrixDense(k,l);
  return *_Y1;
}

#ifdef DEEP_CHECKING
#include <vector>
using namespace std;
void hiopHessianLowRank::timesVecCmn(double beta, hiopVector& y, double alpha, const hiopVector& x, bool addLogTerm) 
{
  long long n=St->n();
  assert(l_curr-1==St->m());
  assert(y.get_size()==n);
  //we have B+=B-B*s*B*s'/(s'*B*s)+yy'/(y'*s)
  //B0 is sigma*I (and is NOT this->H0, since this->H0=(B0+Dx)^{-1})

  bool print=true;
  if(print) {
    nlp->log->printf(hovMatrices, "---hiopHessianLowRank::timesVec \n");
    nlp->log->write("S':", *St, hovMatrices);
    nlp->log->write("Y':", *Yt, hovMatrices);
    nlp->log->write("DhInv:", *DhInv, hovMatrices);
    nlp->log->printf(hovMatrices, "sigma=%22.16e  addLogTerm=%d\n", sigma, addLogTerm);
    nlp->log->printf(hovMatrices, "y=beta*y + alpha*this*x : beta=%g alpha=%g\n", beta, alpha);
    nlp->log->write("x_in:", x, hovMatrices);
    nlp->log->write("y_in:", y, hovMatrices);
  }

  hiopVectorPar *yk=dynamic_cast<hiopVectorPar*>(nlp->alloc_primal_vec());
  hiopVectorPar *sk=dynamic_cast<hiopVectorPar*>(nlp->alloc_primal_vec());
  //allocate and compute a_k and b_k
  vector<hiopVectorPar*> a(l_curr),b(l_curr);
  for(int k=0; k<l_curr-1; k++) {
    //bk=yk/sqrt(yk'*sk)
    yk->copyFrom(Yt->local_data()[k]);
    sk->copyFrom(St->local_data()[k]);
    double skTyk=yk->dotProductWith(*sk);
    assert(skTyk>0);
    b[k]=dynamic_cast<hiopVectorPar*>(nlp->alloc_primal_vec());
    b[k]->copyFrom(*yk);
    b[k]->scale(1/sqrt(skTyk));

    a[k]=dynamic_cast<hiopVectorPar*>(nlp->alloc_primal_vec());

    //compute ak by an inner loop
    a[k]->copyFrom(*sk);
    if(addLogTerm)
      a[k]->componentDiv(*DhInv);
    else
      a[k]->scale(sigma);
    for(int i=0; i<k; i++) {
      double biTsk = b[i]->dotProductWith(*sk);
      a[k]->axpy(biTsk, *b[i]);
      double aiTsk = a[i]->dotProductWith(*sk);
      a[k]->axpy(aiTsk, *a[i]);
    }
    double skTak = a[k]->dotProductWith(*sk);
    a[k]->scale(1/sqrt(skTak));
  }

  //new we have B= Dx+B_0 + sum{ bk bk' - ak ak' : k=0,1,...,l_curr-1} (H0=(Dx+B0)^{-1})
  //compute the product with x
  //y = beta*y+alpha*H0_inv*x + alpha* sum { bk'x bk - ak'x ak : k=0,1,...,l_curr-1}
  y.scale(beta);
  if(addLogTerm) 
    y.axdzpy(alpha,x,*DhInv);
  else
    y.axpy(alpha*sigma, x); 
  for(int k=0; k<l_curr-1; k++) {
    double bkTx = b[k]->dotProductWith(x);
    double akTx = a[k]->dotProductWith(x);
    
    y.axpy( alpha*bkTx, *b[k]);
    y.axpy(-alpha*akTx, *a[k]);
  }

  if(print) {
    nlp->log->write("y_out:", y, hovMatrices);
  }

  for(vector<hiopVectorPar*>::iterator it=a.begin(); it!=a.end(); ++it) 
    delete *it;
  for(vector<hiopVectorPar*>::iterator it=b.begin(); it!=b.end(); ++it) 
    delete *it;

  delete yk;
  delete sk;
}

void hiopHessianLowRank::timesVec(double beta, hiopVector& y, double alpha, const hiopVector&x)
{
  this->timesVecCmn(beta, y, alpha, x, true);
}

void hiopHessianLowRank::timesVec_noLogBarrierTerm(double beta, hiopVector& y, double alpha, const hiopVector&x)
{
  this->timesVecCmn(beta, y, alpha, x, false);
}
#endif

/**************************************************************************
 * Internal helpers
 *************************************************************************/

/* symmetric multiplication W = beta*W + alpha*X*Diag*X^T 
 * W is kxk local, X is kxn distributed and Diag is n, distributed
 * The ops are perform locally. The reduce is done separately/externally to decrease comm
 */
void hiopHessianLowRank::
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
void hiopHessianLowRank::
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
/**************************************************************************
 * this code is going to be removed
 *************************************************************************/
hiopHessianInvLowRank_obsolette::hiopHessianInvLowRank_obsolette(const hiopNlpDenseConstraints* nlp_, int max_mem_len)
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
  _l_vec1 = _l_vec2 = NULL;
  _n_vec1 = H0->alloc_clone();
  _n_vec2 = H0->alloc_clone();
  //H0->setToConstant(sigma);

  sigma=sigma0;
  sigma_update_strategy = SIGMA_STRATEGY1;
  sigma_safe_min=1e-8;
  sigma_safe_max=1e+8;
  nlp->log->printf(hovScalars, "Hessian Low Rank: initial sigma is %g\n", sigma);
}
hiopHessianInvLowRank_obsolette::~hiopHessianInvLowRank_obsolette()
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
  if(_n_vec1) delete _n_vec1;
  if(_n_vec2) delete _n_vec2;
}

#include <limits>

bool hiopHessianInvLowRank_obsolette::
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

  if(l_curr>0) {
    long long n=grad_f_curr.get_size();
    //compute s_new = x_curr-x_prev
    hiopVectorPar& s_new = new_n_vec1(n);  s_new.copyFrom(*it_curr.x); s_new.axpy(-1.,*_it_prev->x);
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
      //y_new.axzpy(-1.0, s_new, *it_curr.zl);
      //y_new.axzpy( 1.0, s_new, *it_curr.zu);
      
      double sTy = s_new.dotProductWith(y_new), s_nrm2=s_new.twonorm(), y_nrm2=y_new.twonorm();
      nlp->log->printf(hovLinAlgScalarsVerb, "hiopHessianInvLowRank_obsolette: s^T*y=%20.14e ||s||=%20.14e ||y||=%20.14e\n", sTy, s_nrm2, y_nrm2);
      nlp->log->write("hiopHessianInvLowRank_obsolette s_new",s_new, hovIteration);
      nlp->log->write("hiopHessianInvLowRank_obsolette y_new",y_new, hovIteration);

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
	sigma=fmax(fmin(sigma_safe_max, sigma), sigma_safe_min);
	nlp->log->printf(hovLinAlgScalarsVerb, "hiopHessianInvLowRank_obsolette: sigma was updated to %16.10e\n", sigma);
      } else { //sTy is too small or negative -> skip
	 nlp->log->printf(hovLinAlgScalarsVerb, "hiopHessianInvLowRank_obsolette: s^T*y=%12.6e not positive enough... skipping the Hessian update\n", sTy);
      }
    } else {// norm of s_new is too small -> skip
      nlp->log->printf(hovLinAlgScalarsVerb, "hiopHessianInvLowRank_obsolette: ||s_new||=%12.6e too small... skipping the Hessian update\n", s_infnorm);
    }

    //save this stuff for next update
    _it_prev->copyFrom(it_curr);  _grad_f_prev->copyFrom(grad_f_curr); 
    _Jac_c_prev->copyFrom(Jac_c_curr); _Jac_d_prev->copyFrom(Jac_d_curr);
    nlp->log->printf(hovLinAlgScalarsVerb, "hiopHessianInvLowRank_obsolette: storing the iteration info as 'previous'\n", s_infnorm);

  } else {
    //this is the first optimization iterate, just save the iterate and exit
    if(NULL==_it_prev)     _it_prev     = it_curr.new_copy();
    if(NULL==_grad_f_prev) _grad_f_prev = grad_f_curr.new_copy();
    if(NULL==_Jac_c_prev)  _Jac_c_prev  = Jac_c_curr.new_copy();
    if(NULL==_Jac_d_prev)  _Jac_d_prev  = Jac_d_curr.new_copy();

    nlp->log->printf(hovLinAlgScalarsVerb, "HessianInvLowRank on first update, just saving iteration\n");

    l_curr++;
  }
  return true;
}

bool hiopHessianInvLowRank_obsolette::updateLogBarrierDiagonal(const hiopVector& Dx)
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
void hiopHessianInvLowRank_obsolette::apply(double beta, hiopVector& y_, double alpha, const hiopVector& x_)
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
void hiopHessianInvLowRank_obsolette::apply(double beta, hiopMatrix& Y, double alpha, const hiopMatrix& X)
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
void hiopHessianInvLowRank_obsolette::
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
  W.print();
  assert(W.assertSymmetry(1e-14));
#endif

}

/* symmetric multiplication W = beta*W + alpha*X*Diag*X^T 
 * W is kxk local, X is kxn distributed and Diag is n, distributed
 * The ops are perform locally. The reduce is done separately/externally to decrease comm
 */
void hiopHessianInvLowRank_obsolette::
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
void hiopHessianInvLowRank_obsolette::
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
void hiopHessianInvLowRank_obsolette::triangularSolve(const hiopMatrixDense& R, hiopMatrixDense& rhs)
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

void hiopHessianInvLowRank_obsolette::triangularSolve(const hiopMatrixDense& R, hiopVectorPar& rhs)
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
void hiopHessianInvLowRank_obsolette::triangularSolveTrans(const hiopMatrixDense& R, hiopVectorPar& rhs)
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
void hiopHessianInvLowRank_obsolette::growR(const int& lmem_curr, const int& lmem_max, const hiopVectorPar& STy, const double& sTy)
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

void hiopHessianInvLowRank_obsolette::growD(const int& lmem_curr, const int& lmem_max, const double& sTy)
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

void hiopHessianInvLowRank_obsolette::updateR(const hiopVectorPar& STy, const double& sTy)
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
void hiopHessianInvLowRank_obsolette::updateD(const double& sTy)
{
  int l=D->get_size();
  double* D_vec = D->local_data();
  for(int i=0; i<l-1; i++)
    D_vec[i]=D_vec[i+1];
  D_vec[l-1]=sTy;
}


hiopMatrixDense& hiopHessianInvLowRank_obsolette::new_S1(const hiopMatrixDense& St, const hiopMatrixDense& X)
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

hiopMatrixDense& hiopHessianInvLowRank_obsolette::new_Y1(const hiopMatrixDense& Yt, const hiopMatrixDense& X)
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
hiopMatrixDense& hiopHessianInvLowRank_obsolette::new_DpYtH0Y(const hiopMatrixDense& Yt)
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
hiopMatrixDense& hiopHessianInvLowRank_obsolette::new_S3(const hiopMatrixDense& Left, const hiopMatrixDense& Right)
{
  int l=Left.m(), k=Right.n();
#ifdef DEEP_CHECKING
  assert(Right.m()==l);
  assert(Left.n()==l);
  if(_S3!=NULL) assert(_S3->m()<=l); //< when the representation grows, = when it doesn't
#endif
  if(_S3!=NULL && _S3->m()!=l) { delete _S3; _S3=NULL;}

  if(_S3==NULL) _S3 = new hiopMatrixDense(l,k);
  return *_S3;
}
hiopVectorPar&  hiopHessianInvLowRank_obsolette::new_l_vec1(int l)
{
  if(_l_vec1!=NULL && _l_vec1->get_size()==l) return *_l_vec1;
  
  if(_l_vec1!=NULL) {
    delete _l_vec1;
  }
  _l_vec1= new hiopVectorPar(l);
  return *_l_vec1;
}
hiopVectorPar&  hiopHessianInvLowRank_obsolette::new_l_vec2(int l)
{
  if(_l_vec2!=NULL && _l_vec2->get_size()==l) return *_l_vec2;
  
  if(_l_vec2!=NULL) {
    delete _l_vec2;
  }
  _l_vec2= new hiopVectorPar(l);
  return *_l_vec2;
}
hiopVectorPar&  hiopHessianInvLowRank_obsolette::new_l_vec3(int l)
{
  if(_l_vec3!=NULL && _l_vec3->get_size()==l) return *_l_vec3;
  
  if(_l_vec3!=NULL) {
    delete _l_vec3;
  }
  _l_vec3= new hiopVectorPar(l);
  return *_l_vec3;
}

#ifdef DEEP_CHECKING
// #include <vector>
// using namespace std;
// void hiopHessianInvLowRank_obsolette::timesVecCmn(double beta, hiopVector& y, double alpha, const hiopVector& x, bool addLogTerm) 
// {
//   long long n=St->n();
//   assert(l_curr-1==St->m());
//   assert(y.get_size()==n);
//   //we have B+=B-B*s*B*s'/(s'*B*s)+yy'/(y'*s)
//   //B0 is sigma*I (and is NOT this->H0, since this->H0=(B0+Dx)^{-1})

//   bool print=true;
//   if(print) {
//     nlp->log->printf(hovMatrices, "---hiopHessianInvLowRank_obsolette::timesVec \n");
//     nlp->log->write("S':", *St, hovMatrices);
//     nlp->log->write("Y':", *Yt, hovMatrices);
//     nlp->log->write("H0:", *H0, hovMatrices);
//     nlp->log->printf(hovMatrices, "sigma=%22.16e  addLogTerm=%d\n", sigma, addLogTerm);
//     nlp->log->printf(hovMatrices, "y=beta*y + alpha*this*x : beta=%g alpha=%g\n", beta, alpha);
//     nlp->log->write("x_in:", x, hovMatrices);
//     nlp->log->write("y_in:", y, hovMatrices);
//   }

//   hiopVectorPar *yk=dynamic_cast<hiopVectorPar*>(nlp->alloc_primal_vec());
//   hiopVectorPar *sk=dynamic_cast<hiopVectorPar*>(nlp->alloc_primal_vec());
//   //allocate and compute a_k and b_k
//   vector<hiopVectorPar*> a(l_curr),b(l_curr);
//   for(int k=0; k<l_curr-1; k++) {
//     //bk=yk/sqrt(yk'*sk)
//     yk->copyFrom(Yt->local_data()[k]);
//     sk->copyFrom(St->local_data()[k]);
//     double skTyk=yk->dotProductWith(*sk);
//     assert(skTyk>0);
//     b[k]=dynamic_cast<hiopVectorPar*>(nlp->alloc_primal_vec());
//     b[k]->copyFrom(*yk);
//     b[k]->scale(1/sqrt(skTyk));

//     a[k]=dynamic_cast<hiopVectorPar*>(nlp->alloc_primal_vec());

//     //compute ak by an inner loop
//     a[k]->copyFrom(*sk);
//     if(addLogTerm)
//       a[k]->componentDiv(*H0);
//     else
//       a[k]->scale(sigma);
//     for(int i=0; i<k; i++) {
//       double biTsk = b[i]->dotProductWith(*sk);
//       a[k]->axpy(biTsk, *b[i]);
//       double aiTsk = a[i]->dotProductWith(*sk);
//       a[k]->axpy(aiTsk, *a[i]);
//     }
//     double skTak = a[k]->dotProductWith(*sk);
//     a[k]->scale(1/sqrt(skTak));
//   }

//   //new we have B= Dx+B_0 + sum{ bk bk' - ak ak' : k=0,1,...,l_curr-1} (H0=(Dx+B0)^{-1})
//   //compute the product with x
//   //y = beta*y+alpha*H0_inv*x + alpha* sum { bk'x bk - ak'x ak : k=0,1,...,l_curr-1}
//   y.scale(beta);
//   if(addLogTerm) 
//     y.axdzpy(alpha,x,*H0);
//   else
//     y.axpy(alpha*sigma, x); 
//   for(int k=0; k<l_curr-1; k++) {
//     double bkTx = b[k]->dotProductWith(x);
//     double akTx = a[k]->dotProductWith(x);
    
//     y.axpy( alpha*bkTx, *b[k]);
//     y.axpy(-alpha*akTx, *a[k]);
//   }

//   if(print) {
//     nlp->log->write("y_out:", y, hovMatrices);
//   }

//   for(vector<hiopVectorPar*>::iterator it=a.begin(); it!=a.end(); ++it) 
//     delete *it;
//   for(vector<hiopVectorPar*>::iterator it=b.begin(); it!=b.end(); ++it) 
//     delete *it;

//   delete yk;
//   delete sk;
// }

// void hiopHessianInvLowRank_obsolette::timesVec(double beta, hiopVector& y, double alpha, const hiopVector&x)
// {
//   this->timesVecCmn(beta, y, alpha, x, true);
// }

// void hiopHessianInvLowRank_obsolette::timesVec_noLogBarrierTerm(double beta, hiopVector& y, double alpha, const hiopVector&x)
// {
//   this->timesVecCmn(beta, y, alpha, x, false);
// }
#endif

