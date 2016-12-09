#ifndef HIOP_HESSIANLOWRANK
#define HIOP_HESSIANLOWRANK

#include "hiopNlpFormulation.hpp"
#include "hiopIterate.hpp"

#include <cassert>



/* Abstract class for low-rank Hessian or inverse Hessian.
 * Stores the Hessian or inverse Hessian as D + M*N*M^T where D is nxn diag, 
 * M is nxk, and N is kxk. Here n>>k (k=O(10)).Usually k=2l. 
 * 
 * For parallel computations, D is a distributed vector, M is distributed 
 * column-wise, and N is local (stored on all processors).
 */
class hiopHessianLowRank
{
public:
  hiopHessianLowRank(const hiopNlpDenseConstraints* nlp_, int max_memory_length) 
    : l_max(max_memory_length), l_curr(0), sigma(1.), sigma0(1.), nlp(nlp_)  {}

  virtual ~hiopHessianLowRank() {};
  /* return false if the update destroys hereditary positive definitness and the BFGS update is not taken*/
  virtual bool update(const hiopIterate& x_curr, const hiopVector& grad_f_curr,
		      const hiopMatrix& Jac_c_curr, const hiopMatrix& Jac_d_curr) = 0;
  /*virtual bool update(const hiopIterate& x_prev, const hiopIterate& x_curr,
		      const hiopVector& grad_f_prev, const hiopVector& grad_f_curr,
		      const hiopMatrix& Jac_c_prev, const hiopMatrix& Jac_c_curr,
		      const hiopMatrix& Jac_d_prev, const hiopMatrix& Jac_d_curr) = 0;
  */
  /* recomputes the diagonal of the representation from B0+Dx form, where B0=sigma*I */
  virtual bool updateDiagonal(const hiopVector& Dx) = 0;
  /* Y = beta*Y + alpha*this*X 
   * For the Hessian***Inv***LowRank class this correspond to solving with the Hessian,
   * however, at the cost of a mat-vec.
   */
  virtual void apply(double beta, hiopVector& y, double alpha, const hiopVector& x) = 0;
  virtual void apply(double beta, hiopMatrix& Y, double alpha, const hiopMatrix& X) = 0;
#ifdef DEEP_CHECKING
  /* computes the product of the Hessian with a vector: y=beta*y+alpha*H*x.
   * The function is supposed to use the underlying ***recursive*** definition of the 
   * quasi-Newton Hessian and is used for checking/testing/error calculation.
   */
  virtual void timesVec(double beta, hiopVector& y, double alpha, const hiopVector&x) = 0;
#endif
protected:
  int l_max; //max memory size
  int l_curr; //current memory
  double sigma; //initial scaling factor of identity
  double sigma0; //initial/default scaling factor of identity
  const hiopNlpDenseConstraints* nlp;
private:
  hiopHessianLowRank() {};
  hiopHessianLowRank(const hiopHessianLowRank&) {};
  hiopHessianLowRank& operator=(const hiopHessianLowRank&) {};
};

/* Low-rank representation for the inverse of a low-rank matrix (Hessian).
 * In HIOP we need to solve with the BFGS quasi-Newton Hessian given by
 * the matrix M=B0+[B0*S Y] [S^T*B0*S  L] [ S^T*B0 ]
 *                          [ L^T     -D] [   Y   ]
 * Reference: Byrd, Nocedal, Schnabel, "Representations of quasi-Newton matrices and
 * and there use in limited memory methods", Math. Programming 63 (1994), p. 129-156.
 * 
 * To save on computations, we maintain a direct representaton of its inverse 
 * M^{-1} = H0 + [S HO*Y] [ R^{-T}*(D+Y^T*H0*Y)*R^{-1}    -R^{-T} ] [ S^T   ]
 *                        [          -R^{-1}                 0    ] [ Y^T*H0]
 * Here, H0=inv(B0). Check the above reference to see what each matrix represent. 
 */
class hiopHessianInvLowRank : public hiopHessianLowRank
{
public:
  hiopHessianInvLowRank(const hiopNlpDenseConstraints* nlp, int max_memory_length);
  virtual bool update(const hiopIterate& x_curr, const hiopVector& grad_f_curr,
		      const hiopMatrix& Jac_c_curr, const hiopMatrix& Jac_d_curr);

  virtual bool updateDiagonal(const hiopVector& Dx);

  /* ! these method uses quantities computed in symmetricTimesMat, thus they should be 
   * after symmetricTimesMat
   */
  virtual void apply(double beta, hiopVector& y, double alpha, const hiopVector& x);
  virtual void apply(double beta, hiopMatrix& Y, double alpha, const hiopMatrix& X);

  /* W = beta*W + alpha*X*this*X^T
   * ! make sure this is called before 'apply'
   */
  virtual void symmetricTimesMat(double beta, hiopMatrixDense& W,
				 double alpha, const hiopMatrixDense& X);

  virtual ~hiopHessianInvLowRank();
#ifdef DEEP_CHECKING
  /* computes the product of the Hessian with a vector: y=beta*y+alpha*H*x.
   * The function is supposed to use the underlying ***recursive*** definition of the 
   * quasi-Newton Hessian and is used for checking/testing/error calculation.
   */
  virtual void timesVec(double beta, hiopVector& y, double alpha, const hiopVector&x);
#endif
private: //internal methods
  
  /* symmetric multiplication W = beta*W + alpha*X*Diag*X^T */
  static void symmMatTimesDiagTimesMatTrans_local(double beta, hiopMatrixDense& W_,
					   double alpha, const hiopMatrixDense& X_,
					   const hiopVectorPar& d);
  /* W=S*Diag*X^T */
  static void matTimesDiagTimesMatTrans_local(hiopMatrixDense& W, const hiopMatrixDense& S, const hiopVectorPar& d, const hiopMatrixDense& X);

  /* rhs = R \ rhs, where R is upper triangular lxl and rhs is lx */
  static void triangularSolve(const hiopMatrixDense& R, hiopMatrixDense& rhs);
  static void triangularSolve(const hiopMatrixDense& R, hiopVectorPar& rhs);
  static void triangularSolveTrans(const hiopMatrixDense& R, hiopVectorPar& rhs);

  //grows R when the number of BFGS updates is less than the max memory
  void growR(const int& l_curr, const int& l_max, const hiopVectorPar& STy, const double& sTy);
  void growD(const int& l_curr, const int& l_max, const double& sTy);
  void updateR(const hiopVectorPar& STy, const double& sTy);
  void updateD(const double& sTy);
private:
  hiopVectorPar* H0;
  hiopMatrixDense *St,*Yt; //we store the transpose to easily access columns in S and T
  hiopMatrixDense *R;
  hiopVectorPar* D;

  int sigma_update_strategy;
  double sigma_safe_min, sigma_safe_max;

  //also stored are the iterate, gradient obj, and Jacobians at the previous iterations
  hiopIterate *_it_prev;
  hiopVectorPar *_grad_f_prev;
  hiopMatrixDense *_Jac_c_prev, *_Jac_d_prev;

  //internals buffers
  double* _buff_kxk; // size = num_constraints^2 
  double* _buff_lxk; // size = q-Newton mem size x num_constraints
  double* _buff_lxl;
  //auxiliary objects
  hiopMatrixDense *_S1, *_Y1, *_DpYtH0Y; //aux matrices to hold St*X, Yt*H0*X, and D+Y^T*H0*Y in symmetricTimesMat
  hiopMatrixDense& new_S1(const hiopMatrixDense& St, const hiopMatrixDense& X);
  hiopMatrixDense& new_Y1(const hiopMatrixDense& Yt, const hiopMatrixDense& X);
  hiopMatrixDense& new_DpYtH0Y(const hiopMatrixDense& Yt);
  //similar for S3=DpYtH0Y*S2
  hiopMatrixDense *_S3;
  hiopMatrixDense& new_S3(const hiopMatrixDense& Left, const hiopMatrixDense& Right);
  hiopVectorPar *_l_vec1, *_l_vec2, *_l_vec3, *_n_vec1, *_n_vec2;
  hiopVectorPar& new_l_vec1(int l);
  hiopVectorPar& new_l_vec2(int l);
  hiopVectorPar& new_l_vec3(int l);
  inline hiopVectorPar& new_n_vec1(long long n)
  {
#ifdef DEEP_CHECKING
    assert(_n_vec1!=NULL);
    assert(_n_vec1->get_size()==n);
#endif
    return *_n_vec1;
  }
  inline hiopVectorPar& new_n_vec2(long long n)
  {
#ifdef DEEP_CHECKING
    assert(_n_vec2!=NULL);
    assert(_n_vec2->get_size()==n);
#endif
    return *_n_vec2;
  }
};
#endif
