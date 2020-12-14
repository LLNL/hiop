#ifndef HIOP_MAGMASOLVER
#define HIOP_MAGMASOLVER

#include "hiopNlpFormulation.hpp"
#include "hiopLinSolver.hpp"


#ifdef HIOP_USE_MAGMA
#include "magma_v2.h"

/** Wrapper classes around MAGMA solver for symmetric indefinite matrices
 *
 */

//these FLOPS counter are from MAGMA
#define FADDS_POTRF(n_) ((n_) * (((1. / 6.) * (n_)      ) * (n_) - (1. / 6.)))
#define FADDS_POTRI(n_) ( (n_) * ((1. / 6.) + (n_) * ((1. / 3.) * (n_) - 0.5)) )
#define FADDS_POTRS(n_, nrhs_) ((nrhs_) * (n_) * ((n_) - 1 ))
#define FMULS_POTRF(n_) ((n_) * (((1. / 6.) * (n_) + 0.5) * (n_) + (1. / 3.)))
#define FMULS_POTRI(n_) ( (n_) * ((2. / 3.) + (n_) * ((1. / 3.) * (n_) + 1. )) )
#define FMULS_POTRS(n_, nrhs_) ((nrhs_) * (n_) * ((n_) + 1 ))
#define FLOPS_DPOTRF(n_) (     FMULS_POTRF((double)(n_)) +       FADDS_POTRF((double)(n_)) )
#define FLOPS_DPOTRI(n_) (     FMULS_POTRI((double)(n_)) +       FADDS_POTRI((double)(n_)) )
#define FLOPS_DPOTRS(n_, nrhs_) (     FMULS_POTRS((double)(n_), (double)(nrhs_)) +       FADDS_POTRS((double)(n_), (double)(nrhs_)) )


namespace hiop {

/** 
 * Solver based on Magma's CPU interface for GPU implementation of 'dsytrf' using 
 * Bunch-Kaufmann. This is a numerically stable factorization with decent peak 
 * performance on large matrices.
 *
 */
class hiopLinSolverIndefDenseMagmaBuKa : public hiopLinSolverIndefDense
{
public:
  hiopLinSolverIndefDenseMagmaBuKa(int n, hiopNlpFormulation* nlp_);
  virtual ~hiopLinSolverIndefDenseMagmaBuKa();

  /** Triggers a refactorization of the matrix, if necessary. */
  int matrixChanged();

  /** Solves a linear system.
   * param 'x' is on entry the right hand side(s) of the system to be solved. On
   * exit is contains the solution(s).  
   */
  bool solve(hiopVector& x_in);

  inline hiopMatrixDense& sysMatrix() 
  { 
    return *M_; 
  }
protected:

  /**
   * Computes inertia of matrix, namely the triplet of non-negative numbers 
   * consisting of the counts of positive, negative, and null eigenvalues
   *
   * @pre The system matrix is factorized and, as a result, `ipiv` has been 
   * also updated properly.
   */
  virtual bool compute_inertia(int n_in, int *ipiv, 
			       int& posEigvals, int& negEigvals, int& zeroEigvals); 
protected:
  int* ipiv_;

  //allocated on demand; for example, it may only be required by Magma SIDI.
  double* work_;
  //magma_queue_t magma_device_queue;
  //magmaDouble_ptr device_M, device_rhs_;
private:
  hiopLinSolverIndefDenseMagmaBuKa() { assert(false); }
};

class hiopLinSolverIndefDenseMagmaBuKaDev : public hiopLinSolverIndefDenseMagmaBuKa
{
public:
  hiopLinSolverIndefDenseMagmaBuKaDev(int n, hiopNlpFormulation* nlp_)
    : hiopLinSolverIndefDenseMagmaBuKa(n, nlp_)
  {

  }
  virtual ~hiopLinSolverIndefDenseMagmaBuKaDev()
  {
  }
  
protected:
  /**
   * Computes inertia of matrix, namely the triplet of non-negative numbers 
   * consisting of the counts of positive, negative, and null eigenvalues. This 
   * method runs on host.
   *
   * @pre The system matrix is factorized and, as a result, `ipiv` has been 
   * also correctly set 
   */
  virtual bool compute_inertia(int n, int *ipiv_in, 
			       int& posEigvals, int& negEigvals, int& zeroEigvals); 
};


/**
 * Solver class for MAGMA symmetric indefinite GPU factorization "_nopiv". This
 * is as much as twice as faster but numerically less stable than the above
 * factorization.
 *
 */

class hiopLinSolverIndefDenseMagmaNopiv : public hiopLinSolverIndefDense
{
public:
  hiopLinSolverIndefDenseMagmaNopiv(int n, hiopNlpFormulation* nlp);

  virtual ~hiopLinSolverIndefDenseMagmaNopiv();

  /** Triggers a refactorization of the matrix, if necessary. */
  int matrixChanged();

  /** solves a linear system.
   * param 'x' is on entry the right hand side(s) of the system to be solved. On
   * exit it contains the solution(s).  
   */
  bool solve(hiopVector& x_in);

  inline hiopMatrixDense& sysMatrix() 
  { 
    return *M_; 
  }

  void inline set_fake_inertia(int nNegEigs)
  {
    nFakeNegEigs_ = nNegEigs;
  }
protected:
  /**
   * Computes inertia of matrix, namely the triplet of non-negative numbers 
   * of positive, negative, and null eigenvalues. This method runs on device and
   * accesses the device pointer(s). All the parameters reside on device.
   *
   * @pre The system matrix is factorized on the device.
   * also correctly set 
   */
  bool compute_inertia(int n, int& posEigvals, int& negEigvals, int& zeroEigvals); 

protected:
  magma_queue_t magma_device_queue_;
  magmaDouble_ptr device_M_, device_rhs_;
  magma_int_t ldda_, lddb_;
  int nFakeNegEigs_;
private:
  hiopLinSolverIndefDenseMagmaNopiv() 
  {
    assert(false); 
  }
};


} //end namespace hiop

#endif //of HIOP_USE_MAGMA
#endif
