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
 * Solver based on Magma's GPU interface for 'dsytrf' using Bunch-Kaufmann. This is 
 * a numerically stable factorization with decent peak performance on large matrices.
 * 
 * @note: The option "compute_mode" decides whether this class is instantiated (values
 * "gpu" and "hybrid"); otherwise, the CPU, LAPACK-based counterpart linear solver 
 * class is used. However, this class works with all the values for "mem_space" option. 
 * Regardless of the value of the "mem_space" option (and when instructed by the 
 * "compute_mode" option mentioned above) this class will always offload to the device 
 * the factorization and the solve with the triangular factors. 
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
  virtual bool compute_inertia(int n, int *ipiv_in, int& posEig, int& negEig, int& zeroEig);
protected:
  int* ipiv_;
  /// array storing the inertia (on the device pointer)
  int* dinert_; 
  magma_queue_t magma_device_queue_;
  magmaDouble_ptr device_M_, device_rhs_;
  magma_int_t ldda_, lddb_;
private:
  hiopLinSolverIndefDenseMagmaBuKa() { assert(false); }
};


/**
 * Solver class for MAGMA symmetric indefinite GPU factorization "_nopiv". This
 * is as much as twice faster but numerically less stable than the above
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

  /** 
   * Solves a linear system with the right-hand side `x`. This is also an out
   * parameter and on exit it contains the solution.
   */
  bool solve(hiopVector& x);

  inline hiopMatrixDense& sysMatrix() 
  { 
    return *M_; 
  }
protected:
  /**
   * Computes inertia of matrix, namely the triplet of non-negative numbers 
   * of positive, negative, and null eigenvalues. This method runs on device and
   * accesses the device pointer(s). All the parameters reside on device.
   *
   * @pre The system matrix is factorized and is present on the device.
   * 
   */
  bool compute_inertia(int n, int& posEigvals, int& negEigvals, int& zeroEigvals); 

protected:
  magma_queue_t magma_device_queue_;
  magmaDouble_ptr device_M_, device_rhs_;
  magma_int_t ldda_, lddb_;
private:
  hiopLinSolverIndefDenseMagmaNopiv() 
  {
    assert(false); 
  }
};

#if 0

/** 
 * Solver based on Magma's CPU interface for GPU implementation of 'dsytrf' using 
 * Bunch-Kaufmann. This is a numerically stable factorization with decent peak 
 * performance on large matrices.
 *
 * Superceeded by BuKa solver - code is disabled
 */
class hiopLinSolverIndefDenseMagmaBuKa_old2 : public hiopLinSolverIndefDense
{
public:
  hiopLinSolverIndefDenseMagmaBuKa_old2(int n, hiopNlpFormulation* nlp_);
  virtual ~hiopLinSolverIndefDenseMagmaBuKa_old2();

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
  virtual bool compute_inertia(int n, int *ipiv_in, int& posEig, int& negEig, int& zeroEig); 

protected:
  int* ipiv_;

  //allocated on demand; for example, it may only be required by Magma SIDI.
  double* work_;

protected:

private:
  hiopLinSolverIndefDenseMagmaBuKa_old2() { assert(false); }
};

#endif //0


} //end namespace hiop

#endif //of HIOP_USE_MAGMA
#endif
