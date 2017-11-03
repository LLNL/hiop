#ifndef HIOP_VECTOR
#define HIOP_VECTOR

#ifdef WITH_MPI
#include "mpi.h"
#else 
#define MPI_Comm int
#define MPI_COMM_NULL 0
#include <cstddef>
#endif 


#include <cstdio>

namespace hiop
{

//forward declarations
class hiopInnerProdWeight;

class hiopVector
{
public:
  hiopVector() { n=0;}
  virtual ~hiopVector() {};
  /** Set all elements to zero. */
  virtual void setToZero() = 0;
  /** Set all elements  to  c */
  virtual void setToConstant( double c ) = 0;
  /** Set all elements that are not zero in ix to  c, and the rest to 0 */
  virtual void setToConstant_w_patternSelect( double c, const hiopVector& ix)=0;
  /** Copy the elements of v */
  virtual void copyFrom(const hiopVector& v ) = 0;
  /* Copy v in 'this' starting at start_index in  'this'. */
  virtual void copyFromStarting(const hiopVector& v, int start_index) = 0;
  /* Copy 'this' to double array, which is assumed to be at least of 'n_local' size.*/
  virtual void copyTo(double* dest) const = 0;
  /* Copy 'this' to v starting at start_index in 'this'. */
  virtual void copyToStarting(hiopVector& v, int start_index) = 0;
  /** Return the two norm */
  virtual double twonorm() const = 0;
  /** Return the infinity norm */
  virtual double infnorm() const = 0;
  /** Return the one norm */
  virtual double onenorm() const = 0;
  /** Return the weighted norm */
  //virtual double Hnorm(const hiopInnerProdWeight& H) const = 0;
  /** Multiply the components of this by the components of v. */
  virtual void componentMult( const hiopVector& v ) = 0;
  /** Divide the components of this hiopVector by the components of v. */
  virtual void componentDiv ( const hiopVector& v ) = 0;
  /* Elements of this that corespond to nonzeros in ix are divided by elements of v.
   * The rest of elements of this are set to zero.
   */
  virtual void componentDiv_p_selectPattern( const hiopVector& v, const hiopVector& ix) = 0;
  /** Scale each element of this  by the constant alpha */
  virtual void scale( double alpha ) = 0;
  /** this += alpha * x */
  virtual void axpy  ( double alpha, const hiopVector& x ) = 0;
  /** this += alpha * x * z */
  virtual void axzpy ( double alpha, const hiopVector& x, const hiopVector& z ) = 0;
  /** this += alpha * x / z */
  virtual void axdzpy( double alpha, const hiopVector& x, const hiopVector& z ) = 0;
  /** Add c to the elements of this */
  virtual void addConstant( double c ) = 0;
  virtual void addConstant_w_patternSelect(double c, const hiopVector& ix) = 0;
  /** Return the dot product of this hiopVector with v */
  virtual double dotProductWith( const hiopVector& v ) const = 0;
  /** Negate all the elements of this */
  virtual void negate() = 0;
  /** Invert (1/x) the elements of this */
  virtual void invert() = 0;
  /* compute log barrier term, that is sum{ln(x_i):i=1,..,n}. !!!This is a "local" function. */
  virtual double logBarrier(const hiopVector& select) const = 0;
  /* adds the gradient of the log barrier, namely this=this+alpha*1/select(x) */
  virtual void addLogBarrierGrad(double alpha, const hiopVector& x, const hiopVector& select)=0;

  /* computes the log barrier's linear damping term of the Filter-IPM method of WaectherBiegler (see paper, section 3.7).
   * Essentially compute  kappa_d*mu* \sum { this[i] | ixleft[i]==1 and ixright[i]==0 } */
  virtual double linearDampingTerm(const hiopVector& ixleft, const hiopVector& ixright, 
				   const double& mu, const double& kappa_d)const=0;
  /** True if all elements of this are positive. */
  virtual int allPositive() = 0;
  /** True if elements corresponding to nonzeros in w are all positive */
  virtual int allPositive_w_patternSelect(const hiopVector& w) = 0;
  /** Return the minimum value in this vector, and the index at which it occurs. */
  virtual void min( double& m, int& index ) const = 0;
  /** Project the vector into the bounds, used for shifting the ini pt in the bounds */
  virtual void projectIntoBounds(const hiopVector& xl, const hiopVector& ixl, 
				 const hiopVector& xu, const hiopVector& ixu,
				 double kappa1, double kappa2) = 0;
  /* max{a\in(0,1]| x+ad >=(1-tau)x} */
  virtual double fractionToTheBdry(const hiopVector& dx, const double& tau) const = 0;
  virtual double fractionToTheBdry_w_pattern(const hiopVector& dx, const double& tau, const hiopVector& ix) const = 0;
  /** Entries corresponding to zeros in ix are set to zero */
  virtual void selectPattern(const hiopVector& ix) = 0;
  /** checks whether entries in this matches pattern in ix */
  virtual bool matchesPattern(const hiopVector& ix) const = 0;

  /** allocates a vector that mirrors this, but doesn't copy the values  */
  //virtual hiopVector* new_alloc() const = 0;
  /** allocates a vector that mirrors this, and copies the values  */
  //virtual hiopVector* new_copy() const = 0;

  /* dual adjustment -> see hiopIterate::adjustDuals_primalLogHessian */
  virtual void adjustDuals_plh(const hiopVector& x, const hiopVector& ix, const double& mu, const double& kappa)=0;

  /* prints up to max_elems (by default all), on rank 'rank' (by default on all) */
  virtual void print(FILE*, const char* message=NULL,int max_elems=-1, int rank=-1) const = 0;
  
  inline long long get_size() const { return n; }
protected:
  long long n; //we assume sequential data

protected:
  hiopVector(const hiopVector& v) : n(v.n) {};
};

class hiopVectorPar : public hiopVector
{
public:
  hiopVectorPar(const long long& glob_n, long long* col_part=NULL, MPI_Comm comm=MPI_COMM_NULL);
  virtual ~hiopVectorPar();

  virtual void setToZero();
  virtual void setToConstant( double c );
  virtual void setToConstant_w_patternSelect(double c, const hiopVector& select);
  virtual void copyFrom(const hiopVector& v );
  virtual void copyFrom(const double* v_local_data); //v should be of length at least n_local
  virtual void copyFromStarting(const hiopVector& v, int start_index);
  virtual void copyTo(double* dest) const;
  virtual void copyToStarting(hiopVector& v, int start_index);
  virtual double twonorm() const;
  virtual double dotProductWith( const hiopVector& v ) const;
  virtual double infnorm() const;
  virtual double infnorm_local() const;
  virtual double onenorm() const;
  virtual double onenorm_local() const; 
  //virtual double Hnorm(const hiopInnerProdWeight& H) const;
  //virtual double Hnorm_local(const hiopInnerProdWeight& H) const;
  virtual void componentMult( const hiopVector& v );
  virtual void componentDiv ( const hiopVector& v );
  virtual void componentDiv_p_selectPattern( const hiopVector& v, const hiopVector& ix);
  virtual void scale( double alpha );
  /** this += alpha * x */
  virtual void axpy  ( double alpha, const hiopVector& x );
  /** this += alpha * x * z */
  virtual void axzpy ( double alpha, const hiopVector& x, const hiopVector& z );
  /** this += alpha * x / z */
  virtual void axdzpy( double alpha, const hiopVector& x, const hiopVector& z );
  virtual void axdzpy_w_pattern( double alpha, const hiopVector& x, const hiopVector& z, const hiopVector& select ); 
  /** Add c to the elements of this */
  virtual void addConstant( double c );
  virtual void addConstant_w_patternSelect(double c, const hiopVector& ix);
  virtual void min( double& m, int& index ) const;
  virtual void negate();
  virtual void invert();
  virtual double logBarrier(const hiopVector& select) const;
  virtual void addLogBarrierGrad(double alpha, const hiopVector& x, const hiopVector& select);

  virtual double linearDampingTerm(const hiopVector& ixl_select, const hiopVector& ixu_select, 
				   const double& mu, const double& kappa_d) const;
  virtual int allPositive();
  virtual int allPositive_w_patternSelect(const hiopVector& w);
  virtual void projectIntoBounds(const hiopVector& xl, const hiopVector& ixl, 
				 const hiopVector& xu, const hiopVector& ixu,
				 double kappa1, double kappa2);
  virtual double fractionToTheBdry(const hiopVector& dx, const double& tau) const;
  virtual double fractionToTheBdry_w_pattern(const hiopVector& dx, const double& tau, const hiopVector& ix) const;
  virtual void selectPattern(const hiopVector& ix);
  virtual bool matchesPattern(const hiopVector& ix) const;

  virtual hiopVectorPar* alloc_clone() const;
  virtual hiopVectorPar* new_copy () const;

  virtual void adjustDuals_plh(const hiopVector& x, const hiopVector& ix, const double& mu, const double& kappa);

  virtual void print(FILE*, const char* withMessage=NULL, int max_elems=-1, int rank=-1) const;

  /* more accessers */
  inline long long get_local_size() const { return n_local; }
  inline double* local_data() { return data; }
  inline const double* local_data_const() const { return data; }

protected:
  MPI_Comm comm;
  double* data;
  long long glob_il, glob_iu;
  long long n_local;
private:
  /** copy constructor, for internal/private use only (it doesn't copy the elements.) */
  hiopVectorPar(const hiopVectorPar&);

};

}
#endif
