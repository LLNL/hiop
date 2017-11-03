#ifndef HIOP_NLP_FORMULATION
#define HIOP_NLP_FORMULATION

#include "hiopInterface.hpp"
#include "hiopVector.hpp"
#include "hiopMatrix.hpp"

#ifdef WITH_MPI
#include "mpi.h"  
#endif

#include "hiopRunStats.hpp"
#include "hiopLogger.hpp"
#include "hiopOptions.hpp"

namespace hiop
{
class hiopInnerProdWeight; 
class hiopInnerProdMatrixFreeWeight;

class hiopNlpFormulation
{
public:
  hiopNlpFormulation(hiopInterfaceBase& interface);
  virtual ~hiopNlpFormulation();

  /* starting point */
  virtual bool get_starting_point(hiopVector& x0)=0;
  /** linear algebra factory */
  virtual hiopVector* alloc_primal_vec() const=0;
  virtual hiopVector* alloc_dual_eq_vec() const=0;
  virtual hiopVector* alloc_dual_ineq_vec() const=0;
  virtual hiopVector* alloc_dual_vec() const=0;

  virtual void user_callback_solution(hiopSolveStatus status,
				      const hiopVector& x,
				      const hiopVector& z_L,
				      const hiopVector& z_U,
				      const hiopVector& c, const hiopVector& d,
				      const hiopVector& yc, const hiopVector& yd,
				      double obj_value)=0;
  virtual bool user_callback_iterate(int iter, double obj_value,
				     const hiopVector& x,
				     const hiopVector& z_L,
				     const hiopVector& z_U,
				     const hiopVector& c, const hiopVector& d,
				     const hiopVector& yc, const hiopVector& yd,
				     double inf_pr, double inf_du,
				     double mu,
				     double alpha_du, double alpha_pr,
				     int ls_trials)=0;
  virtual bool applyH(double* x) = 0;
  virtual bool applyHInv(double* x) = 0;

  hiopInnerProdWeight* H;
  //inline hiopInnerProdWeight& get_H() {return *H;}

  /* outputing and debug-related functionality */
  hiopLogger* log;
  hiopRunStats runStats;
  hiopOptions* options;
  //prints a summary of the problem
  virtual void print(FILE* f=NULL, const char* msg=NULL, int rank=-1) const = 0;
#ifdef WITH_MPI
  inline MPI_Comm get_comm() const { return comm; }
  inline int      get_rank() const { return rank; }
  inline int      get_num_ranks() const { return num_ranks; }
#endif
protected:
  
#ifdef WITH_MPI
  MPI_Comm comm;
  int rank, num_ranks;
#endif
private:
  hiopNlpFormulation(const hiopNlpFormulation&) {};
};

/* Class is for NLPs that has a small number of general/dense constraints *
 * Splits the constraints in ineq and eq.
 */
class hiopNlpDenseConstraints : public hiopNlpFormulation
{
public:
  hiopNlpDenseConstraints(hiopInterfaceDenseConstraints& interface);
  virtual ~hiopNlpDenseConstraints();

  /* wrappers for the interface calls. Can be overridden for specialized formulations required by the algorithm */
  virtual bool eval_f(const double* x, bool new_x, double& f);
  virtual bool eval_grad_f(const double* x, bool new_x, double* gradf);
  virtual bool eval_c(const double*x, bool new_x, double* c);
  virtual bool eval_d(const double*x, bool new_x, double* d);
  virtual bool eval_d(const hiopVector& x, bool new_x, hiopVector& d);
  virtual bool eval_Jac_c(const double* x, bool new_x, double** Jac_c);
  virtual bool eval_Jac_d(const double* x, bool new_x, double** Jac_d);
  virtual bool get_starting_point(hiopVector& x0);

  /* linear algebra factory */
  virtual hiopVector* alloc_primal_vec() const;
  virtual hiopVector* alloc_dual_eq_vec() const;
  virtual hiopVector* alloc_dual_ineq_vec() const;
  virtual hiopVector* alloc_dual_vec() const;
  virtual hiopMatrixDense* alloc_Jac_c() const;
  virtual hiopMatrixDense* alloc_Jac_d() const;
  /* this is in general for a dense matrix witn n_vars cols and a small number of 
   * 'nrows' rows. The second argument indicates how much total memory should the
   * matrix (pre)allocate.
   */
  virtual hiopMatrixDense* alloc_multivector_primal(int nrows, int max_rows=-1) const;

  virtual bool applyH(double* x)    { return interface.applyH(x); };
  virtual bool applyHInv(double* x) { return interface.applyHInv(x); };

  virtual inline 
  void user_callback_solution(hiopSolveStatus status,
			      const hiopVector& x,
			      const hiopVector& z_L,
			      const hiopVector& z_U,
			      const hiopVector& c, const hiopVector& d,
			      const hiopVector& yc, const hiopVector& yd,
			      double obj_value) {
    const hiopVectorPar& xp = dynamic_cast<const hiopVectorPar&>(x);
    const hiopVectorPar& zl = dynamic_cast<const hiopVectorPar&>(z_L);
    const hiopVectorPar& zu = dynamic_cast<const hiopVectorPar&>(z_U);
    assert(xp.get_size()==n_vars);
    assert(c.get_size()+d.get_size()==n_cons);
    //!petra: to do: assemble (c,d) into cons and (yc,yd) into lambda based on cons_eq_mapping and cons_ineq_mapping
    interface.solution_callback(status, 
				(int)n_vars, xp.local_data_const(), zl.local_data_const(), zu.local_data_const(),
				(int)n_cons, NULL, //cons, 
				NULL, //lambda,
				obj_value);
  };
  virtual inline 
  bool user_callback_iterate(int iter, double obj_value,
			     const hiopVector& x, const hiopVector& z_L, const hiopVector& z_U,
			     const hiopVector& c, const hiopVector& d, const hiopVector& yc, const hiopVector& yd,
			     double inf_pr, double inf_du, double mu, double alpha_du, double alpha_pr, int ls_trials){
    const hiopVectorPar& xp = dynamic_cast<const hiopVectorPar&>(x);
    const hiopVectorPar& zl = dynamic_cast<const hiopVectorPar&>(z_L);
    const hiopVectorPar& zu = dynamic_cast<const hiopVectorPar&>(z_U);
    assert(xp.get_size()==n_vars);
    assert(c.get_size()+d.get_size()==n_cons);
    //!petra: to do: assemble (c,d) into cons and (yc,yd) into lambda based on cons_eq_mapping and cons_ineq_mapping
    return interface.iterate_callback(iter, obj_value, 
				      (int)n_vars, xp.local_data_const(), zl.local_data_const(), zu.local_data_const(),
				      (int)n_cons, NULL, //cons, 
				      NULL, //lambda,
				      inf_pr, inf_du, mu, alpha_du, alpha_pr,  ls_trials);
      }
  /** const accessors */
  inline const hiopVectorPar& get_xl ()  const { return *xl;   }
  inline const hiopVectorPar& get_xu ()  const { return *xu;   }
  inline const hiopVectorPar& get_ixl()  const { return *ixl;  }
  inline const hiopVectorPar& get_ixu()  const { return *ixu;  }
  inline const hiopVectorPar& get_dl ()  const { return *dl;   }
  inline const hiopVectorPar& get_du ()  const { return *du;   }
  inline const hiopVectorPar& get_idl()  const { return *idl;  }
  inline const hiopVectorPar& get_idu()  const { return *idu;  }
  inline const hiopVectorPar& get_crhs() const { return *c_rhs;}
  inline long long n() const      {return n_vars;}
  inline long long m() const      {return n_cons;}
  inline long long m_eq() const   {return n_cons_eq;}
  inline long long m_ineq() const {return n_cons_ineq;}
  inline long long n_low() const  {return n_bnds_low;}
  inline long long n_upp() const  {return n_bnds_upp_local;;}
  inline long long n_low_local() const {return n_bnds_low_local;}
  inline long long n_upp_local() const {return n_bnds_upp_local;}
  inline long long m_ineq_low() const {return n_ineq_low;}
  inline long long m_ineq_upp() const {return n_ineq_upp;}
  inline long long n_complem()  const {return m_ineq_low()+m_ineq_upp()+n_low()+n_upp();}
  //inline long long n_complem_local()  const {return m_ineq_low()+m_ineq_upp()+n_low_local()+n_upp_local();}
  virtual void print(FILE* f=NULL, const char* msg=NULL, int rank=-1) const;
private:
  /* problem data */
  //various sizes
  long long n_vars, n_cons, n_cons_eq, n_cons_ineq;
  long long n_bnds_low, n_bnds_low_local, n_bnds_upp, n_bnds_upp_local, n_ineq_low, n_ineq_upp;
  long long n_bnds_lu, n_ineq_lu;
  hiopVectorPar *xl, *xu, *ixu, *ixl; //these will be global, memory distributed
  hiopInterfaceBase::NonlinearityType* vars_type; //C array containing the types for local vars

  hiopVectorPar *c_rhs; //local
  hiopInterfaceBase::NonlinearityType* cons_eq_type;

  hiopVectorPar *dl, *du,  *idl, *idu; //these will be local
  hiopInterfaceBase::NonlinearityType* cons_ineq_type;
  // keep track of the constraints indexes in the original, user's formulation
  long long *cons_eq_mapping, *cons_ineq_mapping; 
private:

  /* interface implemented and provided by the user */
  hiopInterfaceDenseConstraints& interface;
};

}
#endif
