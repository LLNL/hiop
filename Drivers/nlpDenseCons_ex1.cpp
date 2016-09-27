#include "hiopVector.hpp"
#include "hiopNlpFormulation.hpp"
#include "hiopInterface.hpp"
#include "hiopAlgFilterIPM.hpp"

#ifdef WITH_MPI
#include "mpi.h"
#else
#define MPI_COMM_WORLD 0
#define MPI_Comm int
#endif

#include <cmath>
#include <cstdio>
#include <cassert>

/* Example 1: min sum{x_i^2 | i=1,..,n} s.t. x_2>=1 */

class Ex1Interface : public hiopInterfaceDenseConstraints
{
public: 
  Ex1Interface(int num_vars=4)
    : n_vars(num_vars), n_cons(0), comm(MPI_COMM_WORLD)
  {
    comm_size=1; my_rank=0; 
#ifdef WITH_MPI
    int ierr = MPI_Comm_size(comm, &comm_size); assert(MPI_SUCCESS==ierr);
    ierr = MPI_Comm_rank(comm, &my_rank); assert(MPI_SUCCESS==ierr);
#endif
    // set up vector distribution for primal variables - easier to store it as a member in this simple example
    col_partition = new long long[comm_size];
    long long quotient=n_vars/comm_size, remainder=n_vars-comm_size*quotient;
    if(my_rank==0) printf("reminder=%d quotient=%d\n", remainder, quotient);
    int i=0; col_partition[i]=0; i++;
    while(i<=remainder) { col_partition[i] = col_partition[i-1]+quotient+1; i++; }
    while(i<=comm_size) { col_partition[i] = col_partition[i-1]+quotient;   i++; }

    if(my_rank==0) {
      for(int i=0;i<=comm_size;i++) 
        printf("%3d ", col_partition[i]);
      printf("\n");
    }
  }
  virtual ~Ex1Interface()
  {
    delete[] col_partition;
  }
  bool get_prob_sizes(long long& n, long long& m)
  { n=n_vars; m=n_cons; return true; }

  bool get_vars_info(const long long& n, double *xlow, double* xupp, NonlinearityType* type)
  {
    int i_local;
    for(int i_global=col_partition[my_rank]; i_global<col_partition[my_rank+1]; i_global++) {
      i_local=idx_global2local(n,i_global);
      if(i_global==2) xlow[i_local]= 1.0;
      else            xlow[i_local]=-1e20;
      type[i_local] = hiopLinear;
      xupp[i_local] = 1e20;
    }
    return true;
  }
  bool get_cons_info(const long long& m, double* clow, double* cupp, NonlinearityType* type)
  {
    assert(m==n_cons);
    //no constraints
    return true;
  }
  bool eval_f(const long long& n, const double* x, bool new_x, double& obj_value)
  {
    int n_local=col_partition[my_rank+1]-col_partition[my_rank];
    obj_value=0.; 
    for(int i=0;i<n_local;i++) obj_value += x[i]*x[i];
#ifdef WITH_MPI
    double obj_global;
    int ierr=MPI_Allreduce(&obj_value, &obj_global, 1, MPI_DOUBLE, MPI_SUM, comm); assert(ierr==MPI_SUCCESS);
    obj_value=obj_global;
#endif
    return true;
  }
  bool eval_grad_f(const long long& n, const double* x, bool new_x, double* gradf)
  {
    int n_local=col_partition[my_rank+1]-col_partition[my_rank];
    for(int i=0;i<n_local;i++) gradf[i] = 2*x[i];
    return true;
  }
  /** Sum(x[i])<=10 and sum(x[i])>= 1  (we pretend are different)
   */
  bool eval_cons(const long long& n, 
		 const long long& m,  
		 const long long& num_cons, const long long* idx_cons,
		 const double* x, bool new_x, double* cons)
  {
    assert(n==n_vars); assert(m==n_cons);
    //no constraints
    return true;
  }

  bool eval_Jac_cons(const long long& n, const long long& m, 
		     const long long& num_cons, const long long* idx_cons,
                     const double* x, bool new_x, double** Jac) 
  {
    assert(n==n_vars); assert(m==n_cons);
    //no constraints
    return true;
  }

  bool get_vecdistrib_info(long long global_n, long long* cols)
  {
    if(global_n==n_vars)
      for(int i=0; i<=comm_size; i++) cols[i]=col_partition[i];
    else 
      assert(false && "You shouldn't need distrib info for this size.");
    return true;
  }
private:
  int n_vars, n_cons;
  MPI_Comm comm;
  int my_rank, comm_size;
  long long* col_partition;
public:
  inline int idx_local2global(long long global_n, int idx_local) 
  { 
    assert(idx_local + col_partition[my_rank]<col_partition[my_rank+1]);
    if(global_n==n_vars)
      return idx_local + col_partition[my_rank]; 
    assert(false && "You shouldn't need global index for a vector of this size.");
  }
  inline int idx_global2local(long long global_n, long long idx_global)
  {
    assert(idx_global>=col_partition[my_rank]   && "global index does not belong to this rank");
    assert(idx_global< col_partition[my_rank+1] && "global index does not belong to this rank");
    assert(global_n==n_vars && "your global_n does not match the number of variables?");
    return idx_global-col_partition[my_rank];
  }
};

int main(int argc, char **argv)
{
  int rank=0, numRanks=1;
#ifdef WITH_MPI
  MPI_Init(&argc, &argv);
  assert(MPI_SUCCESS==MPI_Comm_rank(MPI_COMM_WORLD,&rank));
  assert(MPI_SUCCESS==MPI_Comm_size(MPI_COMM_WORLD,&numRanks));
  if(0==rank) printf("Support for MPI is enabled\n");
#endif

  long long numVars=3;
  Ex1Interface problem(numVars);
  if(rank==0) printf("interface created\n");
  hiopNlpDenseConstraints nlp(problem);
  if(rank==0) printf("nlp formulation created\n");
  
  hiopAlgFilterIPM solver(&nlp);
  solver.run();

#ifdef WITH_MPI
  MPI_Finalize();
#endif
  
  return 0;
}
