// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// Written by Cosmin G. Petra, petra1@llnl.gov.
// LLNL-CODE-742473. All rights reserved.
//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop. HiOp 
// is released under the BSD 3-clause license (https://opensource.org/licenses/BSD-3-Clause). 
// Please also read “Additional BSD Notice” below.
//
// Redistribution and use in source and binary forms, with or without modification, 
// are permitted provided that the following conditions are met:
// i. Redistributions of source code must retain the above copyright notice, this list 
// of conditions and the disclaimer below.
// ii. Redistributions in binary form must reproduce the above copyright notice, 
// this list of conditions and the disclaimer (as noted below) in the documentation and/or 
// other materials provided with the distribution.
// iii. Neither the name of the LLNS/LLNL nor the names of its contributors may be used to 
// endorse or promote products derived from this software without specific prior written 
// permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
// SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR 
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS 
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED 
// AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Additional BSD Notice
// 1. This notice is required to be provided under our contract with the U.S. Department 
// of Energy (DOE). This work was produced at Lawrence Livermore National Laboratory under 
// Contract No. DE-AC52-07NA27344 with the DOE.
// 2. Neither the United States Government nor Lawrence Livermore National Security, LLC 
// nor any of their employees, makes any warranty, express or implied, or assumes any 
// liability or responsibility for the accuracy, completeness, or usefulness of any 
// information, apparatus, product, or process disclosed, or represents that its use would
// not infringe privately-owned rights.
// 3. Also, reference herein to any specific commercial products, process, or services by 
// trade name, trademark, manufacturer or otherwise does not necessarily constitute or 
// imply its endorsement, recommendation, or favoring by the United States Government or 
// Lawrence Livermore National Security, LLC. The views and opinions of authors expressed 
// herein do not necessarily state or reflect those of the United States Government or 
// Lawrence Livermore National Security, LLC, and shall not be used for advertising or 
// product endorsement purposes.

#ifndef HIOP_INTERFACE_BASE
#define HIOP_INTERFACE_BASE

#include "hiop_defs.hpp"

#ifdef HIOP_USE_MPI
#include "mpi.h"

#else
#ifndef MPI_Comm
#define MPI_Comm int
#endif

#ifndef MPI_COMM_WORLD
#define MPI_COMM_WORLD 0
#endif 
#endif

namespace hiop
{
  /** Solver status codes. */
enum hiopSolveStatus {
  //(partial) success 
  Solve_Success=0,
  Solve_Success_RelTol=1,
  Solve_Acceptable_Level=2,
  Infeasible_Problem=5,
  Iterates_Diverging=6,
  Feasible_Not_Optimal = 7,
  //solver stopped based on user-defined criteria that are not related to optimality
  Max_Iter_Exceeded=10,
  Max_CpuTime_Exceeded=11,
  User_Stopped=12,

  //NLP algorithm/solver reports issues in solving the problem and stops without being certain 
  //that is solved the problem to optimality or that the problem is infeasible.
  //Feasible_Point_Found, 
  NlpAlgorithm_failure=-1, 
  Diverging_Iterates=-2,
  Search_Dir_Too_Small=-3,
  Steplength_Too_Small=-4,
  Err_Step_Computation=-5,
  //errors related to user-provided data (e.g., inconsistent problem specification, 'nans' in the 
  //function/sensitivity evaluations, invalid options)
  Invalid_Problem_Definition=-11,
  Invalid_Parallelization=-12,
  Invalid_UserOption=-13,
  Invalid_Number=-14,

  //ungraceful errors and returns
  Exception_Unrecoverable=-100,
  Memory_Alloc_Problem=-101,
  SolverInternal_Error=-199,

  //unknown NLP solver errors or return codes
  UnknownNLPSolveStatus=-1000,

  //intermediary statuses for the solver
  NlpSolve_IncompleteInit=-10001,
  NlpSolve_SolveNotCalled=-10002,
  NlpSolve_Pending=-10003
};

/** The base interface class */
class hiopInterfaceBase
{
  /** Base class for the solver's interface that has no assumptions how the 
   *  matrices are stored. The vectors are dense and distributed row-wise. 
   *  The data distribution is decided by the calling code (that implements 
   *  this interface) and specified to the optimization via 'get_vecdistrib_info'
   *
   *  Two possible implementations are for sparse NLPs and NLPs with small 
   *  number of global constraints.
   *  
   *  
   */
public:
  enum NonlinearityType{ hiopLinear=0, hiopQuadratic, hiopNonlinear};
public:
  hiopInterfaceBase() {};
  virtual ~hiopInterfaceBase() {};

  /** problem dimensions: n number of variables, m number of constraints */
  virtual bool get_prob_sizes(long long& n, long long& m)=0;
  /** bounds on the variables 
   *  (xlow<=-1e20 means no lower bound, xupp>=1e20 means no upper bound) */
  virtual bool get_vars_info(const long long& n, double *xlow, double* xupp, NonlinearityType* type)=0;
  /** bounds on the constraints 
   *  (clow<=-1e20 means no lower bound, cupp>=1e20 means no upper bound) */
  virtual bool get_cons_info(const long long& m, double* clow, double* cupp, NonlinearityType* type)=0;

  //! initial point specification

  /** Objective function evaluation
   *  When MPI enabled, each rank returns the obj. value. Also, x points to the local entries and 
   *  the function is responsible for knowing the local buffer size.
   */
  virtual bool eval_f(const long long& n, const double* x, bool new_x, double& obj_value)=0;
  /** Gradient of objective.
   *  When MPI enabled, each rank works only with local buffers x and gradf.
   */
  virtual bool eval_grad_f(const long long& n, const double* x, bool new_x, double* gradf)=0;

  /** Evaluates a subset of the constraints cons(x) (where clow<=cons(x)<=cupp). The subset is of size
   *  'num_cons' and is described by indexes in the 'idx_cons' array. The methods may be called 
   *  multiple times, each time for a subset of the constraints, for example, for the 
   *  subset containing the equalities and for the subset containing the inequalities. However, each 
   *  constraint will be inquired EXACTLY once. This is done for performance considerations, to avoid 
   *  temporary holders and memory copying.
   *
   *  Parameters:
   *   - n, m: the global number of variables and constraints
   *   - num_cons, idx_cons (array of size num_cons): the number and indexes of constraints to be evaluated
   *   - x: the point where the constraints are to be evaluated
   *   - new_x: whether x has been changed from the previous call to f, grad_f, or Jac
   *   - cons: array of size num_cons containing the value of the  constraints indicated by idx_cons
   *  
   *  When MPI enabled, every rank populates 'cons' since the constraints are not distributed.
   */
  virtual bool eval_cons(const long long& n, const long long& m, 
			 const long long& num_cons, const long long* idx_cons,  
			 const double* x, bool new_x, 
			 double* cons)=0;
  /** Jacobian of constraints is to be specified in a derived class since it can be sparse 
   *  or dense+distributed
  virtual bool eval_Jac_cons(const long long& n, const long long& m, const double* x, bool new_x, ...)
  */

  /** pass the communicator, defaults to MPI_COMM_WORLD (dummy for non-MPI builds)  */
  virtual bool get_MPI_comm(MPI_Comm& comm_out) { comm_out=MPI_COMM_WORLD; return true;}
  /**  column partitioning specification for distributed memory vectors 
  *  Process P owns cols[P], cols[P]+1, ..., cols[P+1]-1, P={0,1,...,NumRanks}.
  *  Example: for a vector x of 6 elements on 3 ranks, the col partitioning is cols=[0,2,4,6].
  *  The caller manages memory associated with 'cols', array of size NumRanks+1 
  */
  virtual bool get_vecdistrib_info(long long global_n, long long* cols) {
    return false; //defaults to serial 
  }

  /* Method providing a primal starting point. This point is subject to internal adjustments in hiOP.
   * The method returns true (and populate x0) or return false, in which case hiOP will use set 
   * x0 to all zero (still subject to internal adjustement).
   *
   * ToDo: provide API for a full, primal-dual restart. 
   */
  virtual bool get_starting_point(const long long&n, double* x0) { return false; }

  /** callback for the optimal solution.
   *  Note that:
   *   i. x, z_L, z_U contain only the array slice that is local to the calling process
   *  ii. g, lambda are replicated across all processes, which means they can be used as-is, without reducing them.
   * iii. all other scalar quantities are replicated across all processes, which means they can be used as-is, 
   * without reducing them.
   */
  virtual void solution_callback(hiopSolveStatus status,
				 int n, const double* x,
				 const double* z_L,
				 const double* z_U,
				 int m, const double* g,
				 const double* lambda,
				 double obj_value) { };

  /** Callback for the iteration: at the end of each iteration. This is NOT called during the line-searches.
   * Note: all the notes for @solution_callback apply.
   */
  virtual bool iterate_callback(int iter, double obj_value,
				int n, const double* x,
				const double* z_L,
				const double* z_U,
				int m, const double* g,
				const double* lambda,
				double inf_pr, double inf_du,
				double mu,
				double alpha_du, double alpha_pr,
				int ls_trials) {return true;}
  


private:
  hiopInterfaceBase(const hiopInterfaceBase& ) {};
  void operator=(const hiopInterfaceBase&) {};
};

/** Specialized interface for NLPs with 'global' but few constraints. 
 */
class hiopInterfaceDenseConstraints : public hiopInterfaceBase 
{
public:
  hiopInterfaceDenseConstraints() {};
  virtual ~hiopInterfaceDenseConstraints() {};
  /** Evaluates the Jacobian of the subset of constraints indicated by idx_cons and of size num_cons.
   *  Example: Assuming idx_cons[k]=i, which means that the gradient of the (i+1)th constraint is
   *  to be evaluated, one needs to do Jac[k][0]=d/dx_0 con_i(x), Jac[k][1]=d/dx_1 con_i(x), ...
   *  When MPI enabled, each rank computes only the local columns of the Jacobian, that is the partials
   *  with respect to local variables.
   *
   *  Parameters: see eval_cons
   */
  virtual bool eval_Jac_cons(const long long& n, const long long& m, 
			     const long long& num_cons, const long long* idx_cons,  
			     const double* x, bool new_x,
			     double** Jac) = 0;
};

}
#endif
