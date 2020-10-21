
#ifndef HIOP_PRIDECOMP
#define HIOP_PRIDECOMP

#include "hiopInterfacePrimalDecomp.hpp"

#include <cassert>

namespace hiop
{

 
class hiopAlgPrimalDecomposition
{
public:

  //constructor
  hiopAlgPrimalDecomposition(hiopInterfacePriDecProblem* prob_in,
                             MPI_Comm comm_world=MPI_COMM_WORLD)
    : master_prob_(prob_in), comm_world_(comm_world)
  {
    S_ = master_prob_->get_num_rterms();
    n_ = master_prob_->get_num_vars();

    //determine rank and rank type
    int ierr = MPI_Comm_rank(comm_world, &my_rank_); assert(ierr == MPI_SUCCESS);
    if(my_rank_==0) { my_rank_type_ = 0; }
    else {
      if(my_rank_==1) { my_rank_type_ = 1;}
      else            { my_rank_type_ = 2;}
    }

    x_ = new double[n_];
    
  }
  virtual ~hiopAlgPrimalDecomposition()
  {
    delete [] x_;
  }

  //we should make the public methods to look like hiopAlgFilterIPMBase
  
  virtual hiopSolveStatus run()
  {

    //initial point = ?
    //for now set to all zero
    for(int i=0; i<n_; i++) {
      x_[i] = 0.;
    }
    
    bool bret;
    
    //all the cool stuff

    //this is an example of the usage of the classes' design and has almost nothing to do with
    //the actual algorithm (master rank does all the computations)
    
    //for now solve the basecase, add quadratic_regularizatin (does nothing for now), and resolve master
    if(my_rank_type_ == 0) {
      //solve master problem (solver rank supposed to do it)
      solver_status_ = master_prob_->solve_master(x_);
      //todo error control

      //
      //workers
      //
      double rec_val = 0., aux;
      for(int ri=0; ri<S_; ri++) {
        aux = 0.;
        bret = master_prob_->eval_f_rterm(ri, n_, x_, aux);
        if(!bret) {
          //todo
        }
        rec_val += aux;
      }

      double grad_acc[n_];
      for(int i=0; i<n_; i++) grad_acc[i] = 0.;
      
      double grad_aux[n_];
      for(int ri=0; ri<S_; ri++) {
        bret = master_prob_->eval_grad_rterm(ri, n_, x_, grad_aux);
        if(!bret)
        {
          //todo
        }
        for(int i=0; i<n_; i++)
          grad_acc[i] += grad_aux[i];
      }

      //do something with the func eval and gradient to determine the quadratic regularization 
      
      bret = master_prob_->set_quadratic_regularization(/* params = ? */);
      if(!bret)
      {
        //todo
      }

      solver_status_ = master_prob_->solve_master(x_);
      
      return solver_status_;
    } else {
      return Solve_Success;
    }
    
  }

  virtual double getObjective() const
  {
    return master_prob_->get_objective();
  }
  void getSolution(double* x) const
  {
    master_prob_->get_solution(x);
  }
  void getDualSolutions(double* zl, double* zu, double* lambda)
  {
    assert(false && "not implemented");
  }
  /* returns the status of the solver */
  inline hiopSolveStatus getSolveStatus() const
  {
    return solver_status_;
  }
  /* returns the number of iterations, meaning how many times the master was solved */
  int getNumIterations() const
  {
    assert(false && "not yet implemented");
    return 9;
  }
  
private:
  //MPI stuff
  MPI_Comm comm_world_;
  MPI_Comm my_rank_;

  //master (0), solver(1), or worker(2)
  int my_rank_type_;
  
  //pointer to the problem to be solved (passed as argument)
  hiopInterfacePriDecProblem* master_prob_;
  hiopSolveStatus solver_status_;
  
  //current primal iterate
  double* x_;

  //dimension of x_
  size_t n_;

  //number of recourse terms
  size_t S_;
};

} //end of namespace

#endif
