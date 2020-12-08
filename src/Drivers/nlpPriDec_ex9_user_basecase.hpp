#include "nlpSparse_ex6.hpp"

class PriDecBasecaseProblemEx9 : public Ex6
{
public:
  PriDecBasecaseProblemEx9(int n)
    : Ex6(n), rec_evaluator_(nullptr)
  {
  }

  virtual ~PriDecBasecaseProblemEx9()
  {
    delete rec_evaluator_;
  }

  bool eval_f(const long long& n, const double* x, bool new_x, double& obj_value)
  {
    if(!Ex6::eval_f(n, x, new_x, obj_value)) {
      return false;
    }
    //add regularization to the objective based on rec_evaluator_
    assert(false);
    
    return true;
  }

  bool eval_grad_f(const long long& n, const double* x, bool new_x, double* gradf)
  {
    if(!Ex6::eval_grad_f(n, x, new_x, gradf)) {
      return false;
    }
    //add regularization gradient
    assert(false);
    
    return true;
  }

  bool eval_Hess_Lagr(const long long& n, const long long& m,
                      const double* x, bool new_x, const double& obj_factor,
                      const double* lambda, bool new_lambda,
                      const int& nnzHSS, int* iHSS, int* jHSS, double* MHSS)
  {
    if(!Ex6::eval_Hess_Lagr(n, m, x, new_x, obj_factor, lambda, new_lambda,
                            nnzHSS, iHSS, jHSS, MHSS)) {
      return false;
    }
    //add diagonal to the Hessian

    if(iHSS!=nullptr && jHSS!=nullptr) {
      //nothing to do as the parent already added (i,i) corresponding to the diagonal
    }

    if(MHSS!=nullptr) {
      //use rec_evaluator_ to add diagonal entries in the Hessian
      assert(nnzHSS == n);
      for(int i=0; i<n; i++) {
        assert(false && "to be implemented");
      }
    }
    return true;
  }

  bool set_quadratic_terms(const int& n, const RecourseApproxEvaluator* evaluator)
  {
    assert(false && "to be implemented");
    return true;
  }
protected:
  RecourseApproxEvaluator* rec_evaluator_;
};
