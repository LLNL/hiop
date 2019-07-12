
#define NLP_CLASS_IN Ipopt::TNLP

//hiop::hiopInterfaceDenseConstraints

class hiopAugLagrAdapter : public hiop::hiopInterfaceDenseConstraints
{
  hiopAugLagrAdapter(NLP_CLASS_IN* nlp_in_)
    : nlp_in(nlp_in_) {}
  virtual ~hiopAugLagrAdapter();

  bool get_prob_sizes(long long& n, long long& m)
  { n=n_vars; m=0; return true; }

  bool get_vars_info(const long long& n, double *xlow, double* xupp, NonlinearityType* type)
  {
    return true;
  }
  bool get_cons_info(const long long& m, double* clow, double* cupp, NonlinearityType* type)
  {
    assert(m==0);
    return true;
  }
  //this will be the augmented lagrangian function
  bool eval_f(const long long& n, const double* x_in, bool new_x, double& obj_value)
  {

    return true;
  }
  bool eval_grad_f(const long long& n, const double* x_in, bool new_x, double* gradf)
  {

    return true;
  }

  bool eval_cons(const long long& n, 
		 const long long& m,  
		 const long long& num_cons, const long long* idx_cons,
		 const double* x_in, bool new_x, double* cons)
  {

    if(0==num_cons) return true; //this may happen when Hiop asks for inequalities, which we don't have in this example
    return true;
  }

  bool eval_Jac_cons(const long long& n, const long long& m, 
		     const long long& num_cons, const long long* idx_cons,
                     const double* x_in, bool new_x, double** Jac) 
  {
    if(0==num_cons) return true; //this may happen when Hiop asks for inequalities, which w    return true;
  }

  bool get_starting_point(const long long &global_n, double* x0)
  {
    //call starting point from nlp_in
    return true;
  }

public:
  //AugLagr specific code
  void set_rho(const double& rho_in);

protected:
  //general nlp to be "adapted" to Augmented Lagrangian form
  //needed by HiOp's AugLagr solver
  Ipopt::TNLP* nlp_in;
  //hiop::hiopInterfaceDenseConstraints* nlp;

  //penalty parameter
  double rho;
  //multipliers vector
  double* lambda;
};
