#ifndef HIOP_PERTURB_PD_LINSSYS
#define HIOP_PERTURB_PD_LINSSYS

namespace hiop
{

class hiopPDPerturbation
{
public:
  /** Default constructor 
   * Provides complete initialization, but uses algorithmic parameters from the Ipopt
   * implementation paper. \ref initialize(hiopNlpFormulation*) should be used to initialize
   * this class based on HiOp option file or HiOp user-supplied runtime options.
   */
  hiopPDPerturbation()
    : delta_w_min_bar_(1e-20),
      delta_w_max_bar_(1e+40),
      delta_w_0_bar_(1e-4),
      kappa_w_minus_(1./3),
      kappa_w_plus_bar_(100.),
      kappa_w_plus_(8.),
      delta_c_bar_(1e-8),
      kappa_c_(0.25),
      delta_wx_curr_(0.),
      delta_wd_curr_(0.), 
      delta_cc_curr_(0.), 
      delta_cd_curr_(0.), 
      delta_wx_last_(0.), 
      delta_wd_last_(0.), 
      delta_cc_last_(0.), 
      delta_cd_last_(0.),
      hess_degenerate_(dtNotEstablished),
      jac_degenerate_(dtNotEstablished),
      num_degen_iters_(0),
      num_degen_max_iters_(3),
      deltas_test_type_(dttNoTest),
      mu_(1e-8)
      
  {
  }

  /** Initializes and reinitializes object based on the 'options' parameters of the
   * 'nlp_' object.
   * Returns 'false' if something goes wrong, otherwise 'true'
   */
  bool initialize(hiopNlpFormulation* nlp)
  {
    delta_w_min_bar_ = nlp->options->GetNumeric("delta_w_min_bar");
    delta_w_max_bar_ = nlp->options->GetNumeric("delta_w_max_bar");
    delta_w_0_bar_     = nlp->options->GetNumeric("delta_0_bar");
    kappa_w_minus_   = nlp->options->GetNumeric("kappa_w_minus");
    kappa_w_plus_bar_= nlp->options->GetNumeric("kappa_w_plus_bar");
    kappa_w_plus_    = nlp->options->GetNumeric("kappa_w_plus");
    delta_c_bar_     = nlp->options->GetNumeric("delta_c_bar");
    kappa_c_         = nlp->options->GetNumeric("kappa_c");

    delta_wx_curr_ = delta_wd_curr_ = 0.;
    delta_cc_curr_ = delta_cd_curr_ = 0.;

    delta_wx_last_ = delta_wd_last_ = 0.;
    delta_cc_last_ = delta_cd_last_;

    num_degen_iters_ = 0;

    deltas_test_type_ = dttNoTest;
    return true;
  }

  /** Set log-barrier mu. */
  inline void set_mu(const double& mu)
  {
    mu_ = mu;
  }

  /** Called when a new linear system is attempted to be factorized 
   */
  bool compute_initial_deltas(double& delta_wx, double& delta_wd,
			      double& delta_cc, double& delta_cd)
  {
    update_degeneracy_type();
      
    if(delta_wx_curr_>0.)
      delta_wx_last_ = delta_wx_curr_;
    if(delta_wd_curr_>0.)
      delta_wd_last_ = delta_wd_curr_;

    if(delta_cc_curr_>0.)
      delta_cc_last_ = delta_cc_curr_;
    if(delta_cd_curr_>0.)
      delta_cd_last_ = delta_cd_curr_;

    if(hess_degenerate_ == dtNotEstablished || jac_degenerate_ == dtNotEstablished) {
      deltas_test_type_ = dttDeltac0Deltaw0;
    } else {
      deltas_test_type_ = dttNoTest;
    }

    if(jac_degenerate_ == dtDegenerate) {
      delta_cc = delta_cd = compute_delta_c(mu_);
    } else {
      delta_cc = delta_cd = 0.;
    }
    delta_cc_curr_ = delta_cc;
    delta_cd_curr_ = delta_cd;

    
    if(hess_degenerate_ == dtDegenerate) {
      delta_wx_curr_ = delta_wd_curr_ = 0.;
      if(!guts_of_compute_perturb_wrong_inertia(delta_wx, delta_wd)) {
	return false;
      }
    } else {
      delta_wx = delta_wd = 0.;
    }

    delta_wx_curr_ = delta_wx;
    delta_wd_curr_ = delta_wd;
    return true;
  }

  /** Method for correcting inertia */
  bool compute_perturb_wrong_inertia(double& delta_wx, double& delta_wd,
				     double& delta_cc, double& delta_cd)
  {    
    update_degeneracy_type();

    assert(delta_wx_curr_ == delta_wd_curr_);
    assert(delta_cc_curr_ == delta_cd_curr_);
    
    delta_cc = delta_cc_curr_;
    delta_cd = delta_cd_curr_;

    bool ret = guts_of_compute_perturb_wrong_inertia(delta_wx, delta_wd);
    if(!ret && delta_cc==0.) {
      delta_wx_curr_ = delta_wd_curr_ = 0.;
      delta_cc_curr_ = delta_cd_curr_ = delta_c_bar_ * pow(mu_, kappa_c_);
      deltas_test_type_ = dttNoTest;
      if(hess_degenerate_ == dtDegenerate) {
        hess_degenerate_ = dtNotEstablished;
      }
      
      delta_cc = delta_cc_curr_;
      delta_cd = delta_cd_curr_;
      ret = guts_of_compute_perturb_wrong_inertia(delta_wx, delta_wd);
    }
    return ret;
  }

  /** Method for correcting singular Jacobian 
   *  (follows Ipopt closely since the paper seems to be outdated)
   */
  
  bool compute_perturb_singularity(double& delta_wx, double& delta_wd,
				     double& delta_cc, double& delta_cd)
  {    
    assert(delta_wx_curr_ == delta_wd_curr_);
    assert(delta_cc_curr_ == delta_cd_curr_);
    
    if (hess_degenerate_ == dtNotEstablished ||
        jac_degenerate_ == dtNotEstablished) {
      switch (deltas_test_type_) {
      case dttDeltac0Deltaw0:
	//this is the first call
        if (jac_degenerate_ == dtNotEstablished) {
          delta_cc_curr_ = delta_cd_curr_ = compute_delta_c(mu_);
          deltas_test_type_ = dttDeltacposDeltaw0;
        }
        else {
          assert(hess_degenerate_ == dtNotEstablished);
          if(!guts_of_compute_perturb_wrong_inertia(delta_wx, delta_wd))
            return false;

          assert(delta_cc == 0. && delta_cd == 0.);
          deltas_test_type_ = dttDeltac0Deltawpos;
        }
        break;
      case dttDeltacposDeltaw0:
        assert(delta_wx_curr_ == 0. && delta_cc_curr_ > 0.);
        assert(jac_degenerate_ == dtNotEstablished);
	delta_cd_curr_ = delta_cc_curr_ = 0.;
	if(!guts_of_compute_perturb_wrong_inertia(delta_wx, delta_wd)) {
	  return false;
	}
	deltas_test_type_ = dttDeltac0Deltawpos;
        break;
      case dttDeltac0Deltawpos:
        assert(delta_wx_curr_ > 0. && delta_cc_curr_ == 0.);
        delta_cc_curr_ = delta_cd_curr_ = compute_delta_c(mu_);
        if(!guts_of_compute_perturb_wrong_inertia(delta_wx, delta_wd)) {
          return false;
        }
        deltas_test_type_ = dttDeltacposDeltawpos;
        break;
      case dttDeltacposDeltawpos:
        if(!guts_of_compute_perturb_wrong_inertia(delta_wx, delta_wd))
	  return false;
        break;
      case dttNoTest:
        assert(false && "something went wrong - should not get here");
      }
    } else {
      if(delta_cc_curr_ > 0.) { 
        // If we already used a perturbation for the constraints, we do
        // the same thing as if we were encountering negative curvature
        if(!guts_of_compute_perturb_wrong_inertia(delta_wx, delta_wd)) {
	  //todo: need some error message (so that we know what and more important
	  //where something went wrong)
	  return false;
	}
      } else {
        // Otherwise we now perturb the Jacobian part 
        delta_cd_curr_ = delta_cc_curr_ = compute_delta_c(mu_);
      }
    }

    delta_wx = delta_wx_curr_;
    delta_wd = delta_wd_curr_;
    delta_cc = delta_cc_curr_;
    delta_cd = delta_cd_curr_;
    
    return true;
  }

  inline bool get_curr_perturbations(double& delta_wx, double& delta_wd,
				     double& delta_cc, double& delta_cd)
  {
    delta_wx = delta_wx_curr_;
    delta_wd = delta_wd_curr_;
    delta_cc = delta_cc_curr_;
    delta_cd = delta_cd_curr_;
    return true;
  }
private:
  /** Current and last perturbations, primal is split in x and d, dual in c and d. */
  double delta_wx_curr_, delta_wd_curr_;
  double delta_cc_curr_, delta_cd_curr_;

  double delta_wx_last_, delta_wd_last_;
  double delta_cc_last_, delta_cd_last_;

  /** Algorithmic parameters */

  /** Smallest possible perturbation for Hessian (for primal 'x' and 's' variables). */
  double delta_w_min_bar_;
  /** Maximal perturbation for for Hessian (for primal 'x' and 's' variables). */
  double delta_w_max_bar_;
  /** First trial value for delta_w perturbation. */
  double delta_w_0_bar_;
  /** Decrease factor for delta_w. */
  double kappa_w_minus_;
  /** Increase factor for delta_w for first required perturbation. */
  double kappa_w_plus_bar_;
  /** Increase factor for delta_w for later perturbations. */
  double kappa_w_plus_;
  
  /** Factor for regularization for potentially rank-deficient Jacobian. */
  double delta_c_bar_;
  /** Exponent of mu when computing regularization for Jacobian. */
  double kappa_c_;
  
  /** Degeneracy is handled as in Ipopt*/
  
  /** Type for degeneracy flags */
  enum DegeneracyType
  {
    dtNotEstablished,
    dtNotDegenerate,
    dtDegenerate
  };
  /** Structural degeneracy of the Hessian */
  DegeneracyType hess_degenerate_;

  /** Structural degeneracy of the Jacobian */
  DegeneracyType jac_degenerate_;

  /* Counter for inertia corrections at which the matrix is suspected to be degenerate. */
  int num_degen_iters_;
  /* Max number of iters after which to conclude matrix is degenerate. */
  const int num_degen_max_iters_;
  
  /** Status of current trial configuration */
  enum DeltasTestType
  {
    dttNoTest,
    dttDeltac0Deltaw0,
    dttDeltacposDeltaw0,
    dttDeltac0Deltawpos,
    dttDeltacposDeltawpos
  };

  /** Current status */
  DeltasTestType deltas_test_type_;
  
  /** Log barrier mu in the outer loop. */
  double mu_;
private: //methods
  /** Decides degeneracy @hess_degenerate_ and @jac_degenerate_ based on @deltas_test_type_ 
   *  when the @num_degen_iters_ > @num_degen_max_iters_
   */
  void update_degeneracy_type()
  {
   switch (deltas_test_type_) {
   case dttNoTest:
     return;
   case dttDeltac0Deltaw0:
     if(hess_degenerate_ == dtNotEstablished &&
	 jac_degenerate_ == dtNotEstablished) {
       hess_degenerate_ = dtNotDegenerate;
       jac_degenerate_ = dtNotDegenerate;
     } else if(hess_degenerate_ == dtNotEstablished) {
       hess_degenerate_ = dtNotDegenerate;
     } else if(jac_degenerate_ == dtNotEstablished) {
       jac_degenerate_ = dtNotDegenerate;
     }
     break;
   case dttDeltacposDeltaw0:
     if(hess_degenerate_ == dtNotEstablished) {
       hess_degenerate_ = dtNotDegenerate;
     }
     
     if(jac_degenerate_ == dtNotEstablished) {
       num_degen_iters_++;
       if(num_degen_iters_ >= num_degen_max_iters_) {
	 jac_degenerate_ = dtDegenerate;
       }
     }
     break;
   case dttDeltac0Deltawpos:
     if(jac_degenerate_ == dtNotEstablished) {
        jac_degenerate_ = dtNotDegenerate;
	
     }
     if(hess_degenerate_ == dtNotEstablished) {
       num_degen_iters_++;
       if(num_degen_iters_ >= num_degen_max_iters_) {
	 hess_degenerate_ = dtDegenerate;
       }
     }
     break;
   case dttDeltacposDeltawpos:
     num_degen_iters_++;
     if(num_degen_iters_ >= num_degen_max_iters_) {
       hess_degenerate_ = dtDegenerate;
       jac_degenerate_ = dtDegenerate;
     }
     break;
   }
  }
  
  /** Internal method implementing the computation of delta_w's to correct wrong inertia
   * 
   */
  bool guts_of_compute_perturb_wrong_inertia(double& delta_wx, double& delta_wd)
  {
    assert(delta_wx_curr_ == delta_wd_curr_ && "these should be equal");
    assert(delta_wx_last_ == delta_wd_last_ && "these should be equal");
    if(delta_wx_curr_ == 0.) {
      if(delta_wx_last_ == 0.) {
	delta_wx_curr_ = delta_w_0_bar_;
      } else {
	delta_wx_curr_ = std::max(delta_w_min_bar_, delta_wx_last_*kappa_w_minus_);
      }
    } else { //delta_wx_curr_ != 0.
      if(delta_wx_last_==0. || 1e5*delta_wx_last_<delta_wx_curr_) {
	delta_wx_curr_ = kappa_w_plus_bar_ * delta_wx_curr_;
      } else {
	delta_wx_curr_ = kappa_w_plus_ * delta_wx_curr_;
      }
    }

    if(delta_wx_curr_ > delta_w_max_bar_) {
      //Hessian perturbation becoming too large
      delta_wx_last_ = delta_wd_last_ = 0.;
      return false;
    }

    delta_wd_curr_ = delta_wx_curr_ ;

    delta_wx = delta_wx_curr_;
    delta_wd = delta_wd_curr_;

    return true;
  }

  inline double compute_delta_c(const double& mu) const
  {
    return delta_c_bar_ * pow(mu, kappa_c_);
  }
};

  
} //end of namespace
#endif
