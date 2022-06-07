#ifndef HIOP_PERTURB_PD_LINSSYS
#define HIOP_PERTURB_PD_LINSSYS

#include "hiopNlpFormulation.hpp"

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
  
  virtual ~hiopPDPerturbation() = default;

  /** Initializes and reinitializes object based on the 'options' parameters of the
   * 'nlp_' object.
   * Returns 'false' if something goes wrong, otherwise 'true'
   */
  bool initialize(hiopNlpFormulation* nlp);

  /** Set log-barrier mu. */
  inline void set_mu(const double& mu)
  {
    mu_ = mu;
  }

  /** Called when a new linear system is attempted to be factorized 
   */
  virtual bool compute_initial_deltas(double& delta_wx, double& delta_wd,
			      double& delta_cc, double& delta_cd);

  /** Method for correcting inertia */
  virtual bool compute_perturb_wrong_inertia(double& delta_wx,
                                             double& delta_wd,
                                             double& delta_cc,
                                             double& delta_cd);

  /** Method for correcting singular Jacobian 
   *  (follows Ipopt closely since the paper seems to be outdated)
   */
  virtual bool compute_perturb_singularity(double& delta_wx, double& delta_wd,
				     double& delta_cc, double& delta_cd);

  inline bool get_curr_perturbations(double& delta_wx, double& delta_wd,
				     double& delta_cc, double& delta_cd)
  {
    delta_wx = delta_wx_curr_;
    delta_wd = delta_wd_curr_;
    delta_cc = delta_cc_curr_;
    delta_cd = delta_cd_curr_;
    return true;
  }
protected:
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

protected: //methods
  /** Decides degeneracy @hess_degenerate_ and @jac_degenerate_ based on @deltas_test_type_ 
   *  when the @num_degen_iters_ > @num_degen_max_iters_
   */
  void update_degeneracy_type();
  
private: //methods
  /** Internal method implementing the computation of delta_w's to correct wrong inertia
   * 
   */
  bool guts_of_compute_perturb_wrong_inertia(double& delta_wx, double& delta_wd);

  double compute_delta_c(const double& mu) const;
};


class hiopPDPerturbationNormalEqn : public hiopPDPerturbation
{
public:
  hiopPDPerturbationNormalEqn()
    : hiopPDPerturbation(),
      delta_c_min_bar_(1e-20),
      delta_c_max_bar_(1e-2),
      kappa_c_plus_(10.)
  {
  }

  ~hiopPDPerturbationNormalEqn()
  {
  }

  /** Called when a new linear system is attempted to be factorized 
   */
  bool compute_initial_deltas(double& delta_wx,
                              double& delta_wd,
                              double& delta_cc,
                              double& delta_cd);

  /** Method for correcting inertia */
  bool compute_perturb_wrong_inertia(double& delta_wx, 
                                     double& delta_wd,
                                     double& delta_cc,
                                     double& delta_cd);
                            
  /** Method for correcting singular Jacobian 
   *  (follows Ipopt closely since the paper seems to be outdated)
   */  
  bool compute_perturb_singularity(double& delta_wx,
                                   double& delta_wd,
                                   double& delta_cc,
                                   double& delta_cd);

private: //methods
  /** 
   * Internal method implementing the computation of delta_w's to correct wrong inertia
   */
  bool compute_primal_perturb_impl(double& delta_wx, double& delta_wd);
                                      
  /** 
   * Internal method implementing the computation of delta_c's to correct wrong inertia
   */
  bool compute_dual_perturb_impl(const double& mu, double& delta_cc, double& delta_cd);

protected: //variables
  double delta_c_max_bar_;
  double delta_c_min_bar_;
  double kappa_c_plus_;
};

} //end of namespace
#endif
